import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        #print('oui')
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
        #print("weight data type", m.weight.data.dtype)


class ACModel(nn.Module, torch_ac.RecurrentACModel): #change it back to RecurrentACModel
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, use_diayn=False,num_skills=10):
        super().__init__()
        #print('pedro')
        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.use_diayn= use_diayn
        self.num_skills=num_skills

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)
        print('self.use_diayn',self.use_diayn)
        if self.use_diayn:
            self.image_embedding_size+= self.num_skills

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory,skill=None): #put skill here and fix
        #print('skill here',skill)
        x = obs.image.transpose(1, 3).transpose(2, 3)
        #print('x',x.shape)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        #print("image before LSTM", x.shape)
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
          
            memory = torch.cat(hidden, dim=1)
           
        else:
            embedding = x
     

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        if self.use_diayn:
            #print('skill',skill)
            #
            #print('embedding',embedding.shape)
            embedding= torch.cat((embedding,skill),dim=1)
            #print('embedding shape',embedding.shape)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

# def init(module, weight_init, bias_init, gain=1):
#     weight_init(module.weight.data, gain=gain)
#     bias_init(module.bias.data)
#     return module



    

# class MinigridForwardDynamicsNet(nn.Module):
#     def __init__(self, obs_space, action_space):
#         super().__init__()
#         self.num_actions= action_space.n

#         #i got init_ function from RIDE
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                         constant_(x, 0), nn.init.calculate_gain('relu'))
#         #image embedding part of the network (I got it from ModelAC above)
#         self.image_conv = nn.Sequential(
#             nn.Conv2d(3, 16, (2, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU()
#         )

#         n = obs_space["image"][0]
#         m = obs_space["image"][1]
#         self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
#         self.embedding_size= self.image_embedding_size + self.num_actions

#         #this part from RIDE
#         self.forward_dynamics = nn.Sequential(
#             init_(nn.Linear(self.image_embedding_size + self.num_actions, 256)), 
#             nn.ReLU(), 
#         )

#         init_ = lambda m: init(m, nn.init.orthogonal_, 
#             lambda x: nn.init.constant_(x, 0))

#         self.fd_out = init_(nn.Linear(256, 64))

#     def forward(self, obs, action):

#         #this part from my code
#         x = obs.image.transpose(1, 3).transpose(2, 3)
#         x = self.image_conv(x)
#         x = x.reshape(x.shape[0], -1)

#         action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
#         inputs = torch.cat((x, action_one_hot), dim=2)
        
#         next_state_emb = self.fd_out(self.forward_dynamics(inputs))
#         return next_state_emb
# class MinigridStateEmbeddingNet(nn.Module):

#     def __init__(self, obs_space):
#         super().__init__()
#         print('pedro')
#         #image embedding part of the network (I got it from ModelAC above)
#         self.image_conv = nn.Sequential(
#             nn.Conv2d(3, 16, (2, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU()
#         )
#         n = obs_space["image"][0]
#         m = obs_space["image"][1]
#         self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
#         self.apply(init_params)

#     def forward(self, obs):
#         x = obs.image.transpose(1, 3).transpose(2, 3)
#         x = self.image_conv(x)
#         x = x.reshape(x.shape[0], -1)
#         # -- [unroll_length x batch_size x height x width x channels]
#         return x
    
# class MinigridForwardDynamicsNet(nn.Module):
#     def __init__(self, image_embedding_size, action_space):
#         super().__init__()
#         self.num_actions= action_space.n
#         self.image_embedding_size=image_embedding_size
#         #i got init_ function from RIDE
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                         constant_(x, 0), nn.init.calculate_gain('relu'))
#         #image embedding part of the network (I got it from ModelAC above)
       
       

#         #this part from RIDE
#         self.forward_dynamics = nn.Sequential(
#             init_(nn.Linear(64 + self.num_actions, 256)), 
#             nn.ReLU(), 
#         )

#         init_ = lambda m: init(m, nn.init.orthogonal_, 
#             lambda x: nn.init.constant_(x, 0))

#         self.fd_out = init_(nn.Linear(256,64))

#     def forward(self, obs, action):

#         #this part from my code

#         action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
#         inputs = torch.cat((obs, action_one_hot), dim=2)
        
#         next_state_emb = self.fd_out(self.forward_dynamics(inputs))
#         return next_state_emb
     
