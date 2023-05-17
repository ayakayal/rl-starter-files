import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
#init_params is not initializing the conv2d layers actually...
def init_params(m):
    
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
       

        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
        #print("weight data type", m.weight.data.dtype)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MinigridStateEmbeddingNet(nn.Module):
#64 is assumed to be the embedding size
    def __init__(self):
        super().__init__()
        #image embedding part of the network (I got it from ModelAC above)
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        # obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}
        # n = obs_space["image"][0]
        # m = obs_space["image"][1]
        # self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.apply(init_params) ##this is not doing anything; it is using pytorch default initialization for conv2D 

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        # -- [unroll_length x batch_size x height x width x channels]
        return x #.to('cuda')
# class MinigridStateEmbeddingNet(nn.Module):

#     def __init__(self, obs_space):
#         super().__init__()
#         #image embedding part of the network (I got it from ModelAC above)
#         init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
#                             constant_(x, 0), nn.init.calculate_gain('relu'))
#         self.image_conv = nn.Sequential(init_(
#             nn.Conv2d(3, 16, (2, 2))),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             init_(nn.Conv2d(16, 32, (2, 2))),
#             nn.ReLU(),
#             init_(nn.Conv2d(32, 64, (2, 2))),
#             nn.ReLU()
#         )
#         obs_space = {"image": obs_space.spaces["image"].shape, "text": 100}
#         n = obs_space["image"][0]
#         m = obs_space["image"][1]
#         self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        

#     def forward(self, obs):
#         x = obs.image.transpose(1, 3).transpose(2, 3)
#         x = self.image_conv(x)
#         x = x.reshape(x.shape[0], -1)
#         # -- [unroll_length x batch_size x height x width x channels]
#         return x#.to('cuda')
    
class MinigridForwardDynamicsNet(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.num_actions= action_space.n
        print('self.num_actions',self.num_actions)
        #self.image_embedding_size=image_embedding_size
        #print('self.image_embedding_size',self.image_embedding_size)
        #i got init_ function from RIDE
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                        constant_(x, 0), nn.init.calculate_gain('relu'))
        #image embedding part of the network (I got it from ModelAC above)
       
       

        #this part from RIDE
        self.forward_dynamics = nn.Sequential(
            init_(nn.Linear(64 + self.num_actions, 256)), 
            nn.ReLU(), 
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))

        self.fd_out = init_(nn.Linear(256,64))

    def forward(self, obs, action):

        #this part from my code
        #print('action',action)
        #print('length',len(action))
        one_hot = torch.zeros(len(action),self.num_actions).to('cuda') #,device='cuda:5'
        for i in range(len(action)):

            #print('hey',action[i])
            action_tensor=torch.tensor([action[i].item()]).long()#, dtype=torch.int32).long() #, device='cuda:5'
            #print('action tensor',action_tensor)
            one_hot[i] = torch.nn.functional.one_hot(action_tensor, num_classes=self.num_actions).to('cuda')
            #one_hot[i]= F.one_hot(action[i], num_classes=self.num_actions).float()
       
        #print('one hot',one_hot.shape)
        #one_hot.to('cuda:5')
        #action_one_hot = F.one_hot(action, num_classes=self.num_actions).float()
        #print('obs tens',obs.shape)
        #print('one hot shape',one_hot)
        inputs = torch.cat((obs, one_hot), dim=1)
        #print('inputs',inputs.shape)
        
        next_state_emb = self.fd_out(self.forward_dynamics(inputs))
        return next_state_emb
    
class MinigridInverseDynamicsNet(nn.Module):
     def __init__(self, action_space):
        super().__init__()
        self.num_actions= action_space.n
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))
        self.inverse_dynamics = nn.Sequential(
            init_(nn.Linear(2 * 64, 128)), 
            nn.ReLU(),  
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, 
            lambda x: nn.init.constant_(x, 0))
        self.id_out = init_(nn.Linear(128, self.num_actions))
     def forward(self, state_embedding, next_state_embedding):
        inputs = torch.cat((state_embedding, next_state_embedding), dim=1)
        action_prob = self.id_out(self.inverse_dynamics(inputs))
        pred_dist = Categorical(logits=F.log_softmax(action_prob, dim=1))
        return pred_dist
