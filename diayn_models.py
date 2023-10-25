import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical

def init_weight(layer, initializer="he normal"):
    if initializer == "xavier uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif initializer == "he normal":
        nn.init.kaiming_normal_(layer.weight)


class Discriminator(nn.Module):
    def __init__(self, n_skills, n_hidden_filters=256):
        super(Discriminator, self).__init__()
        self.n_skills = n_skills
        self.n_hidden_filters = n_hidden_filters
        ###add the CNN to extract features from the image. Pytorch is using default paramters to initialize this
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        ##############
        self.image_embedding_size=64
        self.hidden1 = nn.Linear(in_features=self.image_embedding_size, out_features=self.n_hidden_filters)
        init_weight(self.hidden1)
        self.hidden1.bias.data.zero_()
        self.hidden2 = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_hidden_filters)
        init_weight(self.hidden2)
        self.hidden2.bias.data.zero_()
        self.q = nn.Linear(in_features=self.n_hidden_filters, out_features=self.n_skills)
        init_weight(self.q, initializer="xavier uniform")
        self.q.bias.data.zero_()

    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        #print('x',x.shape)
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)
        x = F.relu(self.hidden1(embedding))
        x = F.relu(self.hidden2(x))
        unnormalized_probs = self.q(x)
        #pred_dist = Categorical(logits=F.log_softmax(unnormalized_probs, dim=1))
        return unnormalized_probs #was unnormalized probs
