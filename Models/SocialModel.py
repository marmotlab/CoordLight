import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_(m):
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=nn.init.calculate_gain('relu'))


def init_rnn_(rnn):
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)
    return rnn


class Normalization(nn.Module):
    """
    Layer normalization module from https://arxiv.org/abs/1607.06450
    """
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalizer(x.reshape(-1, x.size(-1))).view(*x.size())


class ActorNet(nn.Module):
    def __init__(self, input_dim, output_dim, agent_dim, hidden_dim=128):
        super(ActorNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_dim = agent_dim

        self.feat_dim = self.input_dim[1]
        self.flat_dim = self.input_dim[0] * self.input_dim[1]

        self.normalization = Normalization(self.feat_dim)
        self.fc = nn.Sequential(
            init_(nn.Linear(self.flat_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )

        self.pred_layer = init_(nn.Linear(hidden_dim, 24))
        self.policy_layer = init_(nn.Linear(hidden_dim, self.output_dim))

    def forward(self, x, h_n, num_meta):
        # [b*n*5, feat_dim]
        x = x.reshape(-1, self.feat_dim)
        x = self.normalization(x)
        # [b, n, flat_feat]
        x = x.reshape(-1, self.agent_dim * num_meta, self.flat_dim)
        # [b, n, 128]
        x = self.fc(x)

        # [b, n, output_dim]
        policy = F.softmax(self.policy_layer(x), dim=-1)
        pred = self.pred_layer(x)

        return policy, pred, h_n


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, agent_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_dim = agent_dim

        self.neighbor_agent_dim = self.input_dim[0]
        self.neighbor_action_dim = self.input_dim[0] - 1
        self.feat_dim = self.input_dim[1]
        self.flat_dim = self.input_dim[0] * self.input_dim[1]

        self.normalization = Normalization(self.feat_dim)
        self.fc = nn.Sequential(
            init_(nn.Linear(self.flat_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )

        self.fc_q_s_all = nn.Sequential(
            nn.Linear(hidden_dim + 4 * (self.output_dim + 1) + self.flat_dim, hidden_dim),
            nn.ReLU()
        )

        self.pred_layer = init_(nn.Linear(hidden_dim, 24))
        self.value_layer = init_(nn.Linear(hidden_dim, self.output_dim))

    def forward(self, s, a, h_n, num_meta):
        # [b*n*5, feat_dim]
        s = s.reshape(-1, self.feat_dim)
        s = self.normalization(s)
        # [b, n, flat_dim]
        s = s.reshape(-1, self.agent_dim * num_meta, self.flat_dim)
        # [b, n, 128]
        h = self.fc(s)

        # [b * n, 4]
        a = a.view(-1, 4)
        # [b * n, 4, output_dim+1]
        a = F.one_hot(a, self.output_dim + 1)
        # [b, n, 4 * (output_dim+1)]
        a = a.reshape(-1, self.agent_dim * num_meta, self.neighbor_action_dim * (self.output_dim + 1))

        # cat all states [b, n, flat_dim + 4 * (output_dim+1) + 128] -> [b, n, 128]
        s_prime = torch.cat((h, a, s), dim=-1)
        s_prime = self.fc_q_s_all(s_prime)

        # [b, output_dim]
        q = self.value_layer(s_prime)
        pred = self.pred_layer(s_prime)

        return q, pred, h_n


class SocialLightModel(nn.Module):
    def __init__(self, input_dim, output_dim, agent_dim, actor_lr, critic_lr):
        super(SocialLightModel, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_dim = agent_dim

        self.actor_network = ActorNet(input_dim=self.input_dim,
                                      output_dim=self.output_dim,
                                      agent_dim=self.agent_dim)
        self.critic_network = CriticNetwork(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            agent_dim=self.agent_dim)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)

    def reset_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr, eps=1e-5)

    def forward(self, x, h_n, mask, num_meta=1):
        policy, pred, h_n = self.actor_network(x, h_n, num_meta)
        return policy, pred, h_n

    def forward_v(self, s, a, h_n, mask, num_meta=1):
        value, pred, h_n = self.critic_network.forward(s, a, h_n, num_meta)

        return value, pred, h_n


if __name__ == '__main__':
    s_ = torch.randn((10, 25, 5, 20))
    a_ = torch.randint(0, 8, (10, 25, 4))
    m_ = torch.zeros((10, 25, 4))

    model = SocialLightModel(input_dim=[5, 20], output_dim=8, agent_dim=25, actor_lr=1e-5, critic_lr=1e-5)
    pi, pi_pred, _ = model(s_, None, m_)
    q, q_pred, _ = model.forward_v(s_, a_, None, m_)
    print(pi.shape, pi_pred.shape)
    print(q.shape, q_pred.shape)
