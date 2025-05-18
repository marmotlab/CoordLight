import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def init(module, weight_init, bias_init, gain=1):
    """
    Initialize the parameters using orthogonal initialization.
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_(m):
    """
    Initialize the parameters of the linear layer using orthogonal initialization.
    """
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=nn.init.calculate_gain('relu'))


def init_rnn_(rnn):
    """
    Initialize the parameters of the recurrent layer using orthogonal initialization.
    """
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)
    return rnn


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module from https://arxiv.org/abs/1706.03762
    """
    def __init__(self, embedding_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q

        batch_size, target_size, input_dim = h.size()
        n_query = q.size(1)  # n_query = target_size in tsp
        # assert q.size(0) == batch_size
        # assert q.size(2) == input_dim
        # assert input_dim == self.input_dim

        h_flat = h.contiguous().view(-1, input_dim)  # (batch_size*graph_size)*input_dim
        q_flat = q.contiguous().view(-1, input_dim)  # (batch_size*n_query)*input_dim
        shape_v = (self.n_heads, batch_size, target_size, -1)
        shape_k = (self.n_heads, batch_size, target_size, -1)
        shape_q = (self.n_heads, batch_size, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(h_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(h_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim

        # n_heads*batch_size*n_query*targets_size
        U = torch.tensor(self.norm_factor) * torch.matmul(Q, K.transpose(2, 3))

        if mask is not None:
            mask = mask.view(1, batch_size, -1, target_size).expand_as(U)  # copy for n_heads times
            # U = U.masked_fill(mask == 1, -np.inf)
            U[mask.bool()] = -np.inf
        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        if mask is not None:
            attnc = attention.clone()
            attnc[mask.bool()] = 0
            attention = attnc

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size * n_query * n_heads * value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads * value_dim * embedding_dim
        ).view(batch_size, n_query, self.embedding_dim)

        return out  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    """
    Layer normalization module from https://arxiv.org/abs/1607.06450
    """
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalizer(x.reshape(-1, x.size(-1))).view(*x.size())


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        # self.normalization2 = Normalization(embedding_dim)
        # self.feedForward = nn.Sequential(init_(nn.Linear(embedding_dim, embedding_dim)),
        #                                  nn.ReLU(),
        #                                  init_(nn.Linear(embedding_dim, embedding_dim)))

    def forward(self, tgt, memory, mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h = self.multiHeadAttention(q=tgt, h=memory, mask=mask)
        h = h + h0
        # h1 = h
        # h = self.normalization2(h)
        # h = self.feedForward(h)
        # h2 = h + h1
        return h


class Decoder(nn.Module):
    def __init__(self, embedding_dim, n_head, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, mask)
        return tgt


class ActorNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, agent_dim, hidden_dim=128):
        super(ActorNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_dim = agent_dim
        self.hidden_dim = hidden_dim

        self.neighbor_agent_dim = self.input_dim[0]
        self.neighbor_action_dim = self.input_dim[0] - 1
        self.feat_dim = self.input_dim[1]
        self.flat_dim = self.input_dim[0] * self.input_dim[1]

        self.normalization = Normalization(self.feat_dim)
        self.fc_s = nn.Sequential(
            init_(nn.Linear(self.feat_dim, self.hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU()
        )
        self.fc_decoder = Decoder(embedding_dim=self.hidden_dim, n_head=4)
        self.gru = init_rnn_(nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=False))

        self.policy_layer = init_(nn.Linear(self.hidden_dim, self.output_dim))
        self.pred_layer = init_(nn.Linear(self.hidden_dim, 24))

    def forward_pred(self, x, h_n, mask, num_meta):
        # [b*n*5, feat_dim]
        x = x.reshape(-1, self.feat_dim)
        x = self.normalization(x)
        # [b, n, 5, feat_dim]
        x = x.reshape(-1, self.agent_dim * num_meta, self.neighbor_agent_dim, self.feat_dim)
        # [b*n, 4]
        mask = mask.reshape(-1, self.neighbor_action_dim)

        # State embedding [b, n, 5, hidden_dim]
        s = self.fc_s(x)
        # Spatial Aggregation Unit
        # Query vector
        # [b*n, 1, hidden_dim]
        q = s[:, :, 0, :].reshape(-1, 1, self.hidden_dim)
        # Key/Value Vector
        # [b*n, 4, hidden_dim]
        k = s[:, :, 1:, :].reshape(-1, self.neighbor_agent_dim - 1, self.hidden_dim)
        # [b*n, 1, hidden_dim]
        s = self.fc_decoder(q, k, mask)
        # [b, n, hidden_dim]
        s = s.reshape(-1, self.agent_dim * num_meta, self.hidden_dim)
        # Temporal Aggregation Unit
        # [b, n, hidden_dim] corresponds to [seq, batch, feat_dim]
        s_prime, h_n = self.gru(s, h_n)
        # Residual
        s = s + s_prime

        # Prediction output
        # [b, n, 24]
        pred = self.pred_layer(s)
        # Policy output
        # [b, n, 5]
        policy = F.softmax(self.policy_layer(s), dim=-1)

        return policy, pred, h_n


class CriticNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, agent_dim, hidden_dim=128):
        super(CriticNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_dim = agent_dim
        self.hidden_dim = hidden_dim

        self.neighbor_agent_dim = self.input_dim[0]
        self.neighbor_action_dim = self.input_dim[0] - 1
        self.feat_dim = self.input_dim[1]
        self.flat_dim = self.input_dim[0] * self.input_dim[1]

        self.normalization = Normalization(self.feat_dim)
        self.fc_s = nn.Sequential(
            init_(nn.Linear(self.feat_dim, self.hidden_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.hidden_dim, self.hidden_dim)),
            nn.ReLU()
        )

        self.fc_a = nn.Sequential(
            init_(nn.Linear(self.output_dim + self.neighbor_action_dim + 1, self.hidden_dim)),
            nn.ReLU(),
        )
        self.fc_s_decoder = Decoder(embedding_dim=self.hidden_dim, n_head=4)
        self.fc_a_decoder = Decoder(embedding_dim=self.hidden_dim, n_head=4)

        self.gru_s = init_rnn_(nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, batch_first=False))

        self.value_layer = init_(nn.Linear(self.hidden_dim, 1))
        self.pred_layer = init_(nn.Linear(self.hidden_dim, 24))

    def forward_pred(self, x, a, h_n, mask, num_meta):
        # # [b*n*5, feat_dim]
        x = x.reshape(-1, self.feat_dim)
        x = self.normalization(x)
        # [b, n, 5, feat_dim]
        x = x.reshape(-1, self.agent_dim * num_meta, self.neighbor_agent_dim, self.feat_dim)
        # [b*n, 4]
        mask = mask.reshape(-1, self.neighbor_action_dim)

        # State Encoder: Spatio-Temporal Network
        # State embedding [b, n, 5, hidden_dim]
        s = self.fc_s(x)
        # Query vector [b, n, 5, hidden_dim]
        # [b*n, 1, hidden_dim]
        q = s[:, :, 0, :].reshape(-1, 1, self.hidden_dim)
        # Key/Value Vector
        # [b*n, 4, hidden_dim]
        k = s[:, :, 1:, :].reshape(-1, self.neighbor_agent_dim - 1, self.hidden_dim)
        # [b*n, 1, hidden_dim]
        s = self.fc_s_decoder(q, k, mask)

        # Action Encoder
        # [b*n, 4]
        a = a.view(-1, self.neighbor_action_dim)
        # [b*n, 4, output_dim+1]
        a = F.one_hot(a, self.output_dim + 1)
        # [b*n, 4, output_dim+1+4]
        index = torch.eye(4).repeat(a.size(0), 1, 1)
        index = index.to(torch.device('cuda')) if a.get_device() != -1 else index.to(torch.device('cpu'))
        a = torch.cat((a, index), dim=-1)
        # [b*n, 4, hidden_dim]
        a = self.fc_a(a)
        # Action Decoder
        # [b*n, 1, hidden_dim]
        s = self.fc_a_decoder(s, a, mask)
        # [b, n, hidden_dim]
        s = s.reshape(-1, self.agent_dim * num_meta, self.hidden_dim)

        # GRU
        # [b, n, hidden_dim]
        s_prime, h_n[0] = self.gru_s(s, h_n[0])
        s = s + s_prime

        # Prediction output
        # [b, n, 1]
        pred = self.pred_layer(s)
        # Value output
        # [b, n, 1]
        value = self.value_layer(s)

        return value, pred, h_n


class CoordLight(nn.Module):
    def __init__(self, input_dim, output_dim, agent_dim, actor_lr, critic_lr):
        super(CoordLight, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.agent_dim = agent_dim

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_network = ActorNetwork(input_dim=self.input_dim,
                                          output_dim=self.output_dim,
                                          agent_dim=self.agent_dim)
        self.critic_network = CriticNetwork(input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            agent_dim=self.agent_dim)

        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=critic_lr)

    def reset_optimizer(self):
        self.actor_optimizer = optim.Adam(self.actor_network.parameters(), lr=self.actor_lr, eps=1e-5)
        self.critic_optimizer = optim.Adam(self.critic_network.parameters(), lr=self.critic_lr, eps=1e-5)

    def forward(self, x, h_n, mask, num_meta=1):
        policy, pred, h_n = self.actor_network.forward_pred(x, h_n, mask, num_meta)

        return policy, pred, h_n

    def forward_v(self, s, a, h_n, mask, num_meta=1):
        value, pred, h_n = self.critic_network.forward_pred(s, a, h_n, mask, num_meta)

        return value, pred, h_n


if __name__ == '__main__':
    x_ = torch.randn((10, 25, 5, 20))
    y_ = torch.randint(0, 8, (10, 25, 4))
    m_ = torch.zeros((10, 25, 4))
    model = CoordLight(input_dim=[5, 20], output_dim=8, agent_dim=25, actor_lr=1e-5, critic_lr=1e-5)
    pi, pi_pred, _ = model.forward(x_, None, m_)
    v, v_pred, _ = model.forward_v(x_, y_, [None, None], m_)
    print(pi.shape, v.shape)
    print(pi_pred.shape, v_pred.shape)
