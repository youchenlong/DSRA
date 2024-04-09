import torch as th
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.distributions import kl_divergence
# from tensorboardX import SummaryWriter
import numpy as np

from utils.sparsemax import Sparsemax

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, n_dim, softTemperature, dim_q=None, dim_k=None, dim_v=None, verbose=False, isSoftmax=False):
        super(MultiHeadAttention, self).__init__()
        assert (n_dim % n_heads) == 0, "n_heads must divide n_dim"
        attn_dim = n_dim // n_heads
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        self.verbose = verbose
        self.temperature=attn_dim ** 0.5 / softTemperature
        self.isSoftmax = isSoftmax
        if dim_q is None:
            dim_q = n_dim
        if dim_k is None:
            dim_k = dim_q
        if dim_v is None:
            dim_v = dim_k

        self.fc_q = nn.Linear(dim_q, n_dim, bias=False)
        self.fc_k = nn.Linear(dim_k, n_dim, bias=False)
        self.fc_v = nn.Linear(dim_v, n_dim)
        self.fc_final = nn.Linear(n_dim, n_dim)

    def forward(self, h_q, h_k, h_v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        bs = h_q.shape[0]
        q = self.fc_q(h_q).view(bs, -1, self.n_heads, self.attn_dim).transpose(1, 2)
        k_T = self.fc_k(h_k).view(bs, -1, self.n_heads, self.attn_dim).permute(0, 2, 3, 1)
        v = self.fc_v(h_v).view(bs, -1, self.n_heads, self.attn_dim).transpose(1, 2)
        alpha = th.matmul(q / self.temperature, k_T)
        if self.isSoftmax:
            alpha = F.softmax(alpha, dim=-1)
        else:
            sparsemax = Sparsemax(dim=-1)
            alpha = sparsemax(alpha)
        if self.verbose:
            assert self.n_heads == 1
            self.alpha = alpha.squeeze(2).detach()
        res = th.matmul(alpha, v).transpose(1, 2).reshape(bs, -1, self.attn_dim * self.n_heads)
        res = self.fc_final(res)
        return res, self.alpha

class DSRAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(DSRAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_subtasks = args.n_subtasks
        self.latent_dim = args.latent_dim
        self.bs = 0

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, self.n_subtasks*args.n_actions)

        self.fc1_ability = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn_ability = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2_ability = nn.Linear(args.rnn_hidden_dim, args.latent_dim) 
        self.atten = MultiHeadAttention(args.num_heads, self.n_subtasks, args.softTemperature, args.latent_dim, args.latent_dim, args.latent_dim, verbose=True, isSoftmax=args.isSoftmax)

        NN_HIDDEN_SIZE = args.NN_HIDDEN_SIZE
        self.embed_net = nn.Sequential(nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
                                       nn.BatchNorm1d(NN_HIDDEN_SIZE),
                                       nn.LeakyReLU(),
                                       nn.Linear(NN_HIDDEN_SIZE, self.n_subtasks * args.latent_dim * 2))
        self.latent = th.rand(args.n_agents, self.n_subtasks * args.latent_dim * 2) 
        self.decoder = nn.Sequential(nn.Linear(args.latent_dim, 32),
                                    nn.LeakyReLU(),
                                    nn.Linear(32, args.rnn_hidden_dim_subtask_))
        # self.ablation_embed_net = nn.Linear(args.rnn_hidden_dim, args.latent_dim, bias=False)
        self.ablation_embed_net = nn.Linear(self.n_subtasks, args.latent_dim, bias=False)

        self.state_dim = int(np.prod(args.state_shape))
        self.subtask_state_encoder = nn.Sequential(nn.Linear(self.state_dim, args.rnn_hidden_dim_subtask_),
                                                # nn.BatchNorm1d(args.rnn_hidden_dim_subtask_),
                                                nn.LeakyReLU(),
                                                nn.Linear(args.rnn_hidden_dim_subtask_, self.n_subtasks*args.rnn_hidden_dim_subtask_))
        self.subtask_state_decoder = nn.Sequential(nn.Linear(self.n_subtasks*args.rnn_hidden_dim_subtask_, args.rnn_hidden_dim_subtask_),
                                                nn.LeakyReLU(),
                                                nn.Linear(args.rnn_hidden_dim_subtask_, self.state_dim))

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_hidden_ability(self):
        return self.fc1_ability.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_latent(self, bs):
        self.bs = bs
        return None

    def forward(self, inputs, states, hidden_states, hidden_states_ability, train_mode=False):
        # inputs: [bs*n_agent, input_shape]
        # states: [bs, state_dim]
        # hidden_states: [bs*n_agent, rnn_hidden_dim]
        # hidden_states_ability: [bs*n_agent, rnn_hidden_dim]

        x = F.relu(self.fc1(inputs)) 
        h_in = hidden_states.view(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in) # [bs*n_agent, rnn_hidden_dim]

        # subtask representation
        self.latent = self.embed_net(h) # [bs*n_agent, n_subtask * latent_dim * 2]
        if train_mode:
            self.latent[:, -self.n_subtasks * self.latent_dim:] = th.clamp(th.exp(self.latent[:, -self.n_subtasks * self.latent_dim:]), min=self.args.var_floor)
            latent_embed = self.latent.reshape(self.bs * self.n_agents, self.n_subtasks * self.latent_dim * 2)
            gaussian_embed = D.Normal(latent_embed[:, :self.n_subtasks * self.latent_dim],
                                    (latent_embed[:, self.n_subtasks * self.latent_dim:]) ** (1 / 2))
            subtask_latent_embed = gaussian_embed.rsample() # [bs*n_agent, n_subtask*latent_dim]
        else:
            latent_embed = self.latent.reshape(self.bs * self.n_agents, self.n_subtasks * self.latent_dim * 2)
            subtask_latent_embed = latent_embed[:, :self.n_subtasks * self.latent_dim]  
        if self.args.ablation_representation:
            # subtask_latent_embed = self.ablation_embed_net(h) # [bs*n_agent, latent_dim]
            # subtask_latent_embed = subtask_latent_embed.unsqueeze(1).expand(-1, self.n_subtasks, -1).reshape(-1, self.n_subtasks*self.latent_dim) # [bs*n_agent, n_subtask*latent_dim]
            subtask_onehot = th.eye(self.n_subtasks).unsqueeze(0).expand(self.bs, -1, -1).to(self.args.device)
            subtask_latent_embed = self.ablation_embed_net(subtask_onehot)
            subtask_latent_embed = subtask_latent_embed.reshape(-1, self.n_subtasks*self.latent_dim).unsqueeze(1).expand(-1, self.n_agents, self.n_subtasks*self.latent_dim)

        # agent ability
        x_ability = F.relu(self.fc1_ability(inputs)) # [bs*n_agent, rnn_hidden_dim]
        h_in_ability = hidden_states_ability.reshape(-1, self.args.rnn_hidden_dim) # [bs*n_agent, rnn_hidden_dim]
        h_ability = self.rnn_ability(x_ability, h_in_ability) # [bs*n_agent, rnn_hidden_dim]
        ability_embed = self.fc2_ability(h_ability) # [bs*n_agent, latent_dim]

        # subtask policy
        q = self.fc2(h) # [bs*n_agent, n_subtask*n_action]
        q = q.reshape(-1, self.n_subtasks, self.args.n_actions) # [bs*n_agent, n_subtask, n_action]

        # subtask selection
        # ability_embed = ability_embed.unsqueeze(1).expand(-1, self.n_subtasks, -1).reshape(-1, self.latent_dim)  # [bs*n_agent*n_subtask, latent_dim]
        # subtask_latent_embed = subtask_latent_embed.reshape(-1, self.latent_dim)  # [bs*n_agent*n_subtask, latent_dim]
        # subtask_prob_logit = F.cosine_similarity(ability_embed, subtask_latent_embed)  # [bs*n_agent*n_subtask]
        ability_embed = ability_embed.unsqueeze(1) # [bs*n_agent, 1, latent_dim]
        subtask_latent_embed = subtask_latent_embed.reshape(-1, self.n_subtasks, self.latent_dim) # [bs*n_agent, n_subtask, latent_dim]
        _, subtask_prob_logit = self.atten(ability_embed, subtask_latent_embed, subtask_latent_embed) # [bs*n_agent, 1, n_subtask]
        subtask_prob_logit = subtask_prob_logit.reshape(-1, self.n_agents, self.n_subtasks) # [bs, n_agent, n_subtask]
        if not train_mode and self.args.test_argmax:
            prob_max = th.max(subtask_prob_logit, dim=-1, keepdim=True)[1] # [bs, n_agent, 1]
            subtask_prob = th.zeros_like(subtask_prob_logit).scatter_(-1, prob_max, 1) # [bs, n_agent, n_subtask]
        else:
            if self.args.sft_way == "softmax":
                subtask_prob = F.softmax(subtask_prob_logit, dim=-1) # [bs, n_agent, n_subtask]
            elif self.args.sft_way == "gumbel_softmax":
                subtask_prob = F.gumbel_softmax(subtask_prob_logit, hard=True, dim=-1) # [bs, n_agent, n_subtask]
        subtask_prob = subtask_prob.reshape(-1, 1, self.n_subtasks) # [bs*n_agent, 1, n_subtask]
        if self.args.ablation_selection:
            subtask_prob = th.rand(self.bs, self.n_agents, self.n_subtasks) # [bs, n_agent, n_subtask]
            subtask_prob = F.softmax(subtask_prob, dim=-1) # [bs, n_agent, n_subtask]
            subtask_prob = subtask_prob.reshape(-1, 1, self.n_subtasks).to(self.args.device) # [bs*n_agent, 1, n_subtask]
        if self.args.evaluate:
            # print('chosen_subtasksproblogit', subtask_prob_logit.reshape(self.n_agents, self.n_subtasks))
            # print('chosen_subtasksprob', subtask_prob.reshape(self.n_agents, self.n_subtasks))
            self.subtask_prob_logit = subtask_prob_logit.clone().detach().cpu().numpy().tolist()
            self.subtask_prob = subtask_prob.clone().detach().cpu().numpy().tolist()
            self.subtask_latent_embed = subtask_latent_embed.clone().detach().cpu().numpy().tolist()
        q = th.bmm(subtask_prob, q).squeeze(1) # [bs*n_agent, n_action]

        # regularizer
        sim_loss = th.tensor(0.0).to(self.args.device)
        recon_loss = th.tensor(0.0).to(self.args.device)
        state_loss = th.tensor(0.0).to(self.args.device)
        sel_loss = th.tensor(0.0).to(self.args.device)
        reg_loss = th.tensor(0.0).to(self.args.device)
        if train_mode:
            # similarity loss
            standard_n_dist = D.Normal(th.zeros_like(latent_embed[:, :self.n_subtasks * self.latent_dim]), 
                                       th.ones_like(latent_embed[:, self.n_subtasks * self.latent_dim:]))
            sim_loss += kl_divergence(gaussian_embed, standard_n_dist).sum(-1).mean()
            sim_loss *= self.args.vae_beta
            # reconstruction loss
            subtask_latent_embed = subtask_latent_embed.reshape(-1, self.latent_dim) # [bs*n_agent*n_subtask, latent_dim]
            s_hat = self.decoder(subtask_latent_embed) # [bs*n_agent*n_subtask, rnn_hidden_dim_subtask]
            # subtask_states = self.subtask_state_encoder(states) # [bs, n_subtask*rnn_hidden_dim_subtask]
            # s_true = subtask_states.clone().detach().reshape(-1, self.args.rnn_hidden_dim_subtask_) # [bs*n_subtask, rnn_hidden_dim_subtask]
            # s_true = s_true.repeat(1, self.n_agents, 1) # [1, bs*n_subtask*n_agent, rnn_hidden_dim_subtask]
            # s_true = s_true.view(self.bs * self.n_agents * self.n_subtasks, -1) # [bs*n_agent*n_subtask, rnn_hidden_dim_subtask]
            subtask_states = self.subtask_state_encoder(states) # [bs, n_subtask*rnn_hidden_dim_subtask]
            s_true = subtask_states.clone().detach().repeat(self.n_agents, 1).reshape(-1, self.args.rnn_hidden_dim_subtask_) # [bs*n_subtask*n_agent, rnn_hidden_dim_subtask]
            recon_loss += F.mse_loss(s_hat, s_true)          
            # state recon loss
            _states = self.subtask_state_decoder(subtask_states)
            # rho = torch.FloatTensor([0.01 for _ in range(self.args.rnn_hidden_dim_subtask_)]).unsqueeze(0)
            # rho_hat = torch.sum(s_true, dim=0, keepdim=True)
            # state_loss += F.mse_loss(_states, states) + self.args.vae_beta * F.kl_div(F.softmax(rho), F.softmax(rho_hat))
            state_loss += F.mse_loss(_states, states)
            # subtask selection loss
            subtask_prob_logit = subtask_prob_logit.reshape(-1, self.n_subtasks) # [bs*n_agent, n_subtask]
            subtask_prob_logit = th.clamp(subtask_prob_logit, 1e-4)
            sel_loss = -(subtask_prob_logit*th.log2(subtask_prob_logit)).sum(-1).mean()
        sim_loss *= self.args.sim_loss_weight
        recon_loss *= self.args.recon_loss_weight
        state_loss *= self.args.state_loss_weight
        sel_loss *= self.args.sel_loss_weight
        reg_loss = recon_loss + sim_loss + state_loss + sel_loss

        return q, h, h_ability, {"reg_loss": reg_loss, "sim_loss": sim_loss, "recon_loss": recon_loss, "state_loss": state_loss, "sel_loss": sel_loss}