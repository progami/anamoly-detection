# ---
# imports
# ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random 
from sortedcontainers import SortedDict

from config.search_space import *
from config.controller_params import *
from models.autoencoder import Architecture


# ---
# LSTM controller definition
# ---
class Controller(nn.Module):
    """LSTM controller that generates autoencoder reconstruction architectures for time series anomaly detection."""

    def __init__(self, vocab_size, hidden_size, num_layers, embedding_size):
        super(Controller, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers)
        # seperate output layer for every decision to take (different dimension and logic for each decision)
        self.conv_n_out_channels = nn.Linear(hidden_size, len(conv_n_out_channels))
        self.conv_kernel_size = nn.Linear(hidden_size, len(conv_kernel_size))
        self.pool_type = nn.Linear(hidden_size, len(pool_type))
        self.pool_kernel_size = nn.Linear(hidden_size, len(pool_kernel_size))
        self.norm_type = nn.Linear(hidden_size, len(norm_type))
        self.act_funcs = nn.Linear(hidden_size, len(act_funcs))

    def forward(self, input_seq, out_type):
        embedded = self.embedding(input_seq.clone()) # clone to avoid in-place operations
        lstm_out, (c_t, h_t) = self.lstm(embedded)
        last_seq = lstm_out[-1] # many to one architecture, only take last timestep
        if out_type=="conv_n_out_channels": output_seq = self.conv_n_out_channels(last_seq)
        elif out_type=="conv_kernel_size": output_seq = self.conv_kernel_size(last_seq)
        elif out_type=="pool_type": output_seq = self.pool_type(last_seq)
        elif out_type=="pool_kernel_size": output_seq = self.pool_kernel_size(last_seq)
        elif out_type=="norm_type": output_seq = self.norm_type(last_seq)
        elif out_type=="act_funcs": output_seq = self.act_funcs(last_seq)
        probs = nn.Softmax(dim=0)(output_seq)
        return probs


class ModelsHistory():
    """Stores past architectures to be fed to the controller."""

    def __init__(self, n_models, max_size=20):
        self.max_size = max_size
        self.n_models = n_models
        self.buffer = SortedDict()

    def add(self, fitness, architecture):
        """Adds architecture to history, keeping them sorted by fitness."""
        self.buffer.update({fitness: architecture})
        while len(self.buffer)>self.max_size:
            self.buffer.popitem(index=0)

    def sample(self, k):
        """Samples top k architectures found so far."""
        archs = self.buffer.values()[-k:]
        return archs 
    
    def init_history(self):
        """Initializes history with random architectures. will be iteratively replaced by the best ones found by the agent."""
        r = 0.0
        for _ in range(self.n_models-1):
            r += 0.01
            model_arch = []
            for i in range(number_blocks):
                model_arch.append(random.randint(0, len(conv_n_out_channels)-1))
                model_arch.append(random.randint(0, len(conv_kernel_size)-1))
                model_arch.append(random.randint(0, len(pool_type)-1))
                model_arch.append(random.randint(0, len(pool_kernel_size)-1))
                model_arch.append(random.randint(0, len(norm_type)-1))
                model_arch.append(random.randint(0, len(act_funcs)-1))
            self.add(r, model_arch)
        return self.build_history()
    
    def build_history(self):
        """Builds history tensor to be fed to the controller from the top k architectures so far."""
        archs = self.sample(self.n_models-1)
        history = []
        for arch in archs:
            history += arch
        history += [0 for _ in range(number_features_per_block*number_blocks)] # 0 pad for tokens not decided yet
        return torch.tensor(history)


class PolicyGradientAgent:
    """Policy gradient agent that learns to generate autoencoder reconstruction architectures
    for time series anomaly detection. The agent is trained with the REINFORCE algorithm."""

    def __init__(self, learning_rate, gamma, train_dataset_path, test_dataset_path):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.train_dataset_path = train_dataset_path
        self.test_dataset_path = test_dataset_path
        self.policy_network = Controller(vocab_size, hidden_size, num_layers, embedding_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

    def get_action(self, state=None):
        input_seq = state
        layers= ["conv_n_out_channels", "conv_kernel_size", "pool_type", "pool_kernel_size", "norm_type", "act_funcs"]*number_blocks
        arch = []
        choices_logprob = torch.empty(size=(1, len(layers)))[0]
        for i, layer in enumerate(layers):
            probs = self.policy_network(input_seq, layer)
            dist = Categorical(probs=probs)
            choice = dist.sample()
            choice_logprob = dist.log_prob(choice)
            arch.append(choice)
            choices_logprob[i] = choice_logprob
            # print(layer, choice, choice_logprob)
            input_seq[len(state)-len(layers)+i] = choice.item()
        return arch, choices_logprob

    def calc_fitness_loss(self, rewards, log_probs):
        # based on the REINFORCE algorithm's loss
        Gt = 0
        discounted_rewards = torch.empty(size=(1, len(rewards)))[0]
        for t in reversed(range(len(rewards))):
            Gt = Gt * self.gamma + rewards[t]
            discounted_rewards[t] = Gt
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()).abs() / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
        out = - torch.mul(discounted_rewards, log_probs)
        return out.sum()

    def eval_arch(self, sequence, data_type="val", return_metrics=False):
        arch = Architecture(sequence, self.train_dataset_path, self.test_dataset_path) 
        arch.train()
        return arch.fitness(data_type, return_metrics)
    
    def update_policy(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
