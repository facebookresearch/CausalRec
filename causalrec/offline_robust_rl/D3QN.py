# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import torch
import torch.nn as nn
import numpy as np
import copy


class Dueling_Net(nn.Module):
    def __init__(self, args):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.V = nn.Linear(args.hidden_dim, 1)
        self.A = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.max(A, dim=-1, keepdim=True)[0])  # Q(s,a) = V(s) + A(s,a) - max_b A(s,b) => Q(s. a*) = V(s)
        return Q

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename, device='cpu'):
        self.to(device)
        self.load_state_dict(torch.load(filename, map_location=torch.device(device)))


class D3QN(object):
    def __init__(self, args):
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size  # batch size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # learning rate
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Soft update
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq  # hard update
        self.update_count = 0
        self.weight_reg = args.weight_reg

        self.grad_clip = args.grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_double = args.use_double

        self.net = Dueling_Net(args)
        self.target_net = copy.deepcopy(self.net)  # Copy the online_net to the target_net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def choose_action(self, state, epsilon):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            q = self.net(state)
            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action

    def learn(self, replay_buffer, total_steps):
        batch, batch_index = replay_buffer.sample()

        with torch.no_grad():  # q_target has no gradient
            if self.use_double:  # Whether to use the 'double q-learning'
                # Use online_net to select the action
                a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                # Use target_net to estimate the q_target
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)
            else:
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(batch['next_state']).max(dim=-1)[0]  # shape：(batch_size,)

            # IPM uncertainty set
            weight_norm, bias_norm = [], []
            for layer in self.target_net.children():
                if isinstance(layer, nn.Linear) and layer.state_dict()['bias'].numpy().shape[0] != self.action_dim:
                    weight_norm.append(torch.norm(layer.state_dict()['weight']) ** 2)
                    bias_norm.append(torch.norm(layer.state_dict()['bias']) ** 2)
            reg_norm = torch.sqrt(torch.sum(torch.stack(weight_norm)) + torch.sum(torch.stack(bias_norm[0:-1])))
            q_target -= self.weight_reg * reg_norm

        q_current = self.net(batch['state']).gather(-1, batch['action']).squeeze(-1)  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)
        loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)
        return loss

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now


class ReplayBuffer(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.buffer_capacity = args.buffer_capacity
        self.count = 0
        self.buffer = {'state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'action': np.zeros((self.buffer_capacity, 1)),
                       'reward': np.zeros(self.buffer_capacity),
                       'next_state': np.zeros((self.buffer_capacity, args.state_dim)),
                       'terminal': np.zeros(self.buffer_capacity),
                       }

    def sample(self):
        index = np.random.randint(0, self.buffer_capacity, size=self.batch_size)
        batch = {}
        for key in self.buffer.keys():  # numpy->tensor
            if key == 'action':
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.long)
            else:
                batch[key] = torch.tensor(self.buffer[key][index], dtype=torch.float32)

        return batch, None

    def save(self, save_folder, states, actions, next_states, rewards, terminals):
        np.save(f"{save_folder}_state.npy", states[:self.buffer_capacity])
        np.save(f"{save_folder}_action.npy", actions[:self.buffer_capacity])
        np.save(f"{save_folder}_next_state.npy", next_states[:self.buffer_capacity])
        np.save(f"{save_folder}_reward.npy", rewards[:self.buffer_capacity])
        np.save(f"{save_folder}_terminal.npy", terminals[:self.buffer_capacity])

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy", allow_pickle=True)
        size = min(int(size), self.buffer_capacity) if size > 0 else self.buffer_capacity
        self.crt_size = min(reward_buffer.shape[0], size)

        self.buffer['state'][:self.crt_size] = np.load(f"{save_folder}_state.npy", allow_pickle=True)[:self.crt_size]
        self.buffer['action'][:self.crt_size] = np.load(f"{save_folder}_action.npy", allow_pickle=True)[:self.crt_size]
        self.buffer['next_state'][:self.crt_size] = np.load(f"{save_folder}_next_state.npy", allow_pickle=True)[:self.crt_size]
        self.buffer['reward'][:self.crt_size] = np.load(f"{save_folder}_reward.npy", allow_pickle=True)[:self.crt_size]
        self.buffer['terminal'][:self.crt_size] = np.load(f"{save_folder}_terminal.npy", allow_pickle=True)[:self.crt_size]
