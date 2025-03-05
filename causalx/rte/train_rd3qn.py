import os
import torch
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
from D3QN import D3QN, ReplayBuffer
import argparse


class Train:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        self.gendata_pol = args.gendata_pol
        self.eps = args.eps

        self.env = gym.make(env_name)
        self.env_evaluate = gym.make(env_name)  # When evaluating the policy, we need to rebuild an environment
        self.env.seed(seed)
        self.env.action_space.seed(seed)
        self.env_evaluate.seed(seed)
        self.env_evaluate.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.args.state_dim = self.env.observation_space.shape[0]
        self.args.action_dim = self.env.action_space.n
        self.args.episode_limit = self.env._max_episode_steps  # Maximum number of steps per episode
        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        self.replay_buffer = ReplayBuffer(args)
        self.agent = D3QN(args)

        self.algorithm = 'DQN'
        if self.args.use_double:
            self.algorithm += '_Double'
        if self.args.use_dueling:
            self.algorithm += '_Dueling'
        if self.args.weight_reg > 0:
            self.algorithm += '_Robust'

        self.writer = SummaryWriter(log_dir='runs/DQN/{}_env_{}_number_{}_seed_{}'.format(self.algorithm, env_name, number, seed))
        if self.args.weight_reg > 0:
            self.save_path = f"./models/RD2QN_{self.args.env}_seed_{self.seed}"
        else:
            self.save_path = f"./models/D2QN_{self.args.env}_seed_{self.seed}"

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        self.max_value = -np.inf

        self.epsilon = self.args.epsilon_init
        self.epsilon_min = self.args.epsilon_min
        self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def run(self, ):
        self.evaluate_policy()
        prev_reward, flag = 0, 0
        self.replay_buffer.load(f'./offline_data/{self.env_name}_{self.gendata_pol}_e{self.eps}')

        while self.total_steps < self.args.max_train_steps:
            loss = self.agent.learn(self.replay_buffer, self.total_steps)
            self.total_steps += 1

            if self.total_steps % self.args.evaluate_freq == 0:
                eval_reward = self.evaluate_policy()
                print('loss', loss)

                # save actor, critic for evaluation in perturbed env
                if self.env_name == 'CartPole-v1':
                    if eval_reward == 500 and prev_reward == 500:
                        if flag == 2:
                            self.agent.net.save(f'{self.save_path}')
                        flag += 1
                    prev_reward = eval_reward

        # Save reward
        np.save('./data_train/{}_env_{}_number_{}_seed_{}.npy'.format(self.algorithm, self.env_name, self.number, self.seed), np.array(self.evaluate_rewards))

    def evaluate_policy(self, ):
        evaluate_reward = 0
        self.agent.net.eval()
        for _ in range(self.args.evaluate_times):
            state = self.env_evaluate.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)
                next_state, reward, done, _ = self.env_evaluate.step(action)
                episode_reward += reward
                state = next_state
            evaluate_reward += episode_reward

        self.agent.net.train()
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.writer.add_scalar('step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        return evaluate_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for (Robust) D3QN")
    parser.add_argument("--env", type=str, default='CartPole-v1', help="CartPole-v1/LunarLander-v2")
    parser.add_argument("--max_train_steps", type=int, default=int(1e5), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="The maximum replay-buffer capacity ")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="How many steps before the epsilon decays to the minimum")
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="Update frequency of the target network(hard update)")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")
    parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weights of V")
    parser.add_argument("--gendata_pol", default='ppo', type=str)
    parser.add_argument("--eps", default=0.3, type=float)
    parser.add_argument("--use_double", type=bool, default=False, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")

    args = parser.parse_args()
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")
    if not os.path.exists("./models"):
        os.makedirs("./models")
    for seed in [0]:
        runner = Train(args=args, env_name=args.env, number=1, seed=seed)
        runner.run()