import numpy as np
from stable_baselines3 import PPO, SAC, DQN, TD3
import gym
from gym import spaces
import os
import argparse
from D3QN import ReplayBuffer


def generate_dataset(args):
    dataset_name = f'./offline_data/{args.env}_{args.gendata_pol}_e{args.eps}'
    policy_path = f'./models/{args.gendata_pol}_{args.env}'

    if args.gendata_pol == 'ppo':
        policy = PPO.load(policy_path, device=args.device)
    elif args.gendata_pol == 'sac':
        policy = SAC.load(policy_path, device=args.device)
    elif args.gendata_pol == 'dqn':
        policy = DQN.load(policy_path, device=args.device)
    elif args.gendata_pol == 'td3':
        policy = TD3.load(policy_path, device=args.device)
    else:
        raise NotImplementedError

    # prep. environment
    env = gym.make(args.env)
    data = ReplayBuffer(args)
    states = []
    actions = []
    next_states = []
    rewards = []
    terminals = []

    # generate dateset
    count = 0
    episode_limit = env._max_episode_steps
    while count < args.buffer_capacity:
        state, done = env.reset(), False
        episode_steps = 0
        if args.verbose:
            print(f'buffer size={args.buffer_capacity}======current count={count}')
        while not done:
            episode_steps += 1
            if np.random.binomial(n=1, p=args.eps):
                action = env.action_space.sample()
            else:  # else we select expert action
                action, _ = policy.predict(state)
                if 'FrozenLake' in args.env:
                    action = int(action)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            action = np.array([action])

            if np.random.binomial(n=1, p=0.001):
                print('==================================================')
                print('--------random printing offline data point--------')
                print(f'action: {action}')
                print(f'next_state: {next_state}')
                print(f'not_done: {1.0 - done}')
                print(f'reward: {reward}')

            actions.append(action)
            next_states.append(next_state)
            if done and episode_steps != episode_limit:
                terminals.append(True)
            else:
                terminals.append(False)
            rewards.append(np.array(reward))

            # check buffer size
            count += 1
            if count >= args.buffer_capacity:
                break
            else:  # state transition
                state = next_state

    data.save(dataset_name, states, actions, next_states, rewards, terminals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default='CartPole-v1')
    # e-mix (prob. to mix random actions)
    parser.add_argument("--eps", default=0.3, type=float)
    parser.add_argument("--buffer_capacity", default=int(1e5), type=float)
    parser.add_argument("--verbose", default=True, type=str)
    parser.add_argument("--device", default='cpu', type=str)
    parser.add_argument("--gendata_pol", default='ppo', type=str)
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    args = parser.parse_args()

    # determine dimensions
    env = gym.make(args.env)
    args.state_dim = env.observation_space.shape[0]

    # client input sanity check
    if args.device not in ['cpu', 'cuda', 'cuda:0', 'cuda:1', 'auto']:
        raise NotImplementedError

    # check path
    if not os.path.exists("./offline_data"):
        os.makedirs("./offline_data")

    generate_dataset(args)
