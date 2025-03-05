import os
import numpy as np
import gym
from D3QN import D3QN
import argparse
import torch


def evaluate_policy(args, env, agent, x):
    # force-mag
    evaluate_reward = 0
    for _ in range(args.evaluate_times):
        if x < 9.5:
            state = env.reset(force_mag=x, init_angle_mag=0.1)
        else:
            state = env.reset(force_mag=x)
        done = False
        episode_reward = 0
        while not done:
            action = agent.choose_action(state, epsilon=0)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        evaluate_reward += episode_reward

    return evaluate_reward / args.evaluate_times


# def evaluate_policy(args, env, agent, x):
#     # action
#     evaluate_reward = 0
#     for _ in range(args.evaluate_times):
#         state = env.reset()
#         done = False
#         episode_reward = 0
#         while not done:
#             if np.random.binomial(n=1, p=x):
#                 action = env.action_space.sample()
#             else:
#                 action = agent.choose_action(state, epsilon=0)
#             next_state, reward, done, _ = env.step(action)
#             episode_reward += reward
#             state = next_state
#         evaluate_reward += episode_reward
#
#     return evaluate_reward / args.evaluate_times


# def evaluate_policy(args, env, agent, x):
#     # length
#     evaluate_reward = 0
#     for _ in range(args.evaluate_times):
#         state = env.reset(length=x)
#         done = False
#         episode_reward = 0
#         while not done:
#             action = agent.choose_action(state, epsilon=0)
#             next_state, reward, done, _ = env.step(action)
#             episode_reward += reward
#             state = next_state
#         evaluate_reward += episode_reward
#
#     return evaluate_reward / args.evaluate_times


def load_agent(agent, load_path, device):
    agent.net.load(f'{load_path}', device=device)
    return agent


def save_evals(save_path, setting, avgs, stds):
    np.save(f'{save_path}_{setting}_avgs', avgs)
    np.save(f'{save_path}_{setting}_stds', stds)


def main(args):
    file_seed = args.file_seed
    env_seed = args.env_seed
    # evaluate algorithm on perturbed environments
    if args.weight_reg > 0:
        load_path = f"./models/RD2QN_{args.env}_seed_{file_seed}"
        save_path = f"./perturbed_results/RD2QN_{args.env}_seed_{file_seed}"
    else:
        load_path = f"./models/D2QN_{args.env}_seed_{file_seed}"
        save_path = f"./perturbed_results/D2QN_{args.env}_seed_{file_seed}"
    # get perturbed environment
    i = args.env.find('-')
    perturbed_env = f'{args.env[:i]}Perturbed{args.env[i:]}'
    env = gym.make(perturbed_env)
    env_evaluate = gym.make(perturbed_env)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    env.seed(env_seed)
    env.action_space.seed(env_seed)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)
    torch.cuda.manual_seed_all(env_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.n
    args.episode_limit = env._max_episode_steps  # Maximum number of steps per episode
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("episode_limit={}".format(args.episode_limit))

    agent = D3QN(args)
    agent.net.load(f'{load_path}', device=args.device)
    eval_episodes = args.eval_episodes

    if args.env == 'CartPole-v1':
        # perturb 'force_mag'
        setting = 'force_mag'
        ps_fm = np.arange(-0.8, 0.1, 0.1)
        avgs = []
        stds = []

        for p in ps_fm:
            env.reset()
            force_mag = env.force_mag * (1 + p)
            rewards = []
            for _ in range(eval_episodes):
                env.seed(seed=np.random.randint(1000))
                evaluate_reward = evaluate_policy(args, env, agent, force_mag)
                rewards.append(evaluate_reward)

            avg_reward = np.sum(rewards) / eval_episodes
            print("---------------------------------------")
            print(f' force_mag with p {p}')
            print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
            print("---------------------------------------")
            avgs.append(avg_reward)
            stds.append(np.std(rewards))
        save_evals(save_path, setting, avgs, stds)

        # # perturb action
        # setting = 'action'
        # es = np.arange(0, 0.5, 0.1)
        # avgs = []
        # stds = []
        #
        # for e in es:
        #     env.reset()
        #     rewards = []
        #     for _ in range(eval_episodes):
        #         env.seed(seed=np.random.randint(1000))
        #         evaluate_reward = evaluate_policy(args, env, agent, e)
        #         rewards.append(evaluate_reward)
        #
        #     avg_reward = np.sum(rewards) / eval_episodes
        #     print("---------------------------------------")
        #     print(f' action with p {e}')
        #     print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
        #     print("---------------------------------------")
        #     avgs.append(avg_reward)
        #     stds.append(np.std(rewards))
        # save_evals(save_path, setting, avgs, stds)

        # # perturb 'length'
        # setting = 'length'
        # avgs = []
        # stds = []
        # # ps = np.arange(0.0, 0.7, 0.1)
        # ps = np.arange(0.0, 1.2, 0.2)
        # for p in ps:
        #     env.reset()
        #     length = env.length * (1 + p)
        #     rewards = []
        #     for _ in range(eval_episodes):
        #         # complete random environment for each episode
        #         env.seed(seed=np.random.randint(1000))
        #         evaluate_reward = evaluate_policy(args, env, agent, length)
        #         rewards.append(evaluate_reward)
        #
        #     # episodes for current p are done
        #     avg_reward = np.sum(rewards) / eval_episodes
        #     print("---------------------------------------")
        #     print(f' length with p {p}')
        #     print(f" over {eval_episodes} episodes: {avg_reward:.3f}")
        #     print("---------------------------------------")
        #     avgs.append(avg_reward)
        #     stds.append(np.std(rewards))
        # # all p's are done
        # save_evals(save_path, setting, avgs, stds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for (Robust) D3QN")
    parser.add_argument("--env", type=str, default='CartPole-v1', help="CartPole-v1/LunarLander-v2")
    parser.add_argument("--eval_episodes", default=10, type=int)
    parser.add_argument("--max_train_steps", type=int, default=int(3e5), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=10, help="Evaluate times")
    parser.add_argument("--device", default='cpu', type=str)

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
    parser.add_argument("--weight_reg", type=float, default=1e-4, help="Regularization for weights of V")

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="Whether to use dueling network")
    parser.add_argument("--file_seed", type=int, default=0, help="seed of file name")
    parser.add_argument("--env_seed", type=int, default=0, help="seed of env")

    args = parser.parse_args()
    # make folders to dump results
    if not os.path.exists("./perturbed_results"):
        os.makedirs("./perturbed_results")
    if not os.path.exists("./models"):
        os.makedirs("./models")

    main(args)
