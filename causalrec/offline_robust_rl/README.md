## **Session-Level Dynamic Ad Load Optimization using Offline Robust Reinforcement Learning**
Implementation of the algorithm Offline Robust Dueling Deep Q-Network with an integral probability metric uncertainty set. To verify the robust performance of algorithms, perturbed CartPole environment is created based on [the previous implementation](https://github.com/zaiyan-x/RFQI.git).  

### Prerequisites
Here we list our running environment:
- gym == 0.21.0
- PyTorch == 2.1.1
- numpy == 1.22.3
- matplotlib == 3.5.1
- stable-baselines3 == 2.2.1


We then need to properly register the perturbed Gym environments within the folder perturned_env.
1. Add cartpole_perturbed.py under gym/envs/classic_control
2. Add the following to _init_.py under gym/envs:
```
register(
    id="CartPolePerturbed-v1",
    entry_point="gym.envs.mujoco.hopper_perturbed:HopperPerturbedEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)
```



### Instruction
#### Offline Data Generation
To generate the epsilon-greedy dataset for CartPole-v1, please run:
```
python gen_offline_data.py --eps=0.3
```


#### Train Policy
To train an offline robust dueling DQN policy, please run 
```
python train_rd3qn.py --weight_reg=1e-4
```

To train an offline dueling DQN policy, please run 
```
python train_rd3qn.py --weight_reg=0.0
```


#### Evaluate Policy
To evaluate an offline robust dueling DQN policy, please run
```
python eval_rd3qn.py --weight_reg=1e-4
```

To evaluate an offline dueling DQN policy, please run
```
python eval_rd3qn.py --weight_reg=0.0
```
