"""用Gymnasium生成MuJoCo专家数据集"""
import gymnasium as gym
import numpy as np
import os

DATASETS = {
    'humanoid': {'env': 'Humanoid-v5', 'episodes': 1000},
    'humanoid_standup': {'env': 'HumanoidStandup-v5', 'episodes': 1000},
    'ant': {'env': 'Ant-v5', 'episodes': 1000},
    'walker2d': {'env': 'Walker2d-v5', 'episodes': 1000},
}

def generate_expert_data(env_name, output_dir, n_episodes=1000, max_steps=1000):
    """生成专家数据（使用随机策略作为baseline）"""
    print(f'\n{"="*60}')
    print(f'生成: {env_name}')
    print(f'{"="*60}')

    env = gym.make(env_name)
    os.makedirs(f'{output_dir}/train', exist_ok=True)
    os.makedirs(f'{output_dir}/val', exist_ok=True)

    print(f'环境信息:')
    print(f'  Obs dim: {env.observation_space.shape[0]}')
    print(f'  Action dim: {env.action_space.shape[0]}')

    n_train = int(n_episodes * 0.8)
    episodes = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        states = [obs]
        actions = []

        for step in range(max_steps):
            action = env.action_space.sample()  # 随机策略
            obs, reward, terminated, truncated, _ = env.step(action)
            states.append(obs)
            actions.append(action)

            if terminated or truncated:
                break

        states = np.array(states)
        actions = np.array(actions)
        episodes.append((states, actions))

        if (i + 1) % 100 == 0:
            print(f'  生成 {i+1}/{n_episodes} episodes...')

    # 保存
    for i, (states, actions) in enumerate(episodes):
        split = 'train' if i < n_train else 'val'
        filepath = f'{output_dir}/{split}/episode_{i:04d}.npz'
        np.savez(filepath, states=states, actions=actions)

    print(f'完成!')
    print(f'  保存到: {output_dir}')
    print(f'  Train: {n_train}, Val: {n_episodes - n_train}')
    print(f'  Obs dim: {env.observation_space.shape[0]}, Action dim: {env.action_space.shape[0]}')

if __name__ == '__main__':
    for name, cfg in DATASETS.items():
        generate_expert_data(cfg['env'], f'data/{name}', cfg['episodes'])

    print('\n' + '='*60)
    print('所有数据集生成完成!')
    print('='*60)
