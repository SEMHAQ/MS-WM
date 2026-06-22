"""下载Gymnasium MuJoCo数据集并转换格式"""
import minari
import numpy as np
import os

DATASETS = {
    'humanoid': 'mujoco/humanoid/expert-v0',
    'ant': 'mujoco/ant/expert-v0',
    'walker2d': 'mujoco/walker2d/expert-v0',
}

def download_and_convert(dataset_id, output_dir):
    print(f'\n{"="*60}')
    print(f'下载: {dataset_id}')
    print(f'{"="*60}')

    dataset = minari.load_dataset(dataset_id, download=True)
    env = dataset.recover_environment()

    print(f'数据集信息:')
    print(f'  Episodes: {dataset.total_episodes}')
    print(f'  Steps: {dataset.total_steps}')
    print(f'  Obs dim: {env.observation_space.shape[0]}')
    print(f'  Action dim: {env.action_space.shape[0]}')

    os.makedirs(f'{output_dir}/train', exist_ok=True)
    os.makedirs(f'{output_dir}/val', exist_ok=True)

    episodes = list(dataset.iterate_episodes())
    n_train = int(len(episodes) * 0.8)

    for i, ep in enumerate(episodes):
        split = 'train' if i < n_train else 'val'
        filepath = f'{output_dir}/{split}/episode_{i:04d}.npz'
        np.savez(filepath, states=ep.observations, actions=ep.actions[:-1])

        if (i + 1) % 100 == 0:
            print(f'  处理 {i+1}/{len(episodes)} episodes...')

    print(f'完成!')
    print(f'  保存到: {output_dir}')
    print(f'  Train: {n_train}, Val: {len(episodes) - n_train}')

if __name__ == '__main__':
    for name, dataset_id in DATASETS.items():
        download_and_convert(dataset_id, f'data/{name}')

    print('\n' + '='*60)
    print('所有数据集下载完成!')
    print('='*60)
