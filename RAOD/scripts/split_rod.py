#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import random

def split_dataset(root_dir, train_ratio=0.9, seed=42):
    """
    将 root_dir/raws/*.npy.gz 与 root_dir/anno/00_Train/*.json
    按照 train_ratio 划分到 train/ 和 test/ 子目录中。
    """
    random.seed(seed)

    raws_dir = os.path.join(root_dir, 'raws_debayer_fp32_640x640','00Train')
    anno_dir = os.path.join(root_dir, 'anno', '00Train')

    # 1. 列出所有基名（不含扩展）
    raw_files = [f for f in os.listdir(raws_dir) if f.endswith('.npy.gz')]
    base_names = [os.path.splitext(os.path.splitext(f)[0])[0] for f in raw_files]

    # 2. 打乱后切分
    random.shuffle(base_names)
    n_train = int(len(base_names) * train_ratio)
    train_names = set(base_names[:n_train])
    test_names  = set(base_names[n_train:])

    # 3. 创建输出目录
    for split in ('train', 'test'):
        os.makedirs(os.path.join(root_dir, split, 'raws'), exist_ok=True)
        os.makedirs(os.path.join(root_dir, split, 'anno'), exist_ok=True)

    # 4. 移动或复制文件
    for name in base_names:
        # 源路径
        raw_src  = os.path.join(raws_dir,  f'{name}.npy.gz')
        anno_src = os.path.join(anno_dir,  f'{name}.json')

        # 目标子目录
        split = 'train' if name in train_names else 'test'
        raw_dst  = os.path.join(root_dir, split, 'raws',  f'{name}.npy.gz')
        anno_dst = os.path.join(root_dir, split, 'anno',  f'{name}.json')

        # 复制文件，你也可以用 os.rename 做移动
        shutil.copy2(raw_src, raw_dst)
        shutil.copy2(anno_src, anno_dst)


    print(f'Total: {len(base_names)} items → train: {len(train_names)}, test: {len(test_names)}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split raws+anno into train/test')
    parser.add_argument('root_dir', help='dataset root directory (contains raws/ and anno/00_Train/)')
    parser.add_argument('--ratio', type=float, default=0.9, help='train split ratio (default: 0.9)')
    parser.add_argument('--seed',  type=int,   default=42,  help='random seed')
    args = parser.parse_args()

    split_dataset(args.root_dir, train_ratio=args.ratio, seed=args.seed)
