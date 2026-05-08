import os
import pickle
from pathlib import Path
from typing import Union, Optional

import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from tqdm import tqdm
import pandas as pd



def choose_split(dep_name, split_df):
    split = split_df[split_df["names"] == dep_name]['split'].values[0]
    return split


class MDDD(data.Dataset):  # /home/bichenwang/MDDD/
    def __init__(self, root: Union[str, Path] = './MDDD_final/processed', split='train'):
        self.root = root if isinstance(root, Path) else Path(root)
        self.dep_root, self.non_dep_root = os.path.join(root, 'dep_feat.pkl'), os.path.join(root,
                                                                                                 'nondep_feat.pkl')
        split_df = pd.read_csv(os.path.join(root, 'labels.csv'))
        with open(self.dep_root, 'rb') as file:
            dep_data = pickle.load(file)
        dep_feats, dep_names = dep_data['features'], dep_data['name']
        choose_index = [i for i in range(len(dep_names)) if choose_split(dep_names[i], split_df) == split]
        dep_feats = [dep_feats[i] for i in choose_index]
        dep_names = [dep_names[i] for i in choose_index]
        with open(self.non_dep_root, 'rb') as file:
            non_dep_data = pickle.load(file)
        non_dep_feats, non_dep_names = non_dep_data['features'], non_dep_data['name']
        choose_index = [i for i in range(len(non_dep_names)) if choose_split(non_dep_names[i], split_df) == split]
        non_dep_feats = [non_dep_feats[i] for i in choose_index]
        non_dep_names = [non_dep_names[i] for i in choose_index]

        self.features = dep_feats + non_dep_feats
        self.names = dep_names + non_dep_names
        self.labels = [1] * len(dep_feats) + [0] * len(non_dep_feats)
        print(split, len(non_dep_feats), len(dep_feats), len(self.features))

    def __getitem__(self, i: int):
        feature = [f[:60, :] for f in self.features[i]][0:360] # 截断到60帧，前360个视频
        label = self.labels[i]
        name = self.names[i]
        return feature, name, label
# 核心逻辑：确保每一段视频 f 都是 [60, 161]
        # raw_user_features = self.features[i]
        # processed_features = []
        
        # # 限制每个用户最多 360 段视频
        # for f in raw_user_features[:360]:
        #     target_t = 60
        #     curr_t, curr_dim = f.shape
            
        #     # 强制对齐到 60 帧
        #     if curr_t >= target_t:
        #         f_new = f[:target_t, :]
        #     else:
        #         f_new = np.zeros((target_t, curr_dim), dtype=np.float32)
        #         f_new[:curr_t, :] = f
            
        #     processed_features.append(f_new.astype(np.float32))
            
        # label = self.labels[i]
        # name = self.names[i]
        # return processed_features, name, label

    def __len__(self):
        return len(self.labels)
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    features = [d[0] for d in batch]
    names = [d[1] for d in batch]
    labels = [d[2] for d in batch]

    video_features = []
    lengths = []

    # 先确定固定的 feature 维度
    target_feat_dim = 161
    target_frame_len = 60

    for user_feats in features:
        # user_feats: list of np.array, 每个 array shape = [seq_len, feat_dim]
        user_tensors = []
        for f in user_feats:
            f = torch.from_numpy(f)
            # pad frame长度到 target_frame_len
            if f.shape[0] < target_frame_len:
                pad = torch.zeros((target_frame_len - f.shape[0], f.shape[1]), dtype=f.dtype)
                f = torch.cat([f, pad], dim=0)
            else:
                f = f[:target_frame_len, :]

            # pad feature维度到 target_feat_dim
            if f.shape[1] < target_feat_dim:
                pad = torch.zeros((f.shape[0], target_feat_dim - f.shape[1]), dtype=f.dtype)
                f = torch.cat([f, pad], dim=1)
            elif f.shape[1] > target_feat_dim:
                f = f[:, :target_feat_dim]

            user_tensors.append(f)

        # pad用户的视频段数到 batch 内最大
        lengths.append(len(user_tensors))
        user_tensors = pad_sequence(user_tensors, batch_first=True)  # [num_docs, frames, feat_dim]
        video_features.append(user_tensors)

    # pad batch 维度
    video_features = pad_sequence(video_features, batch_first=True)  # [batch, max_docs, frames, feat_dim]
    labels = torch.tensor(labels)

    return (video_features, lengths, names), labels

def get_dvlog_dataset(
        root: Union[str, Path], fold: str = "train", batch_size: int = 8,
        gender: str = "both"
):
    """Get dataloader for DVlog dataset.

    Args:
        root (Union[str, Path]): path to the dvlog dataset. Should be something
            like `*/dvlog-dataset`.
        fold (str, optional): train / valid / test. Defaults to "train".
        batch_size (int, optional): Defaults to 8.
        gender (str, optional): m / f / both. Defaults to both.
        transform (optional): Defaults to None.
        target_transform (optional): Defaults to None.

    Returns:
        the dataloader.
    """
    dataset = DVlog(root, fold, gender)
    dataloader = data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=_collate_fn,
        shuffle=(fold == "train"),
    )
    return dataloader
