from __future__ import print_function, division

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import json

n_seconds = 9
FEATURE_PATH = '/home/kuhu6123/PDData/C3D-FULL-N/array_x_train_resnet_high_resolution_{0}_{1}_{2}.npy'
POSE_PATH = '/home/rsun6573/work/pd_work/two_stream/pose_data/MJFF_{0}/{1}_keypoints.json'


class PoseNomalise(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        sample = {'label': self.video_df.iloc[idx, 0], 'pose': seq_poses, 'context': context_features,
                  'video_id': video_id, 'second_id': second_id}
        context, pose, label = sample['context'], sample['pose'], sample['label']
        pose = pose / self.output_size
        context = context / self.output_size
        return {'context': context, 'pose': pose, 'label': label}


class PoseDataset(Dataset):
    """Pose dataset."""

    def __init__(self, csv_file, fold_id, is_training, transform=None):
        self.video_df = self.get_video_df(csv_file, fold_id, is_training)
        self.transform = transform

    def __len__(self):
        return len(self.video_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.video_df.iloc[idx, 0]
        subject_id = self.video_df.iloc[idx, 1]
        video_id = self.video_df.iloc[idx, 2]
        second_id = self.video_df.iloc[idx, 3]
        subject_id_str = str(subject_id).zfill(5)
        feature_file = FEATURE_PATH.format(subject_id_str, video_id, second_id)
        half_range = n_seconds // 2
        context_features = []
        seq_poses = []
        for new_second_id in range(second_id - half_range, second_id + half_range + 1):
            context_file = FEATURE_PATH.format(subject_id_str, video_id, new_second_id)
            try:
                context_feature = np.load(context_file)
            except FileNotFoundError as err:
                context_feature = np.zeros((1, 32768))
                context_feature = np.float32(context_feature)
            context_features.append(context_feature)

            file_id = (new_second_id - 1) * 25 + 12
            pose_file = POSE_PATH.format(video_id, file_id)
            try:
                pose = self._read_pose(pose_file)
                if len(pose) > 200:
                    pose = pose[0:200]
                pose = pose + [0] * (200 - len(pose))
                pose_marks = np.array([pose])
                pose_marks = pose_marks.astype('float').reshape(1, -1, 2)
                seq_poses.append(pose_marks)
            except FileNotFoundError as err:
                # print(err)
                pose = np.random.rand(200)
                pose_marks = pose.astype('float').reshape(1, -1, 2)
                seq_poses.append(pose_marks)

        seq_poses = np.concatenate(seq_poses, axis=0)

        context_features = np.concatenate(context_features, axis=0)

        sample = {'label': self.video_df.iloc[idx, 0], 'pose': seq_poses, 'context': context_features,
                  'video_id': video_id, 'second_id': second_id}

        return sample

    @staticmethod
    def _read_pose(pose_file):
        with open(pose_file) as f:
            data = json.load(f)

        pose = []
        people = data.get('people')

        for person in people:
            points = person.get('pose_keypoints_2d')

            for idx, point in enumerate(points):
                if idx % 3 == 0:

                    pose.append(point)
                elif idx % 3 == 1:
                    pose.append(point)
                else:
                    pass
        return pose

    @staticmethod
    def get_video_df(csv_file, fold_id, is_training):
        df = pd.read_csv(csv_file)

        if is_training:
            df = df[df['group_id'] != fold_id]
        else:
            df = df[df['group_id'] == fold_id]
            return df

        df_1 = df[df['label'] == 1]
        df_0 = df[df['label'] == 0]
        result = df_1.groupby('video_id').agg('count').reset_index()

        dfs = []
        remainder = 0
        for index, row in result.iterrows():
            video_id = row['video_id']
            cnt = row['label']
            df_tmp = df_0[df_0['video_id'] == video_id]

            if remainder > 0:
                cnt = cnt + remainder
                remainder = 0

            try:
                # random_state=1 will give the same random samples
                df_tmp = df_tmp.sample(n=cnt, random_state=1)
            except:
                print('video:', video_id, row['subject_id'])
                remainder = cnt - df_tmp.shape[0]

            dfs.append(df_tmp)
        df_0 = pd.concat(dfs)
        df_all = pd.concat([df_0, df_1]).reset_index().drop(['index'], axis=1)
        return df_all

# source_file = '/home/rsun6573/work/pd_work/engineering_tools/pose_data_grouped.csv'
# source_file = '/home/rsun6573/work/pd_work/updated_label.csv'
# train_data = PoseDataset(source_file, 1, True)
# # print(len(train_data))

# test_data = PoseDataset(source_file, 1, False)
# # print(len(test_data))

# train_loader = DataLoader(train_data, batch_size=16,
#                         shuffle=True, num_workers=0)

# cnt = 0
# for i_batch, sample_batched in enumerate(train_loader):
#     pose = sample_batched['pose']
#     target = sample_batched['label']
#     context = sample_batched['context']
#     video_id = sample_batched['video_id']
#     second_id = sample_batched['second_id']

#     # for i in range(0, 1):
#     #     print(data.numpy()[i])
#     #     print(target.numpy()[i])
#     #     print(video_id.numpy()[i], second_id.numpy()[i])
#     print('context:', context.shape)
#     print('pose:', pose.shape)
#     cnt += 1
#     if cnt == 1:
#         break

# print(cnt)
