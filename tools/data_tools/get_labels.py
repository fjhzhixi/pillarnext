# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import numpy as np
import warnings
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress
from os import path as osp
import pickle
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
# import open3d as o3d
from mmdet3d.core.visualizer.open3d_vis import Visualizer
import torch
import json

def build_data_cfg(config_path, skip_type):
    """Build data config for loading visualization data."""
    cfg = Config.fromfile(config_path)
    # extract inner dataset of `RepeatDataset` as `cfg.data.train`
    # so we don't need to worry about it later
    if cfg.data.train['type'] == 'RepeatDataset':
        cfg.data.train = cfg.data.train.dataset
    # use only first dataset for `ConcatDataset`
    if cfg.data.train['type'] == 'ConcatDataset':
        cfg.data.train = cfg.data.train.datasets[0]
    train_data_cfg = cfg.data.train
    # eval_pipeline purely consists of loading functions
    # use eval_pipeline for data loading
    train_data_cfg['pipeline'] = [
        x for x in cfg.eval_pipeline if x['type'] not in skip_type
    ]

    return cfg

def main():
    results = {}
    pc_range = [0, -39.68, -3, 92.16, 39.68, 1]
    for split in ['train', 'val']:
        config = 'tools/data_tools/config_original.py'
        skip_type = ['Normalize']
        base_path = '/mnt/data4/jlk/kitti_original'
        info_path = f'/mnt/data4/jlk/kitti_original/kitti_infos_{split}.pkl'

        with open(info_path, 'rb') as f:
            data_infos_origin = pickle.load(f)

        cfg = build_data_cfg(config, skip_type)
        dataset = build_dataset(cfg.data[split], default_args=dict(filter_empty_gt=False))
        data_infos = dataset.data_infos

        for idx, data_info in enumerate(track_iter_progress(data_infos)):
            data_path = data_info['point_cloud']['velodyne_path']
            assert data_path == data_infos_origin[idx]['point_cloud']['velodyne_path']

            gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d']
            labels = dataset.get_ann_info(idx)['gt_names']

            obj_dict = {}
            for idx_box in range(len(gt_bboxes)):
                item = gt_bboxes[idx_box]
                ts = item.tensor.numpy()[0]
                if ts[0] < pc_range[0] or ts[0] > pc_range[3] or ts[1] < pc_range[1] or ts[1] > pc_range[4] or ts[2] + ts[5] / 2.0 < pc_range[2] or ts[2] + ts[5] / 2.0 > pc_range[5]:
                    continue
                label = labels[idx_box]
                obj_dict[label] = obj_dict.get(label, 0) + 1

            id = data_path.split('/')[-1].split('.')[0]
            results[id] = obj_dict
            # break

    with open('tools/data_tools/labels.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
