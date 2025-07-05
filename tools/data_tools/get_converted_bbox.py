import argparse
import numpy as np
import warnings
from mmcv import Config, DictAction, mkdir_or_exist, track_iter_progress
from os import path as osp
import pickle
from mmdet3d.datasets import build_dataset
import open3d as o3d
from mmdet3d.core.visualizer.open3d_vis import Visualizer



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
    split = 'train'
    config = 'tools/data_tools/config.py'
    skip_type = ['Normalize']
    base_path = '/mnt/data4/jlk/kitti'
    info_path = f'/mnt/data4/jlk/kitti/kitti_infos_{split}.pkl'
    save_path = f'/mnt/data4/jlk/kitti/kitti_infos_{split}_converted.pkl'

    with open(info_path, 'rb') as f:
        data_infos_origin = pickle.load(f)


    cfg = build_data_cfg(config, skip_type)
    dataset = build_dataset(cfg.data[split], default_args=dict(filter_empty_gt=False))
    data_infos = dataset.data_infos

    for idx, data_info in enumerate(track_iter_progress(data_infos)):
        data_path = data_info['point_cloud']['velodyne_path']
        assert data_path == data_infos_origin[idx]['point_cloud']['velodyne_path']

        gt_bboxes = dataset.get_ann_info(idx)['gt_bboxes_3d']
        gt_bboxes_center = gt_bboxes.gravity_center

        data_infos_origin[idx]['annos']['converted_bbox'] = gt_bboxes.tensor.numpy()
        data_infos_origin[idx]['annos']['converted_bbox_center'] = gt_bboxes_center.numpy()

        # file_path = osp.join(base_path, data_info['point_cloud']['velodyne_path'])
        # points = np.fromfile(file_path, dtype=np.float32)
        # points = points.reshape((-1, 4))[:, :3]
        # vis = Visualizer(points)
        # vis.add_bboxes(bbox3d=gt_bboxes.tensor, bbox_color=(0, 0, 1))
        # vis.show()
    with open(save_path, 'wb') as f:
        pickle.dump(data_infos_origin, f)


if __name__ == '__main__':
    main()