import os
import numpy as np
import pickle
import itertools
from det3d.datasets.base import BaseDataset

class KittiDataset(BaseDataset):
    """Kitti Dataset class for detection task."""

    def __init__(self,
                 info_path,
                 root_path,
                 sampler=None,
                 loading_pipelines=None,
                 augmentation=None,
                 prepare_label=None,
                 class_names=None,
                 resampling=False,
                 evaluations=None,
                 create_database=False,
                 use_gt_sampling=True):

        super(KittiDataset, self).__init__(
            root_path, info_path, sampler, loading_pipelines, augmentation, prepare_label, evaluations, create_database,
            use_gt_sampling=use_gt_sampling
        )

        self._class_names = list(itertools.chain(*[t for t in class_names]))

        if resampling:
            self.cbgs()

    def cbgs(self):
        """Class-balanced group sampling for imbalanced dataset."""
        _cls_infos = {name: [] for name in self._class_names}
        for info in self.infos:
            for name in set(info["annos"]["name"]):
                if name in self._class_names:
                    _cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
        _cls_dist = {k: len(v) / duplicated_samples for k,
                     v in _cls_infos.items()}

        _kitti_infos = []

        frac = 1.0 / len(self._class_names)
        ratios = [frac / v for v in _cls_dist.values()]

        for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
            _kitti_infos += np.random.choice(cls_infos,
                                           int(len(cls_infos) * ratio)).tolist()

        self.infos = _kitti_infos

    def load_pointcloud(self, res, info):
        """Load point cloud data for KITTI dataset."""
        v_path = info['point_cloud']["velodyne_path"]
        v_feat_num = info['point_cloud']["num_features"]
        points = np.fromfile(
            os.path.join(self._root_path, v_path), dtype=np.float32
        ).reshape(-1, v_feat_num)
        
        res["points"] = points
        return res

    def load_box3d(self, res, info):
        """Load 3D bounding box annotations for KITTI dataset."""
        # For KITTI dataset, gt_boxes includes both 3D boxes and 2D boxes(optional)
        annotations = {}
        
        # Load 3D boxes and names
        annos = info["annos"]
        gt_names = np.array(annos["name"]).copy()

        # load 3d boxes
        gt_boxes = np.array(annos["converted_bbox"]).copy()
        # move center from bottom to center
        gt_boxes[:, 2] += gt_boxes[:, 5] / 2.0
            
        annotations.update({
            'gt_boxes': gt_boxes,
            'gt_names': gt_names,
            'truncated': annos["truncated"],
            'occluded': annos["occluded"],
            'difficulty': annos["difficulty"],
            'num_points_in_gt': annos["num_points_in_gt"],
        })
        
        res["annotations"] = annotations
        return res 
