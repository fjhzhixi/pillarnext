import os
import numpy as np
import pickle
import itertools
import json
import copy
from det3d.datasets.base import BaseDataset
from det3d.datasets.kitti.kitti_eval_utils import compute_type, compute_ap, calculate_iou
import math

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
        self.bev = False
        self.iou_threshold_dict = {
            "Car": 0.5,
            "Truck": 0.5,
            "Pedestrian": 0.5,
            "Cyclist": 0.5,
        }
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
    
    def get_token(self, info):
        return info['point_cloud']["velodyne_path"].split('/')[-1].split('.')[0]

    def load_pointcloud(self, res, info):
        """Load point cloud data for KITTI dataset."""
        v_path = info['point_cloud']["velodyne_path"]
        v_feat_num = info['point_cloud']["num_features"]
        points = np.fromfile(
            os.path.join(self._root_path, v_path), dtype=np.float32
        ).reshape(-1, v_feat_num)
        
        res["points"] = points
        res["token"] = self.get_token(info)
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

    def evaluation(self, detections, output_dir, gt_labels):
        with open('/mnt/data4/jlk/kitti/my_split_spd_final.json', 'r') as f:
            split = json.load(f)
        results = {}

        pred = []
        label = []
        frame_id_to_list = {}
        for key in detections.keys():
            frame_id_to_list[key] = len(pred)
            pred.append(self.process_pred(detections[key]))
            label.append(self.process_label(gt_labels[key]))
        
        # label = label[:50]
        # pred = pred[:50]
        overlaps = calculate_iou(pred, label)

        for seq_id, frame_ids in split.items():
            self.gt_num = {}
            self.all_preds = {}
            for pred_class in self._class_names:
                self.all_preds[pred_class] = []
                self.gt_num[pred_class] = 0
            for frame_id in frame_ids:
                list_id = frame_id_to_list[frame_id]
                self.add_frame(pred[list_id], label[list_id], overlaps[list_id])
            print(f'========== seq {seq_id} ==========')
            results[seq_id] = self.print_ap()
            # break
        print(f'========== all seqs ==========')
        res_val = list(results.values())
        for key in res_val[0].keys():
            sum_ap = 0
            for item in res_val:
                sum_ap += item[key]
            print("%s AP: %.2lf" % (key, sum_ap / len(results) * 100))

        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(results, f, indent=4)


    def add_frame(self, pred, label, overlaps):
        for pred_class in self._class_names:
            pred_result, num_label, num_tp = compute_type(label, pred, overlaps, pred_class, self.iou_threshold_dict[pred_class])
            self.all_preds[pred_class] += pred_result
            self.gt_num[pred_class] += num_label
                # logger.debug("iou: {}, tp: {}, all_pred: {}".format(iou, num_tp, len(pred["labels_3d"])))

    def process_pred(self, info):
        # info: dict_keys(['box3d_lidar', 'scores', 'label_preds', 'token'])
        # target: N * {'box': Array[7], 'label': str, 'score': float}
        result = []
        for idx in range(len(info['box3d_lidar'])):
            box = info['box3d_lidar'][idx].cpu().numpy()[[0,1,2,3,4,5,8]]
            if self.bev:
                box[2] = box[5] = 1
            label_name = self._class_names[info['label_preds'][idx].item()]
            score = info['scores'][idx].item()
            result.append({'box': box, 'label': label_name, 'score': score})
        return result

    def process_label(self, info):
        # info: dict_keys(['gt_boxes', 'gt_names', 'truncated', 'occluded', 'difficulty', 'num_points_in_gt'])
        # target: N * {'box': Array[7], 'label': str, 'score': float}
        result = []
        for idx in range(len(info['gt_boxes'])):
            box = info['gt_boxes'][idx]
            if self.bev:
                box[2] = box[5] = 1
            label = info['gt_names'][idx]
            score = 1.0
            result.append({'box': box, 'label': label, 'score': score})
        return result

    def print_ap(self):
        result = {}
        mAP = 0.0
        for pred_class in self._class_names:
            ap = compute_ap(self.all_preds[pred_class], self.gt_num[pred_class])
            mAP += ap
            result[pred_class] = ap
            print("%s AP: %.2lf" % (pred_class, ap * 100))
        print("mAP: %.2lf" % (mAP / len(self._class_names) * 100))
        result['mAP'] = mAP / len(self._class_names)
        return result