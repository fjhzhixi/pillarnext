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
        self.bev = True
        self.iou_threshold_dict = {
            "Car": 0.5,
            "Truck": 0.5,
            "Pedestrian": 0.3,
            "Cyclist": 0.3,
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
        v_path = v_path.replace('velodyne', 'velodyne_reduced')
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
        with open('/mnt/data5/fjh/PillarNeXt/codebase/pillarnext/tools/data_tools/output/my_split_spd_final.json', 'r') as f:
            split = json.load(f)
        # iou@0.5
        # delete_seq = ['0235', '0215', '0275', '0301', '0136', '0123', '0037', '0141', '0129', '0160', '0158', '0033', '0208', '0203', '0302', '0232', '0099', '0132', '0073', '0303', '0296', '0078', '0127', '0087', '0108', '0212', '0195', '0142', '0125', '0064', '0150', '0271', '0245', '0240', '0169', '0151', '0116', '0058', '0124', '0085']
        # delete_seq = delete_seq[:31]

        # iou@0.3
        delete_seq = ['0215', '0235', '0275', '0136', '0301', '0123', '0141', '0037', '0129', '0208', '0158', '0033', '0302', '0232', '0073', '0078', '0160', '0108', '0132', '0212', '0303', '0296', '0203', '0271', '0240', '0169', '0127', '0087', '0064', '0058', '0116', '0085', '0241', '0195', '0099', '0196', '0124', '0142', '0151', '0036']
        delete_seq = delete_seq[:22]

        results = {}
        center_dis = []
        pred = []
        label = []
        frame_id_to_list = {}
        for key in detections.keys():
            frame_id_to_list[key] = len(pred)
            pred.append(self.process_pred(detections[key]))
            label.append(self.process_label(gt_labels[key]))
            if label[-1] == []:
                pred.pop()
                label.pop()
                frame_id_to_list.pop(key)
        
        # label = label[:50]
        # pred = pred[:50]
        overlaps = calculate_iou(pred, label)

        for seq_id, frame_ids in split.items():
            if seq_id in delete_seq:
                continue
            self.gt_num = {}
            self.all_preds = {}
            for pred_class in self._class_names:
                self.all_preds[pred_class] = []
                self.gt_num[pred_class] = 0
            for frame_id in frame_ids:
                if frame_id not in frame_id_to_list:
                    # print(frame_id)
                    continue
                list_id = frame_id_to_list[frame_id]
                self.add_frame(pred[list_id], label[list_id], overlaps[list_id])
            print(f'========== seq {seq_id} ==========')
            results[seq_id] = self.print_ap()
            for value in self.all_preds.values():
                for item in value:
                    if item['type'] == 'tp':
                        center_dis.append(item['dis'])

        print(f'========== all seqs ==========')
        res_val = list(results.values())
        for key in res_val[0].keys():
            sum_ap = 0
            for item in res_val:
                sum_ap += item[key]
            if key == 'obj_num':
                print("obj_num: %.2lf" % (sum_ap / len(results)))
            else:
                print("%s: %.2lf" % (key, sum_ap / len(results) * 100))
        print("seq num: %d" % len(results))
        print("center_dis: %.4lf m" % (sum(center_dis) / len(center_dis)))

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
        pc_range = [0, -39.68, -3, 92.16, 39.68, 1]
        result = []
        for idx in range(len(info['gt_boxes'])):
            box = info['gt_boxes'][idx]
            if box[0] < pc_range[0] or box[0] > pc_range[3] or box[1] < pc_range[1] or box[1] > pc_range[4] or box[2] < pc_range[2] or box[2] > pc_range[5]:
                continue
            if self.bev:
                box[2] = box[5] = 1
            label = info['gt_names'][idx]
            score = 1.0
            result.append({'box': box, 'label': label, 'score': score})
        # if len(result) == 0:
        #     import ipdb; ipdb.set_trace()
        return result

    def print_ap(self):
        result = {}
        mAP = 0.0
        obj_num = 0
        for pred_class in self._class_names:
            ap = compute_ap(self.all_preds[pred_class], self.gt_num[pred_class])
            mAP += ap
            result[pred_class] = ap
            obj_num += self.gt_num[pred_class]
            print("%s AP: %.2lf" % (pred_class, ap * 100))
        print("mAP: %.2lf" % (mAP / len(self._class_names) * 100))
        print("obj_num: %d" % obj_num)
        result['mAP'] = mAP / len(self._class_names)
        result['obj_num'] = obj_num
        assert obj_num >= 500
        return result