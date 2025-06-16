import numpy as np
import numba
from numba import cuda
import copy
from mmdet3d.structures.bbox_3d.lidar_box3d import LiDARInstance3DBoxes

idx_to_class = ['Car', 'Truck', 'Pedestrian', 'Cyclist']

def calculate_iou(pred, label):
    res = []
    for pred_annos, gt_annos in zip(pred, label):
        pred_boxes = np.stack([item['box'] for item in pred_annos], axis=0)
        gt_boxes = np.stack([item['box'] for item in gt_annos], axis=0)
        pred_boxes = LiDARInstance3DBoxes(pred_boxes, origin=(0.5, 0.5, 0.5))
        gt_boxes = LiDARInstance3DBoxes(gt_boxes, origin=(0.5, 0.5, 0.5))
        res.append(pred_boxes.overlaps(pred_boxes, gt_boxes, mode='iou'))
    return res


def build_label_list(annos, filt):
    res = []
    for idx, item in enumerate(annos):
        if item['label'] == filt:
            new_item = copy.deepcopy(item)
            new_item['idx'] = idx
            res.append(new_item)
    return res

def compute_type(gt_annos, pred_annos, iou_matrix, cla, iou_threshold):
    """
    Input:
        gt_annos, pred_annos: N * {'box': Array[7], 'label': str, 'score': float}
        cla:                  Str, Class of interest
        iou threshold:        Float
    Output:
        result_pred_annos:    List, [{'box': Array[8, 3], 'score': Float, 'type': 'tp'/'fp'}]
        num_gt:               Int, number of ground truths
    """
    gt_annos = build_label_list(gt_annos, filt=cla)
    pred_annos = build_label_list(pred_annos, filt=cla)
    pred_annos = sorted(pred_annos, key=lambda x: x["score"], reverse=True)
    result_pred_annos = []
    num_tp = 0
    p, q = len(pred_annos), len(gt_annos)
    # for i in range(len(pred_annos)):
    #     pred_annos[i]["id"] = i
    for gt_anno in gt_annos:
        # logger.debug("ground truth center: {}".format(np.mean(gt_anno["box"], axis=0)))
        mx = iou_threshold
        mx_pred = None
        for i in range(len(pred_annos)):
            pred_anno = pred_annos[i]

            iou = iou_matrix[pred_anno['idx']][gt_anno['idx']]

            if iou >= mx:
                mx = iou
                mx_pred = i

        if mx_pred is not None:
            result_pred_annos.append(pred_annos[mx_pred])
            del pred_annos[mx_pred]
            result_pred_annos[-1]["type"] = "tp"
            num_tp += 1
    for pred_anno in pred_annos:
        pred_anno["type"] = "fp"
        result_pred_annos.append(pred_anno)
    # logger.debug("num_tp: {}, pred: {}, gt: {}".format(num_tp, p, q))
    return result_pred_annos, len(gt_annos), num_tp


def compute_ap(pred_annos, num_gt):
    """
    Input:
        pred_annos: List, [{'box': Array[8, 3], 'score': Float, 'type': 'tp'/'fp'}]
        num_gt:     Int, number of ground truths
    Output:
        mAP:        Float, evaluation result
    DAIR-V2X 库里的代码有问题，参考https://zhuanlan.zhihu.com/p/37910324修改
    """
    if num_gt == 0:
        return 0.0
    pred_annos = sorted(pred_annos, key=lambda x: x["score"], reverse=True)
    num_tp = np.zeros(len(pred_annos))
    for i in range(len(pred_annos)):
        num_tp[i] = 0 if i == 0 else num_tp[i - 1]
        if pred_annos[i]["type"] == "tp":
            num_tp[i] += 1
    # logger.debug("num tp = {}".format(num_tp))
    precision = num_tp / np.arange(1, len(pred_annos) + 1)
    recall = num_tp / num_gt
    precision = np.concatenate(([0.], precision, [0.]))
    recall = np.concatenate(([0.], recall, [1.]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i], precision[i - 1])
    index = np.where(recall[1:] != recall[:-1])[0]
    return np.sum((recall[index + 1] - recall[index]) * precision[index + 1])