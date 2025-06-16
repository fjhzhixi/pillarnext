# train for debug
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python ./tools/test.py \
    --config-name kitti_det_pp18_aspp_iou_sp \
    +load_from=epoch_50.pth \
    +force_compute=True \
    hydra.run.dir=/mnt/data5/fjh/PillarNeXt/workdir/kitti_det_pp18_aspp_iou_sp_0604 \