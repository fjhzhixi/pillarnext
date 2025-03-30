# train for debug
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python ./tools/train.py --config-name kitti_det_pp18_aspp_iou_sp \
    dataloader.train.batch_size=12 \
    dataloader.train.num_workers=8 \
    scheduler.max_lr=0.003 \
    trainer.max_epochs=20 \
    trainer.eval_every_nepochs=-1 \
    hydra.run.dir=/mnt/data5/fjh/PillarNeXt/workdir/kitti_det_pp18_aspp_iou_sp \