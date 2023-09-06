# dataset settings
dataset_type = 'DaconDataset'
data_root = 'data/dacon'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (940, 540)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
train_dataloader = dict(  # Train dataloader config
    batch_size=2,  # Batch size of a single GPU
    num_workers=2,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(type='InfiniteSampler', shuffle=True),  # Randomly shuffle during training.
    dataset=dict(  # Train dataset config
        type=dataset_type,  # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root,  # The root of dataset.
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),  # Prefix for training data.
        pipeline=train_pipeline)) # Processing pipeline. This is passed by the train_pipeline created before.

val_dataloader = dict(
    batch_size=1,  # Batch size of a single GPU
    num_workers=4,  # Worker to pre-fetch data for each single GPU
    persistent_workers=True,  # Shut down the worker processes after an epoch end, which can accelerate testing speed.
    sampler=dict(type='DefaultSampler', shuffle=False),  # Not shuffle during validation and testing.
    dataset=dict(  # Test dataset config
        type=dataset_type,  # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root,  # The root of dataset.
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),  # Prefix for testing data.
        pipeline=test_pipeline))  # Processing pipeline. This is passed by the test_pipeline created before.
test_dataloader = val_dataloader
# The metric to measure the accuracy. Here, we use IoUMetric.
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            img_dir='img_dir/train',
            ann_dir='ann_dir/train',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline))
