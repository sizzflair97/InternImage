# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/dacon.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_classes=12, in_channels=[192, 384, 768, 1536],
                     loss_decode=dict(
                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
                         class_weight=[0.00279886, 0.03645057, 0.00229447, 0.02978117, 0.07656532,
                                       0.22769068, 0.07193441, 0.00313263, 0.00197123, 0.4080042 ,
                                       1.        , 0.00461332])),
    test_cfg=dict(mode='whole'))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (270, 270)
# img_scale = (540, 960)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=False),
#     # dict(type='FishEye'),
#     # dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
#     # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.5),
#     # dict(type='RandomFlip', prob=0.5),
#     # dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]
# val_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='FishEye'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='ResizeToMultiple', size_divisor=32),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='ResizeToMultiple', size_divisor=32),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ]),
    
# ]
optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=39, layer_decay_rate=0.94,
                       depths=[5, 5, 24, 5], offset_lr_scale=1.0))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 16 GPUs with 1 images per GPU
# data = dict(samples_per_gpu=1,
#             train=dict(pipeline=train_pipeline),
#             val=dict(pipeline=val_pipeline),
#             test=dict(pipeline=test_pipeline))
data = dict(samples_per_gpu=1)
runner = dict(type='IterBasedRunner')
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))
