norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='MSCAN',
        embed_dims=[32, 64, 160, 256],
        mlp_ratios=[8, 8, 4, 4],
        drop_rate=0.0,
        drop_path_rate=0.1,
        depths=[3, 3, 5, 2],
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='pretrained/mscan_t.pth')),
    decode_head=dict(
        type='LightHamHead',
        in_channels=[64, 160, 256],
        in_index=[1, 2, 3],
        channels=256,
        ham_channels=256,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        ham_kwargs=dict(MD_R=16)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'CelebAMaskHQDataset'
data_root = './data/CelebAMaskHQ'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(1.0, 1.0)),
    dict(type='RandomFlip', prob=0),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        img_ratios=[0.5],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip', prob=0),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='RepeatDataset',
        times=50,
        dataset=dict(
            type='CelebAMaskHQDataset',
            data_root='./data/CelebAMaskHQ',
            img_dir='CelebA-HQ-img/',
            ann_dir='CelebA-HQ-mask/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(
                    type='Resize',
                    img_scale=(512, 512),
                    ratio_range=(1.0, 1.0)),
                dict(type='RandomFlip', prob=0),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ],
            split='train_split.txt')),
    val=dict(
        type='CelebAMaskHQDataset',
        data_root='./data/CelebAMaskHQ',
        img_dir='CelebA-HQ-img/',
        ann_dir='CelebA-HQ-mask/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                img_ratios=[0.5],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', prob=0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='val_split.txt'),
    test=dict(
        type='CelebAMaskHQDataset',
        data_root='./data/CelebAMaskHQ',
        img_dir='CelebA-HQ-img/',
        ann_dir='CelebA-HQ-mask/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 1024),
                img_ratios=[0.5],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', prob=0),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='val_split.txt'))
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))))
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=5)
evaluation = dict(interval=10000, metric='mIoU', save_best='mIoU')
find_unused_parameters = True
work_dir = './work_dirs/segnext.tiny.512x512.celebamaskhq.160k'
gpu_ids = [0]
auto_resume = False
