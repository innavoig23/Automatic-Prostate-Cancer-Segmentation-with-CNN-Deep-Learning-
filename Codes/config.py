compile = False
crop_size = (
    256,
    256,
)
data_root = '/content/ProstateMRI/'
dataset_type = 'ProstateMRI'
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=5, type='CheckpointHook'),
    logger=dict(interval=500, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=True, interval=500, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
log_level = 'INFO'
log_processor = dict(by_epoch=True)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        ignore_index=255,
        in_channels=1024,
        in_index=2,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        num_convs=1,
        type='FCNHead'),
    backbone=dict(
        contract_dilation=True,
        depth=50,
        dilations=(
            1,
            1,
            2,
            4,
        ),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=False,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        strides=(
            1,
            2,
            1,
            1,
        ),
        style='pytorch',
        type='ResNetV1c'),
    data_preprocessor=dict(
        mean=[
            0,
            0,
            0,
        ],
        pad_val=255,
        seg_pad_val=255,
        size=(
            256,
            256,
        ),
        std=[
            1,
            1,
            1,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        act_cfg=dict(type='ReLU'),
        align_corners=False,
        channels=512,
        conv_cfg=dict(type='Conv2d'),
        dilations=(
            1,
            12,
            24,
            36,
        ),
        dropout_ratio=0.1,
        in_channels=2048,
        in_index=3,
        loss_decode=dict(alpha=0.3, beta=0.7, type='TverskyLoss'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        num_classes=2,
        type='ASPPHead'),
    pretrained='open-mmlab://resnet50_v1c',
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='BN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=5e-05, type='AdamW', weight_decay=0.01),
    type='AmpOptimWrapper')
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=5e-05, type='AdamW', weight_decay=0.01)
param_scheduler = dict(
    by_epoch=True, milestones=[
        6,
        8,
    ], type='MultiStepLR')
randomness = dict(seed=0)
resume = False
save_dir = '/content/results_mmseg/'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='stacked_dir/test/', seg_map_path='ann_dir/test/'),
        data_root='/content/ProstateMRI/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ProstateMRI'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ignore_index=255, iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=50, val_interval=2)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        data_prefix=dict(
            img_path='stacked_dir/train/', seg_map_path='ann_dir/train/'),
        data_root='/content/ProstateMRI/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                EqualizzazioneIstogrammaFlagADC=False,
                EqualizzazioneIstogrammaFlagHBV=False,
                EqualizzazioneIstogrammaFlagT2W=False,
                FiltraggioGaussianoFlagADC=False,
                FiltraggioGaussianoFlagHBV=False,
                FiltraggioGaussianoFlagT2W=False,
                MinMaxScalingFlagADC=True,
                MinMaxScalingFlagHBV=True,
                MinMaxScalingFlagT2W=True,
                ModificaContrastoFlagADC=False,
                ModificaContrastoFlagHBV=False,
                ModificaContrastoFlagT2W=False,
                SharpeningFlagADC=False,
                SharpeningFlagHBV=False,
                SharpeningFlagT2W=False,
                percContrADC=-50,
                percContrHBV=-50,
                percContrT2W=-50,
                sigmaADC=1.5,
                sigmaHBV=1.5,
                sigmaT2W=1.5,
                type='PreProcessing'),
            dict(type='LoadAnnotations'),
            dict(prob=0.5, type='RandomFlip'),
            dict(degree=(
                -15.0,
                15.0,
            ), prob=0.5, type='RandomRotate'),
            dict(type='PackSegInputs'),
        ],
        type='ProstateMRI'),
    num_workers=2,
    persistent_workers=True,
    sampler=None)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        EqualizzazioneIstogrammaFlagADC=False,
        EqualizzazioneIstogrammaFlagHBV=False,
        EqualizzazioneIstogrammaFlagT2W=False,
        FiltraggioGaussianoFlagADC=False,
        FiltraggioGaussianoFlagHBV=False,
        FiltraggioGaussianoFlagT2W=False,
        MinMaxScalingFlagADC=True,
        MinMaxScalingFlagHBV=True,
        MinMaxScalingFlagT2W=True,
        ModificaContrastoFlagADC=False,
        ModificaContrastoFlagHBV=False,
        ModificaContrastoFlagT2W=False,
        SharpeningFlagADC=False,
        SharpeningFlagHBV=False,
        SharpeningFlagT2W=False,
        percContrADC=-50,
        percContrHBV=-50,
        percContrT2W=-50,
        sigmaADC=1.5,
        sigmaHBV=1.5,
        sigmaT2W=1.5,
        type='PreProcessing'),
    dict(type='LoadAnnotations'),
    dict(prob=0.5, type='RandomFlip'),
    dict(degree=(
        -15.0,
        15.0,
    ), prob=0.5, type='RandomRotate'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='stacked_dir/val/', seg_map_path='ann_dir/val/'),
        data_root='/content/ProstateMRI/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                EqualizzazioneIstogrammaFlagADC=False,
                EqualizzazioneIstogrammaFlagHBV=False,
                EqualizzazioneIstogrammaFlagT2W=False,
                FiltraggioGaussianoFlagADC=False,
                FiltraggioGaussianoFlagHBV=False,
                FiltraggioGaussianoFlagT2W=False,
                MinMaxScalingFlagADC=True,
                MinMaxScalingFlagHBV=True,
                MinMaxScalingFlagT2W=True,
                ModificaContrastoFlagADC=False,
                ModificaContrastoFlagHBV=False,
                ModificaContrastoFlagT2W=False,
                SharpeningFlagADC=False,
                SharpeningFlagHBV=False,
                SharpeningFlagT2W=False,
                percContrADC=-50,
                percContrHBV=-50,
                percContrT2W=-50,
                sigmaADC=1.5,
                sigmaHBV=1.5,
                sigmaT2W=1.5,
                type='PreProcessing'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        type='ProstateMRI'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ignore_index=255, iou_metrics=[
        'mIoU',
        'mDice',
    ], type='IoUMetric')
visualizer = dict(
    classes=[
        'background',
        'tumor',
    ],
    dataset_name='ProstateMRI',
    name='visualizer',
    palette=[
        (
            0,
            0,
            0,
        ),
        (
            255,
            0,
            0,
        ),
    ],
    save_dir='/content/results_mmseg/',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '/content/results_mmseg/'
