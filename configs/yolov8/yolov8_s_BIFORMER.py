_base_ = 'yolov8_s_syncbn_fast_8xb16-500e_coco.py'

log_level = 'INFO'

custom_imports = dict(
    imports=['mmyolo.models.backbones.biformer_mm'],
    allow_failed_imports=False
)

data_root = './data/plantdoc/'
class_names = (
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf',
    'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot',
    'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf',
    'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf',
    'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf',
    'Tomato leaf bacterial spot', 'Tomato leaf late blight',
    'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf',
    'Tomato two spotted spider mites leaf', 'grape leaf', 'grape leaf black rot'
)
num_classes = len(class_names)
metainfo = dict(classes=class_names)

model = dict(
    backbone=dict(
        _delete_=True,
        type='BiFormerBackbone',
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        qk_dims=[64, 128, 320, 512],  # fix: keine None-Werte mehr
        drop_path_rate=0.2,
        kv_per_wins=[2, 2, -1, -1],
        topks=[8, 8, -1, -1],
        init_cfg=None,
        n_win=8,
        out_indices=(1, 2, 3)
    ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=0.33,
        widen_factor=1.0,
        in_channels=[128, 320, 512],  # Entspricht embed_dim[1:]
        out_channels=[128, 320, 512]
    ),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            in_channels=[128, 320, 512],
            widen_factor=1.0,
            num_classes=num_classes
        )
    )
)

max_epochs = 40
train_batch_size_per_gpu = 16
train_num_workers = 2

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=False,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/')
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=train_num_workers,
    persistent_workers=False,
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        test_mode=True
    )
)

test_dataloader = val_dataloader
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox'
)
test_evaluator = val_evaluator
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5)
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    device='mps'  # oder 'cuda'
)
