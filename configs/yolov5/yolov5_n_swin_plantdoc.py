_base_ = 'yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py'

log_level = 'INFO'

# === Dataset ===
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

# === Anchors (kleinere für YOLOv5n) ===
anchors = [
    [(10, 13), (16, 30), (33, 23)],
    [(30, 61), (62, 45), (59, 119)],
    [(116, 90), (156, 198), (373, 326)]
]

# === Trainingsparameter ===
max_epochs = 40
train_batch_size_per_gpu = 16
train_num_workers = 2

# === Swin Tiny Backbone mit YOLOv5n-Größe ===
checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'

model = dict(
    backbone=dict(
        _delete_=True,
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=0.33,  # wie bei YOLOv5n
        widen_factor=1,
        in_channels=[192, 384, 768],
        out_channels=[192, 384, 768],
    ),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            in_channels=[192, 384, 768],
            widen_factor=1,
            num_classes=num_classes
        ),
        prior_generator=dict(base_sizes=anchors)
    )
)

# === Dataloaders ===
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

# === Evaluator ===
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=data_root + 'test/_annotations.coco.json',
    metric='bbox'
)
test_evaluator = val_evaluator

# === Trainingsstrategie ===
train_cfg = dict(max_epochs=max_epochs, val_interval=10)

default_hooks = dict(
    checkpoint=dict(interval=10, max_keep_ckpts=2, save_best='auto'),
    param_scheduler=dict(max_epochs=max_epochs, warmup_mim_iter=10),
    logger=dict(type='LoggerHook', interval=5)
)

# === Hardware ===
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
    device='mps'  # oder 'cpu' falls nötig
)
