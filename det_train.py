import paddle
import paddlex as pdx
from paddlex import transforms as T

train_transforms = T.Compose([
    T.MixupImage(alpha=1.5, beta=1.5, mixup_epoch=int(550 * 25. / 27)),
    T.RandomDistort(
        brightness_range=0.5, brightness_prob=0.5,
        contrast_range=0.5, contrast_prob=0.5,
        saturation_range=0.5, saturation_prob=0.5,
        hue_range=18.0, hue_prob=0.5),
    T.RandomExpand(prob=0.5,
                   im_padding_value=[float(int(x * 255)) for x in [0.485, 0.456, 0.406]]),
    T.RandomCrop(),
    T.Resize(target_size=320, interp='RANDOM'),
    T.RandomHorizontalFlip(prob=0.5),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transforms = T.Compose([
    T.Resize(target_size=320, interp='AREA'),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = pdx.datasets.VOCDetection(
    data_dir='DIC-C2DH-HeLa',
    file_list='DIC-C2DH-HeLa/train_list.txt',
    label_list='DIC-C2DH-HeLa/labels.txt',
    transforms=train_transforms,
    shuffle=True)

eval_dataset = pdx.datasets.VOCDetection(
    data_dir='DIC-C2DH-HeLa',
    file_list='DIC-C2DH-HeLa/val_list.txt',
    label_list='DIC-C2DH-HeLa/labels.txt',
    transforms=eval_transforms)

anchors = pdx.tools.YOLOAnchorCluster(
    num_anchors=9,
    dataset=train_dataset,
    image_size=320)()
print(anchors)

model = pdx.det.PPYOLOTiny(
    num_classes=len(train_dataset.labels),
    backbone='MobileNetV3',
    anchors=anchors)

learning_rate = 0.001
warmup_steps = 66
warmup_start_lr = 0.0
train_batch_size = 8
step_each_epoch = train_dataset.num_samples // train_batch_size

lr_decay_epochs = [130, 540]
boundaries = [b * step_each_epoch for b in lr_decay_epochs]
values = [learning_rate * (0.1 ** i) for i in range(len(lr_decay_epochs) + 1)]
lr = paddle.optimizer.lr.PiecewiseDecay(
    boundaries=boundaries,
    values=values)

lr = paddle.optimizer.lr.LinearWarmup(
    learning_rate=lr,
    warmup_steps=warmup_steps,
    start_lr=warmup_start_lr,
    end_lr=learning_rate)

optimizer = paddle.optimizer.Momentum(
    learning_rate=lr,
    momentum=0.9,
    weight_decay=paddle.regularizer.L2Decay(0.0005),
    parameters=model.net.parameters())

model.train(
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,

    num_epochs=550,
    train_batch_size=train_batch_size,
    optimizer=optimizer,

    save_interval_epochs=30,
    log_interval_steps=step_each_epoch * 5,
    save_dir=r'output/PPYOLOTiny',
    pretrain_weights=r'IMAGENET',
    use_vdl=True)
