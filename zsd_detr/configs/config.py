import copy

import torch.nn as nn
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BasicStem, ResNet
from detectron2.solver import WarmupParamScheduler
from detrex.config import get_config
from detrex.layers import PositionEmbeddingSine
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from fvcore.common.param_scheduler import MultiStepParamScheduler

from data import ZSDCOCOEvaluator
from zsd_detr.modeling import (
    ZSDDETR,
    ZSDDETRCriterion,
    ZSDDETRTransformer,
    ZSDDETRTransformerDecoder,
    ZSDDETRTransformerEncoder,
)

_eval_period = 2000
_test_mode = "unseen"
_train_dataset_name = f"instances_train2014_seen_48_17"
_test_dataset_name = f"instances_val2014_{_test_mode}_48_17"

dataloader = get_config("common/data/coco_detr.py").dataloader

dataloader.train.dataset.names = _train_dataset_name
dataloader.test.dataset.names = _test_dataset_name
dataloader.evaluator = L(ZSDCOCOEvaluator)(dataset_name=_test_dataset_name)

model = L(ZSDDETR)(
    tau=20.0,
    norm_features=True,
    nms_iou_threshold=0.7,
    test_mode=_test_mode,
    train_dataset_name=_train_dataset_name,
    test_dataset_name=_test_dataset_name,
    seen_vec_path="datasets/coco/word_vector/48_17_seen.pth",
    unseen_vec_path="datasets/coco/word_vector/48_17_unseen.pth",
    backbone=L(ResNet)(
        stem=L(BasicStem)(in_channels=3, out_channels=64, norm="FrozenBN"),
        stages=L(ResNet.make_default_stages)(
            depth=50,
            stride_in_1x1=False,
            norm="FrozenBN",
        ),
        out_features=["res3", "res4", "res5"],
        freeze_at=1,
    ),
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    neck=L(ChannelMapper)(
        input_shapes={
            "res3": ShapeSpec(channels=512),
            "res4": ShapeSpec(channels=1024),
            "res5": ShapeSpec(channels=2048),
        },
        in_features=["res3", "res4", "res5"],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),
    transformer=L(ZSDDETRTransformer)(
        encoder=L(ZSDDETRTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        decoder=L(ZSDDETRTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels="${..num_feature_levels}",
            use_checkpoint=False,
        ),
        num_feature_levels=4,
        two_stage_num_proposals="${..num_queries}",
        norm_features="${..norm_features}",
    ),
    embed_dim=256,
    num_classes=48,  # 80 -> 48
    num_queries=900,
    aux_loss=True,
    criterion=L(ZSDDETRCriterion)(
        num_classes="${..num_classes}",
        losses=["class", "boxes", "con"],
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
            "loss_con": 0.03,
            "loss_con_dn": 0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
        two_stage_binary_cls=False,
    ),
    dn_number=100,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    pixel_mean=[123.675, 116.280, 103.530],
    pixel_std=[58.395, 57.120, 57.375],
    vis_period=0,
    input_format="RGB",
    device="cuda",
)

# set aux loss weight dict
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.aux_loss:
    weight_dict = model.criterion.weight_dict
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict

# get default config
optimizer = get_config("common/optim.py").AdamW
scheduler = L(MultiStepParamScheduler)(
    values=[1.0],
    milestones=[12 * 7500],
)
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=scheduler,
    warmup_length=0,
    warmup_method="linear",
    warmup_factor=0.001,
)

train = get_config("common/train.py").train

# modify training config
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"

train.output_dir = "./output/coco_48_17_v"

# max training iterations
train.max_iter = 30000

# run evaluation every 5000 iters
train.eval_period = _eval_period

# log training infomation every 20 iters
train.log_period = 20

# save checkpoint every 5000 iters
train.checkpointer.period = _eval_period

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 8

# mixed precision
train.amp.enabled = False

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 2 * 4

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir

train.model_ema.enabled = False

# for fast debug
# train.fast_dev_run.enabled = True
