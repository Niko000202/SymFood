import os
import os.path as op

from detectron2.data.datasets.coco import register_coco_instances

# fmt: off
COCO_48_17_SEEN_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 'bird', 'horse',
                           'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
                           'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                           'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave',
                           'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush')

COCO_48_17_UNSEEN_CLASSES = ('airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie', 'snowboard', 'skateboard',
                             'cup', 'knife', 'cake', 'couch', 'keyboard', 'sink', 'scissors')

COCO_48_17_SEEN_UNSEEN_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                                  'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                                  'snowboard', 'kite', 'skateboard', 'surfboard', 'bottle', 'cup', 'fork', 'knife', 'spoon',
                                  'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut',
                                  'cake', 'chair', 'couch', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                                  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                                  'toothbrush')

COCO_65_15_SEEN_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'traffic light', 'fire hydrant',
                           'stop sign', 'bench', 'bird', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe',
                           'backpack', 'umbrella', 'handbag', 'tie', 'skis', 'sports ball', 'kite', 'baseball bat',
                           'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'knife',
                           'spoon', 'bowl', 'banana', 'apple', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake',
                           'chair', 'couch', 'potted plant', 'bed', 'dining table', 'tv', 'laptop', 'remote', 'keyboard',
                           'cell phone', 'microwave', 'oven', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                           'teddy bear', 'toothbrush')

COCO_65_15_UNSEEN_CLASSES = ('airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork',
                             'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier')

COCO_65_15_SEEN_UNSEEN_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                                  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                                  'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                                  'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                                  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                                  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                                  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                                  'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

VOC_SEEN_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'cat', 'chair', 'cow', 'diningtable', 'horse',
                    'motorbike', 'person', 'pottedplant', 'sheep', 'tvmonitor')

VOC_UNSEEN_CLASSES = ('car', 'dog', 'sofa', 'train')

VOC_SEEN_UNSEEN_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                           'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
# fmt: on


def register_specific_zsd_coco_dataset(root: str, dataset_name: str):
    path_dir_annotations = op.join(root, "coco/zsd_annotations")
    path_json = op.join(path_dir_annotations, dataset_name) + ".json"

    assert op.exists(path_json), f"file {path_json} does not exist"

    if "train" in dataset_name:
        path_dir_image = op.join(root, "coco/train2014")
    else:
        path_dir_image = op.join(root, "coco/val2014")

    metadata_dict = {}
    if "48_17" in dataset_name:
        metadata_dict["seen_classes"] = COCO_48_17_SEEN_CLASSES
        metadata_dict["unseen_classes"] = COCO_48_17_UNSEEN_CLASSES
        metadata_dict["seen_unseen_classes"] = COCO_48_17_SEEN_UNSEEN_CLASSES
    elif "65_15" in dataset_name:
        metadata_dict["seen_classes"] = COCO_65_15_SEEN_CLASSES
        metadata_dict["unseen_classes"] = COCO_65_15_UNSEEN_CLASSES
        metadata_dict["seen_unseen_classes"] = COCO_65_15_SEEN_UNSEEN_CLASSES
    else:
        raise ValueError(f"dataset_name can not be {dataset_name}")

    metadata_dict["seen_idx"] = []
    metadata_dict["unseen_idx"] = []

    if "seen_unseen" in dataset_name:
        metadata_dict["thing_colors"] = [
            [255, 0, 0] if c in metadata_dict["unseen_classes"] else [0, 255, 0] for c in metadata_dict["seen_unseen_classes"]
        ]

        for i, c in enumerate(metadata_dict["seen_unseen_classes"]):
            if c in metadata_dict["seen_classes"]:
                metadata_dict["seen_idx"].append(i)
            if c in metadata_dict["unseen_classes"]:
                metadata_dict["unseen_idx"].append(i)

    elif "unseen" in dataset_name:
        metadata_dict["thing_colors"] = [[255, 0, 0] for _ in metadata_dict["unseen_classes"]]
        metadata_dict["unseen_idx"] = list(range(len(metadata_dict["unseen_classes"])))

    elif "seen" in dataset_name:
        metadata_dict["thing_colors"] = [[0, 255, 0] for _ in metadata_dict["seen_classes"]]
        metadata_dict["seen_idx"] = list(range(len(metadata_dict["seen_classes"])))
    else:
        raise ValueError(f"dataset_name can not be {dataset_name}")

    if "seen_unseen" in dataset_name:
        metadata_dict["classes_idx"] = []
        COCO_SEEN_UNSEEN_CLASSES_NEW_ORDER = (*metadata_dict["seen_classes"], *metadata_dict["unseen_classes"])
        for name in metadata_dict["seen_unseen_classes"]:
            metadata_dict["classes_idx"].append(COCO_SEEN_UNSEEN_CLASSES_NEW_ORDER.index(name))

        metadata_dict["re_classes_idx"] = []
        for name in COCO_SEEN_UNSEEN_CLASSES_NEW_ORDER:
            metadata_dict["re_classes_idx"].append(metadata_dict["seen_unseen_classes"].index(name))

        metadata_dict["num_seen_classes"] = len(metadata_dict["seen_classes"])
        metadata_dict["num_unseen_classes"] = len(metadata_dict["unseen_classes"])

    register_coco_instances(
        dataset_name,
        metadata_dict,
        path_json,
        path_dir_image,
    )


def register_specific_zsd_voc_dataset(root: str, dataset_name: str):
    path_dir_annotations = op.join(root, "voc/zsd_annotations")
    path_json = op.join(path_dir_annotations, dataset_name) + ".json"

    assert op.exists(path_json), f"file {path_json} does not exist"

    path_dir_image = op.join(root, "voc")

    metadata_dict = {
        "seen_classes": VOC_SEEN_CLASSES,
        "unseen_classes": VOC_UNSEEN_CLASSES,
        "seen_unseen_classes": VOC_SEEN_UNSEEN_CLASSES,
    }

    metadata_dict["seen_idx"] = []
    metadata_dict["unseen_idx"] = []
    if "seen_unseen" in dataset_name:
        metadata_dict["thing_colors"] = [
            [255, 0, 0] if c in metadata_dict["unseen_classes"] else [0, 255, 0] for c in metadata_dict["seen_unseen_classes"]
        ]
        for i, c in enumerate(metadata_dict["seen_unseen_classes"]):
            if c in metadata_dict["seen_classes"]:
                metadata_dict["seen_idx"].append(i)
            if c in metadata_dict["unseen_classes"]:
                metadata_dict["unseen_idx"].append(i)
    elif "unseen" in dataset_name:
        metadata_dict["thing_colors"] = [[255, 0, 0] for _ in metadata_dict["unseen_classes"]]
        metadata_dict["unseen_idx"] = list(range(len(metadata_dict["unseen_classes"])))

    elif "seen" in dataset_name:
        metadata_dict["thing_colors"] = [[0, 255, 0] for _ in metadata_dict["seen_classes"]]
        metadata_dict["seen_idx"] = list(range(len(metadata_dict["seen_classes"])))
    else:
        raise ValueError(f"dataset_name can not be {dataset_name}")

    if "seen_unseen" in dataset_name:
        metadata_dict["classes_idx"] = []
        VOC_SEEN_UNSEEN_CLASSES_NEW_ORDER = (*metadata_dict["seen_classes"], *metadata_dict["unseen_classes"])
        for name in metadata_dict["seen_unseen_classes"]:
            metadata_dict["classes_idx"].append(VOC_SEEN_UNSEEN_CLASSES_NEW_ORDER.index(name))

        metadata_dict["re_classes_idx"] = []
        for name in VOC_SEEN_UNSEEN_CLASSES_NEW_ORDER:
            metadata_dict["re_classes_idx"].append(metadata_dict["seen_unseen_classes"].index(name))

        metadata_dict["num_seen_classes"] = len(metadata_dict["seen_classes"])
        metadata_dict["num_unseen_classes"] = len(metadata_dict["unseen_classes"])

    register_coco_instances(
        dataset_name,
        metadata_dict,
        path_json,
        path_dir_image,
    )


def register_all_zsd_coco_dataset(root):
    path_dir_annotations = op.join(root, "coco/zsd_annotations")
    filenames = os.listdir(path_dir_annotations)
    for f in filenames:
        register_specific_zsd_coco_dataset(root, op.splitext(f)[0])


def register_all_zsd_voc_dataset(root):
    path_dir_annotations = op.join(root, "voc/zsd_annotations")
    filenames = os.listdir(path_dir_annotations)
    for f in filenames:
        register_specific_zsd_voc_dataset(root, op.splitext(f)[0])


if __name__.endswith(".zsd_builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
    register_all_zsd_coco_dataset(_root)
    register_all_zsd_voc_dataset(_root)
