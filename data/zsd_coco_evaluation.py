import copy
import itertools
import json
import math
import os

import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from pycocotools.cocoeval import COCOeval

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


class ZSDCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name: str, **kwargs):
        super().__init__(dataset_name=dataset_name, **kwargs)

        metadata = MetadataCatalog.get(dataset_name)
        self.seen_classes = metadata.seen_classes
        self.unseen_classes = metadata.unseen_classes

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info("Evaluation results for {}: \n".format(iou_type) + create_small_table(results))
        if not np.isfinite(sum(results.values())):
            self._logger.info("Some metrics cannot be computed and is shown as NaN.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        recalls = coco_eval.eval["recall"]
        # precision has dims (iou, recall, cls, area range, max dets)
        # recall has dims (iou, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]
        assert len(class_names) == recalls.shape[1]

        ap50_per_category = []
        ap50_per_seen_category = []
        ap50_per_unseen_category = []
        ar50_per_category = []
        ar50_per_seen_category = []
        ar50_per_unseen_category = []
        ar40_per_unseen_category = []
        ar60_per_unseen_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            # precision = precisions[:, :, idx, 0, -1]
            if precisions.shape[0] == 10:
                precision = precisions[0, :, idx, 0, -1]
            elif precisions.shape[0] == 12:
                precision = precisions[2, :, idx, 0, -1]
            else:
                raise ValueError("not support")
            precision = precision[precision > -1]
            ap50 = np.mean(precision) if precision.size else float("nan")
            ap50_per_category.append(("{}".format(name), float(ap50 * 100)))

            if recalls.shape[0] == 10:
                recall = recalls[0, idx, 0, -1]
                recall = recall[recall > -1]
                ar50 = np.mean(recall if recall.size else float("nan"))
                ar50_per_category.append(("{}".format(name), float(ar50 * 100)))

            elif recalls.shape[0] == 12:
                recall40 = recalls[0, idx, 0, -1]
                recall40 = recall40[recall40 > -1]
                ar40 = np.mean(recall40 if recall40.size else float("nan"))

                recall50 = recalls[2, idx, 0, -1]
                recall50 = recall50[recall50 > -1]
                ar50 = np.mean(recall50 if recall50.size else float("nan"))
                ar50_per_category.append(("{}".format(name), float(ar50 * 100)))

                recall60 = recalls[4, idx, 0, -1]
                recall60 = recall60[recall60 > -1]
                ar60 = np.mean(recall60 if recall60.size else float("nan"))

            else:
                raise ValueError("not support")

            if name in self.seen_classes:
                ap50_per_seen_category.append(float(ap50 * 100))
                ar50_per_seen_category.append(float(ar50 * 100))
            elif name in self.unseen_classes:
                ap50_per_unseen_category.append(float(ap50 * 100))
                ar50_per_unseen_category.append(float(ar50 * 100))
                if recalls.shape[0] == 12:
                    ar40_per_unseen_category.append(float(ar40 * 100))
                    ar60_per_unseen_category.append(float(ar60 * 100))
            else:
                raise RuntimeError("thers is some classes that neither belongs to seen nor belongs to unseen")

        ap50_per_seen_category = list(filter(lambda x: not math.isnan(x), ap50_per_seen_category))
        ar50_per_seen_category = list(filter(lambda x: not math.isnan(x), ar50_per_seen_category))
        ap50_per_unseen_category = list(filter(lambda x: not math.isnan(x), ap50_per_unseen_category))
        ar50_per_unseen_category = list(filter(lambda x: not math.isnan(x), ar50_per_unseen_category))

        ar40_per_unseen_category = list(filter(lambda x: not math.isnan(x), ar40_per_unseen_category))
        ar60_per_unseen_category = list(filter(lambda x: not math.isnan(x), ar60_per_unseen_category))

        if len(ap50_per_seen_category) > 0:
            ap50_seen = np.mean(ap50_per_seen_category)
            ar50_seen = np.mean(ar50_per_seen_category)
            self._logger.info(f"Seen-category {iou_type} AP50: {ap50_seen:.2f}")
            self._logger.info(f"Seen-category {iou_type} AR50: {ar50_seen:.2f}")

            results.update({"seen_AP50": ap50_seen, "seen_AR50": ar50_seen})

        assert len(ap50_per_unseen_category) > 0, "must have unseen category"
        if len(ap50_per_unseen_category) > 0:
            ap50_unseen = np.mean(ap50_per_unseen_category)
            ar50_unseen = np.mean(ar50_per_unseen_category)
            self._logger.info(f"Unseen-category {iou_type} AP50: {ap50_unseen:.2f}")
            self._logger.info(f"Unseen-category {iou_type} AR50: {ar50_unseen:.2f}")

            results.update({"unseen_AP50": ap50_unseen, "unseen_AR50": ar50_unseen})

            if len(ar40_per_unseen_category) > 0:
                ar40_unseen = np.mean(ar40_per_unseen_category)
                ar60_unseen = np.mean(ar60_per_unseen_category)
                self._logger.info(f"Unseen-category {iou_type} AR40: {ar40_unseen:.2f}")
                self._logger.info(f"Unseen-category {iou_type} AR60: {ar60_unseen:.2f}")

        if len(ap50_per_seen_category) > 0 and len(ap50_per_unseen_category) > 0:
            ap50_hm = 2 * ap50_seen * ap50_unseen / (ap50_seen + ap50_unseen)
            ar50_hm = 2 * ar50_seen * ar50_unseen / (ar50_seen + ar50_unseen)
            self._logger.info(f"HM {iou_type} AP50: {ap50_hm:.2f}")
            self._logger.info(f"HM {iou_type} AR50: {ar50_hm:.2f}")

            results.update({"HM_AP50": ap50_hm, "HM_AR50": ar50_hm})

        results.update({"AP-" + name: ap for name, ap in ap50_per_category})
        return results

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format("unofficial" if self._use_fast_impl else "official")
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints"}, f"Got unknown task: {task}!"
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(coco_eval, task, class_names=self._metadata.get("thing_classes"))
            self._results[task] = res


def _evaluate_predictions_on_coco(
    coco_gt,
    coco_results,
    iou_type,
    kpt_oks_sigmas=None,
    use_fast_impl=True,
    img_ids=None,
    max_dets_per_image=None,
):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = (COCOeval_opt if use_fast_impl else COCOeval)(coco_gt, coco_dt, iou_type)
    # For COCO, the default max_dets_per_image is [1, 10, 100].
    if max_dets_per_image is None:
        max_dets_per_image = [1, 10, 100]  # Default from COCOEval
    else:
        assert len(max_dets_per_image) >= 3, "COCOeval requires maxDets (and max_dets_per_image) to have length at least 3"
        # In the case that user supplies a custom input for max_dets_per_image,
        # apply COCOevalMaxDets to evaluate AP with the custom input.
        if max_dets_per_image[2] != 100:
            coco_eval = COCOevalMaxDets(coco_gt, coco_dt, iou_type)

    coco_eval.params.iouThrs = np.linspace(0.4, 0.95, int(np.round((0.95 - 0.4) / 0.05)) + 1, endpoint=True)
    if iou_type != "keypoints":
        coco_eval.params.maxDets = max_dets_per_image

    if img_ids is not None:
        coco_eval.params.imgIds = img_ids

    if iou_type == "keypoints":
        # Use the COCO default keypoint OKS sigmas unless overrides are specified
        if kpt_oks_sigmas:
            assert hasattr(coco_eval.params, "kpt_oks_sigmas"), "pycocotools is too old!"
            coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)
        # COCOAPI requires every detection and every gt to have keypoints, so
        # we just take the first entry from both
        num_keypoints_dt = len(coco_results[0]["keypoints"]) // 3
        num_keypoints_gt = len(next(iter(coco_gt.anns.values()))["keypoints"]) // 3
        num_keypoints_oks = len(coco_eval.params.kpt_oks_sigmas)
        assert num_keypoints_oks == num_keypoints_dt == num_keypoints_gt, (
            f"[COCOEvaluator] Prediction contain {num_keypoints_dt} keypoints. "
            f"Ground truth contains {num_keypoints_gt} keypoints. "
            f"The length of cfg.TEST.KEYPOINT_OKS_SIGMAS is {num_keypoints_oks}. "
            "They have to agree with each other. For meaning of OKS, please refer to "
            "http://cocodataset.org/#keypoints-eval."
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval


class COCOevalMaxDets(COCOeval):
    """
    Modified version of COCOeval for evaluating AP with a custom
    maxDets (by default for COCO, maxDets is 100)
    """

    def summarize(self):
        """
        Compute and display summary metrics for evaluation results given
        a custom value for  max_dets_per_image
        """

        def _summarize(ap=1, iouThr=None, areaRng="all", maxDets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12,))
            # Evaluate AP using the custom limit on maximum detections per image
            stats[0] = _summarize(1, maxDets=self.params.maxDets[2])
            stats[1] = _summarize(1, iouThr=0.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=0.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng="small", maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng="large", maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng="small", maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng="medium", maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng="large", maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=0.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=0.75)
            stats[3] = _summarize(1, maxDets=20, areaRng="medium")
            stats[4] = _summarize(1, maxDets=20, areaRng="large")
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=0.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=0.75)
            stats[8] = _summarize(0, maxDets=20, areaRng="medium")
            stats[9] = _summarize(0, maxDets=20, areaRng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iouType = self.params.iouType
        if iouType == "segm" or iouType == "bbox":
            summarize = _summarizeDets
        elif iouType == "keypoints":
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()
