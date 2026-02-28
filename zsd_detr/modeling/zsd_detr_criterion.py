import torch
from detectron2.utils import comm
from detectron2.utils.logger import setup_logger
from detrex.utils import get_world_size, is_dist_avail_and_initialized

from .two_stage_criterion import TwoStageCriterion

logger = setup_logger(distributed_rank=comm.get_rank(), name=__name__)


class ZSDDETRCriterion(TwoStageCriterion):
    def __init__(self, **kwargs):
        super(ZSDDETRCriterion, self).__init__(**kwargs)

        self.word_vec = None

    def forward(self, outputs, targets, dn_metas=None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        losses = super(ZSDDETRCriterion, self).forward(outputs, targets)
        # import pdb;pdb.set_trace()
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses

        aux_num = 0
        if "aux_outputs" in outputs:
            aux_num = len(outputs["aux_outputs"])
        dn_losses = self.compute_dn_loss(dn_metas, targets, aux_num, num_boxes)
        losses.update(dn_losses)

        return losses

    def loss_con(self, outputs, targets, indices, num_boxes):
        assert "pred_embeddings" in outputs, f"pred_embeddings must in outputs"

        feats = outputs["pred_embeddings"]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        batch_size, num_queries, feat_dim = feats.shape
        total_queries = batch_size * num_queries
        device = feats.device

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,
            dtype=torch.int64,
            device=device,
        )
        target_classes[idx] = target_classes_o
        target_classes = target_classes.reshape(-1, 1)
        mask = torch.eq(target_classes, target_classes.T).float()

        bg_mask = (target_classes == self.num_classes).T.expand(total_queries, -1)
        rand_mask = torch.rand((total_queries, total_queries), device=bg_mask.device) < 100 / (batch_size * num_queries)
        bg_mask = bg_mask * rand_mask

        fg_mask = (target_classes != self.num_classes).T.expand(batch_size * num_queries, -1)

        feats = feats.reshape(-1, feat_dim)
        anchor_dot_contrast = torch.div(torch.matmul(feats, feats.T), 1.0 / 20)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size * num_queries, device=device)
        positives_mask = mask * logits_mask
        negatives_mask = 1.0 - mask

        tmp_vec1 = torch.cat((self.word_vec, torch.zeros((1, self.word_vec.shape[-1]), device=device)), dim=0)
        tmp_vec2 = torch.gather(tmp_vec1, dim=0, index=target_classes.expand(-1, tmp_vec1.shape[-1]))
        mu = torch.matmul(tmp_vec2, tmp_vec2.T)

        mu = torch.exp(-1.0 * mu + 0.0)  #

        # denominator = torch.sum(exp_logits * negatives_mask * mu, dim=1, keepdim=True) \
        #             + torch.sum(exp_logits * positives_mask * mu, dim=1, keepdim=True)

        denominator = torch.sum(exp_logits * fg_mask * mu, dim=1, keepdim=True) + torch.sum(
            exp_logits * bg_mask * mu, dim=1, keepdim=True
        )

        log_probs = logits - torch.log(denominator)

        fg_mask = target_classes.squeeze() != self.num_classes
        log_probs = log_probs[fg_mask]
        positives_mask = positives_mask[fg_mask]

        num_positives_per_row = torch.sum(positives_mask, dim=1)

        if torch.sum(num_positives_per_row > 0) > 0:
            log_probs = (
                torch.sum(log_probs * positives_mask, dim=1)[num_positives_per_row > 0]
                / num_positives_per_row[num_positives_per_row > 0]
            )
            loss = -log_probs
            loss = loss.mean()
        else:
            loss = torch.tensor(0.0, device=device)

        return {"loss_con": loss}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
            "con": self.loss_con,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
        """
        compute dn loss in criterion
        Args:
            dn_metas: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if dn_metas and "output_known_lbs_bboxes" in dn_metas:
            output_known_lbs_bboxes, dn_num, single_padding = (
                dn_metas["output_known_lbs_bboxes"],
                dn_metas["dn_num"],
                dn_metas["single_padding"],
            )
            dn_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"])).long().cuda()
                    t = t.unsqueeze(0).repeat(dn_num, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(dn_num)) * single_padding).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_idx.append((output_idx, tgt_idx))
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, dn_idx, num_boxes * dn_num, **kwargs))

            l_dict = {k + "_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            losses["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            losses["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
            if "con" in self.losses:
                losses["loss_con_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
            # dn aux loss
            l_dict = {}
            if dn_metas and "output_known_lbs_bboxes" in dn_metas:
                output_known_lbs_bboxes_aux = output_known_lbs_bboxes["aux_outputs"][i]
                for loss in self.losses:
                    kwargs = {}
                    if "labels" in loss:
                        kwargs = {"log": False}
                    l_dict.update(
                        self.get_loss(
                            loss,
                            output_known_lbs_bboxes_aux,
                            targets,
                            dn_idx,
                            num_boxes * dn_num,
                            **kwargs,
                        )
                    )
                l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
            else:
                l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict["loss_class_dn"] = torch.as_tensor(0.0).to("cuda")
                if "con" in self.losses:
                    l_dict["loss_con_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses
