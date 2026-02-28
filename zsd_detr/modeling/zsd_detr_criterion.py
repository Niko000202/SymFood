import torch
import torch.nn.functional as F
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
        losses = super(ZSDDETRCriterion, self).forward(outputs, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算语义一致性损失
        if "pred_embeddings" in outputs and self.word_vec is not None:
            pred_embeddings = outputs["pred_embeddings"]
            pred_embeddings = F.normalize(pred_embeddings, dim=-1)
            word_vec = F.normalize(self.word_vec, dim=-1)
            semantic_loss = 1 - F.cosine_similarity(
                pred_embeddings.unsqueeze(2), word_vec.unsqueeze(0).unsqueeze(0), dim=-1
            ).mean()
            losses["loss_semantic"] = semantic_loss

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

        mu = torch.exp(-1.0 * mu + 0.0)

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
            "semantic": self.loss_semantic,  # 修改为直接调用 loss_semantic
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

#     def loss_semantic(self, outputs, targets, indices, num_boxes):
#         """Compute semantic consistency loss."""
#         if "pred_embeddings" not in outputs or self.word_vec is None:
#             return {"loss_semantic": torch.tensor(0.0, device=next(iter(outputs.values())).device)}
        
#         pred_embeddings = outputs["pred_embeddings"]
#         pred_embeddings = F.normalize(pred_embeddings, dim=-1)
#         word_vec = F.normalize(self.word_vec, dim=-1)
#         semantic_loss = 1 - F.cosine_similarity(
#             pred_embeddings.unsqueeze(2), word_vec.unsqueeze(0).unsqueeze(0), dim=-1
#         ).mean()
#         return {"loss_semantic": semantic_loss}
    def loss_semantic(self, outputs, targets, indices, num_boxes):
        """
        Compute semantic consistency loss specifically for matched positive predictions
        and their corresponding ground truth class semantics.
        """
        if "pred_embeddings" not in outputs or self.word_vec is None:
            return {"loss_semantic": torch.tensor(0.0, device=next(iter(outputs.values())).device)}

        # 1. 获取匹配到的源（预测）和目标（真实）的索引
        # idx[0] 是批次中每个正样本的batch_idx
        # idx[1] 是批次中每个正样本的query_idx
        idx = self._get_src_permutation_idx(indices) 
        if idx[0].numel() == 0: # 如果没有匹配到的正样本
            return {"loss_semantic": torch.tensor(0.0, device=outputs["pred_embeddings"].device)}

        # 2. 提取匹配到的预测视觉嵌入
        # outputs["pred_embeddings"] shape: (batch_size, num_queries, embed_dim)
        # target_query_embeddings shape: (num_matched_boxes, embed_dim)
        target_query_embeddings = outputs["pred_embeddings"][idx]
        target_query_embeddings = F.normalize(target_query_embeddings, p=2, dim=-1)

        # 3. 提取匹配到的真实类别标签，并获取对应的词向量
        # target_classes_o shape: (num_matched_boxes,) 包含了每个匹配框的真实类别ID
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # 从 self.word_vec (N_classes, embed_dim) 中提取目标类别的词向量
        # target_class_word_embeddings shape: (num_matched_boxes, embed_dim)
        if self.word_vec.device != target_classes_o.device: # 确保在同一设备
            self.word_vec = self.word_vec.to(target_classes_o.device)

        target_class_word_embeddings = F.normalize(self.word_vec[target_classes_o], p=2, dim=-1)

        # 4. 计算针对性对齐损失 (例如，1 - cosine_similarity)
        # 我们希望匹配到的视觉嵌入和其对应的类别词向量尽可能相似
        # cosine_similarity 会输出 [-1, 1]，值越大越相似。
        # 损失可以是 1 - similarity 或者负的similarity。
        # 这里使用 1 - similarity，最小化这个损失等同于最大化相似度。

        # 确保两个张量维度一致以便直接计算cosine_similarity
        # target_query_embeddings: (num_matched_boxes, embed_dim)
        # target_class_word_embeddings: (num_matched_boxes, embed_dim)

        # 逐元素计算余弦相似度，然后取平均
        # F.cosine_similarity 默认计算最后一个维度
        cosine_sim = F.cosine_similarity(target_query_embeddings, target_class_word_embeddings, dim=-1)

        # semantic_loss = (1 - cosine_sim).mean() # 确保有正样本才计算mean

        # 更稳健的写法，如果 cosine_sim 为空（虽然前面已经有 numel() == 0 的判断）
        if cosine_sim.numel() > 0:
            semantic_loss = (1 - cosine_sim).mean()
        else:
            semantic_loss = torch.tensor(0.0, device=target_query_embeddings.device)

        return {"loss_semantic": semantic_loss}

    def compute_dn_loss(self, dn_metas, targets, aux_num, num_boxes):
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
            if "semantic" in self.losses:
                losses["loss_semantic_dn"] = torch.as_tensor(0.0).to("cuda")

        for i in range(aux_num):
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
                if "semantic" in self.losses:
                    l_dict["loss_semantic_dn"] = torch.as_tensor(0.0).to("cuda")
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses