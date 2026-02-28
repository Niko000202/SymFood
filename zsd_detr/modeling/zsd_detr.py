import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import json # 用于读取新的JSON文件
import os
from PIL import Image # 用于加载图像 (或者你使用的数据加载库中的图像加载功能)
from torchvision import transforms # 如果需要图像预处理
import copy
from typing import List, Optional
import time
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, MultiScaleDeformableAttention
from detrex.utils import inverse_sigmoid
from torchvision.ops import batched_nms
from .visual_prompts import VisualPromptsEncoder
from .FFTVL import FFTBiVLFuse

class ZSDDETR(nn.Module):
    def __init__(
        self,
        tau: float,
        norm_features: bool,
        nms_iou_threshold: float,
        test_mode: str,
        train_dataset_name: str,
        test_dataset_name: str,
        seen_vec_path: str,
        unseen_vec_path: str,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int, # 总类别数
        num_queries: int,
        criterion: nn.Module,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        input_format: Optional[str] = "RGB",
        vis_period: int = 0,
        preselected_prompts_definition_path: Optional[str] = "/22zhangwenli/yolo/MM23/data/mscoco_vp/visual_prompts_mscoco_seen_48_17.json",
        precomputed_embeddings_path: Optional[str] = "/22zhangwenli/yolo/MM23/data/mscoco_vp/visual_prompts_mscoco_seen_48_17.pt",
        force_recompute_visual_prompts: bool = False,
        image_base_path_for_prompts: Optional[str] = "/22zhangwenli/yolo/MM23/dataset/coco/train2014", # 修改这里为 uecfood256 根目录
        vp_d_model: int = 512,
        vp_num_encoder_layers: int = 3,
        vp_num_heads: int = 8,
        vp_input_coord_dim: int = 4,
        vp_num_feature_levels: int = 4,
        vp_use_aggregator: bool = True,
        vp_aggregator_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"ZSDDETR model initialized on device: {self.device}")

        self.tau = tau
        self.norm_features = norm_features
        self.nms_iou_threshold = nms_iou_threshold # 存储以备后用
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.criterion = criterion.to(self.device) if criterion is not None else None
        self.aux_loss = aux_loss
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.input_format = input_format
        self.vis_period = vis_period
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation
        self.vp_d_model = vp_d_model
        # 加载文本嵌入
        seen_vec = torch.load(seen_vec_path, map_location=self.device)
        unseen_vec = torch.load(unseen_vec_path, map_location=self.device)
        vecs = [seen_vec, unseen_vec]
        for i, vec in enumerate(vecs):
            vec = vec.float()
            vec /= vec.norm(dim=-1, keepdim=True)
            vecs[i] = vec
        self.seen_vec, self.unseen_vec = vecs # (num_seen, D_text), (num_unseen, D_text)
        self.text_embed_dim = self.seen_vec.shape[-1]
        self.image_base_path_for_prompts = image_base_path_for_prompts
        assert test_mode in ("seen", "unseen", "seen_unseen")
        self.test_mode = test_mode

        self.train_metadata = MetadataCatalog.get(train_dataset_name)
        self.test_metadata = MetadataCatalog.get(test_dataset_name)
        self.FFTBiVLFuse = FFTBiVLFuse(feature_dim=256, text_dim=self.text_embed_dim,num_heads=8)
        self.vp_fuse_module = VectorFusionX_MHA(visual_dim = self.vp_d_model, text_dim = self.text_embed_dim, embed_dim = self.embed_dim, num_heads = vp_num_heads).to(self.device)
        # ------------------------------------------------------------------------------------------------
        # ---- 初始化 self.categories_in_seen_vec_order 和 self.seen_idx_in_seen_unseen_vec ----
        self.categories_in_seen_vec_order: List[int] = []
        self.seen_idx_in_seen_unseen_vec: Optional[torch.Tensor] = [1, 2, 3, 4, 8, 9, 15, 16, 19, 20, 24, 25, 27, 31, 35, 38, 42, 44, 50, 51, 52, 53, 55, 56, 57, 59, 60, 62, 65, 72, 73, 75, 78, 79, 82, 84, 85, 86, 90, 7, 23, 33, 34, 48, 54, 70, 74, 80]
        # 1. 构建 categories_in_seen_vec_order (原始JSON ID, 顺序与 self.seen_vec 一致)
        #    这依赖于 self.train_metadata.seen_classes 的顺序与 self.seen_vec 对应
        #    并且 self.train_metadata.json_file 包含用于查找ID的 'categories'
        train_json_categories = None
        if hasattr(self.train_metadata, 'json_file') and self.train_metadata.json_file and \
           os.path.exists(self.train_metadata.json_file): # 确保json_file是路径且存在
            try:
                with open(self.train_metadata.json_file, 'r') as f:
                    train_json_content = json.load(f)
                if 'categories' in train_json_content:
                    train_json_categories = train_json_content['categories']
            except Exception as e:
                print(f"[警告] __init__: 解析训练JSON文件 {self.train_metadata.json_file} 失败: {e}")
        
        if train_json_categories and hasattr(self.train_metadata, 'seen_classes'):
            name_to_id_map = {cat['name']: cat['id'] for cat in train_json_categories}
            for class_name in self.train_metadata.seen_classes: # seen_classes 是可见类的名称列表
                if class_name in name_to_id_map:
                    self.categories_in_seen_vec_order.append(name_to_id_map[class_name])
                else:
                    print(f"[警告] __init__: 训练元数据中的类别名 '{class_name}' 在其JSON的categories中未找到对应ID。")
            
            if len(self.categories_in_seen_vec_order) != self.seen_vec.size(0):
                print(f"[错误] __init__: categories_in_seen_vec_order 长度 ({len(self.categories_in_seen_vec_order)}) "
                      f"与 seen_vec ({self.seen_vec.size(0)}) 不匹配！融合逻辑可能出错。")
                self.categories_in_seen_vec_order = [] # 清空以避免后续错误使用
        else:
            if not train_json_categories: print("[错误] __init__: 无法从训练元数据加载 categories JSON 部分。")
            if not hasattr(self.train_metadata, 'seen_classes'): print("[错误] __init__: train_metadata 缺少 'seen_classes'。")
            print("[错误] __init__: 无法构建 'categories_in_seen_vec_order'。")

        # 2. 构建 self.seen_unseen_vec 和 self.seen_idx_in_seen_unseen_vec
        if test_mode == "seen_unseen":
            # ZSDDETR 的 metadata 通常已经包含了这些预处理好的索引
            if hasattr(self.test_metadata, 'classes_idx') and \
               hasattr(self.test_metadata, 'seen_idx') and \
               self.test_metadata.seen_idx is not None:
                # print(self.test_metadata)
                # time.sleep(1000)
                self.classes_idx = self.test_metadata.classes_idx
                self.re_classes_idx = self.test_metadata.re_classes_idx
                self.num_seen_classes_from_meta = self.test_metadata.num_seen_classes # 重命名以区分
                self.num_unseen_classes_from_meta = self.test_metadata.num_unseen_classes
                
                # 构建拼接的 seen_unseen_vec (按 seen, unseen 顺序)
                concatenated_vec = torch.cat((self.seen_vec, self.unseen_vec), dim=0)
                # 使用 classes_idx (从 test_metadata 获取) 重排它以匹配最终的分类顺序
                self.seen_unseen_vec = concatenated_vec[self.classes_idx]

                if len(self.seen_idx_in_seen_unseen_vec) != self.seen_vec.size(0):
                    print(f"[错误] __init__: test_metadata.seen_idx 长度 ({len(self.seen_idx_in_seen_unseen_vec)})"
                          f"与 seen_vec ({self.seen_vec.size(0)}) 长度不匹配!")
            else:
                print("[警告] __init__: test_mode='seen_unseen' 但 test_metadata 缺少 ZSD 索引。使用回退逻辑。")
                self.seen_unseen_vec = torch.cat((self.seen_vec, self.unseen_vec), dim=0)
                self.seen_idx_in_seen_unseen_vec = torch.arange(self.seen_vec.size(0), device=self.device, dtype=torch.long)
        elif test_mode == "seen":
            self.seen_unseen_vec = self.seen_vec # 此时 word_vec 就是 seen_vec
            self.seen_idx_in_seen_unseen_vec = torch.arange(self.seen_vec.size(0), device=self.device, dtype=torch.long)
        elif test_mode == "unseen":
            self.seen_unseen_vec = self.unseen_vec
            self.seen_idx_in_seen_unseen_vec = None # 没有可见类，所以没有索引
        else:
            raise ValueError(f"不支持的 test_mode: {test_mode}")
        # --- 类别ID和索引初始化结束 ---

        # 模块移动到设备
        self.backbone = backbone.to(self.device)
        self.position_embedding = position_embedding.to(self.device)
        self.neck = neck.to(self.device)
        self.transformer = transformer.to(self.device)
        
        self.label_enc = nn.Embedding(num_classes, embed_dim).to(self.device)
        self.dn_number = dn_number
        
        base_class_embed = nn.Linear(embed_dim, self.text_embed_dim).to(self.device)
        base_bbox_embed = MLP(embed_dim, embed_dim, 4, 3).to(self.device)
        num_pred = self.transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(base_class_embed) for _ in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(base_bbox_embed) for _ in range(num_pred)])
        if self.bbox_embed: # 确保 ModuleList 非空
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            for bbox_embed_layer in self.bbox_embed:
                nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.word_vec_proj = nn.Linear(self.text_embed_dim, embed_dim)

        # 视觉提示投影层
        self.vp_proj = nn.Linear(256, self.text_embed_dim).to(self.device)

        # 存储原始pixel_mean/std
        self.pixel_mean_raw_for_init = pixel_mean
        self.pixel_std_raw_for_init = pixel_std
        self.pixel_mean_tensor = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        self.pixel_std_tensor = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - self.pixel_mean_tensor) / self.pixel_std_tensor
        
        self.single_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[m / 255.0 for m in self.pixel_mean_raw_for_init],
                        std=[s / 255.0 for s in self.pixel_std_raw_for_init])
        ])

        self.visual_prompts_encoder = VisualPromptsEncoder(
            d_model=vp_d_model, num_encoder_layers=vp_num_encoder_layers, num_heads=vp_num_heads,
            input_coord_dim=vp_input_coord_dim, num_feature_levels=vp_num_feature_levels, dropout=0.1,
            use_aggregator=vp_use_aggregator, aggregator_hidden_dim=vp_aggregator_hidden_dim
        ).to(self.device)

        self.precomputed_visual_concept_prompts = None
        self.precomputed_visual_concept_prompts_path = precomputed_embeddings_path
        self.preselected_prompts_definition_path = preselected_prompts_definition_path
        # ------------------------------------------------------------------------------------------------
        # ---- 初始化 precomputed_visual_concept_prompts ----
        if self.precomputed_visual_concept_prompts_path and os.path.exists(self.precomputed_visual_concept_prompts_path) and not force_recompute_visual_prompts:
            print(f"尝试从 {self.precomputed_visual_concept_prompts_path} 加载预计算的视觉概念提示...")
            try:
                loaded_embeds = torch.load(self.precomputed_visual_concept_prompts_path, map_location='cpu')
                if isinstance(loaded_embeds, dict) and all(isinstance(k, int) and isinstance(v, torch.Tensor) for k, v in loaded_embeds.items()):
                    self.precomputed_visual_concept_prompts = {}
                    valid_load = True
                    for k, v_tensor in loaded_embeds.items():
                        if v_tensor.ndim == 2 and v_tensor.shape[0] == 32:
                            self.precomputed_visual_concept_prompts[k] = v_tensor.to(self.device)
                        else:
                            print(f"[错误] 类别 {k} 加载的嵌入形状 {v_tensor.shape} 与期望 (32, {vp_d_model}) 不符。"); valid_load = False; break
                    if valid_load: print(f"已成功加载并验证 {len(self.precomputed_visual_concept_prompts)} 个类别的预计算视觉提示。")
                    else: self.precomputed_visual_concept_prompts = None
                else: print(f"[错误] 加载的视觉提示文件 {self.precomputed_visual_concept_prompts_path} 格式不正确。"); self.precomputed_visual_concept_prompts = None
            except Exception as e: print(f"[警告] 加载预计算的视觉提示失败: {e}。"); self.precomputed_visual_concept_prompts = None
        # 强制重新计算
        else:
            if self.preselected_prompts_definition_path and os.path.exists(self.preselected_prompts_definition_path) and self.image_base_path_for_prompts:
                if force_recompute_visual_prompts: print(f"强制重新计算视觉概念提示...")
                elif self.precomputed_visual_concept_prompts is None: print(f"未找到或加载预计算文件失败。将重新计算...")
                expected_ids = self._get_expected_seen_category_ids()
                if not expected_ids: print("[错误] __init__: 无法获取期望的可见类ID列表，不能进行视觉提示预计算。")
                else:
                    computed_embeds = self._initialize_and_encode_visual_prompts(self.preselected_prompts_definition_path, self.image_base_path_for_prompts, expected_seen_category_ids=expected_ids, vp_dim_for_padding=vp_d_model)
                    if computed_embeds:
                        self.precomputed_visual_concept_prompts = computed_embeds
                        if self.precomputed_visual_concept_prompts_path:
                            print(f"计算完成。正在保存到: {self.precomputed_visual_concept_prompts_path}")
                            try:
                                save_dir = os.path.dirname(self.precomputed_visual_concept_prompts_path)
                                if save_dir and not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
                                torch.save(self.precomputed_visual_concept_prompts, precomputed_embeddings_path)
                                print(f"已成功保存到 {precomputed_embeddings_path}")
                            except Exception as e: print(f"[错误] 保存失败: {e}")
                    else: print(f"[警告] 预计算失败或结果为空。")
            elif force_recompute_visual_prompts: print(f"[错误] 强制重新计算，但定义文件或图像路径缺失。")

        # ---- 构建最终的融合概念向量 (final_concept_vec) ----
        base_vec = self.get_word_vec_ori() # 初始化
        self.fused_seen_embeddings = base_vec.clone().detach()
        if self.precomputed_visual_concept_prompts:
            # 使用 self.categories_in_seen_vec_order (原始ID, 与self.seen_vec顺序对应)
            # 和 self.seen_idx_in_seen_unseen_vec (seen类在final_concept_vec中的索引)
            if self.categories_in_seen_vec_order: # 只需要 categories_in_seen_vec_order 即可
                if len(self.categories_in_seen_vec_order) == self.seen_vec.size(0): # 确保长度匹配
                    for i, original_cat_id in enumerate(self.seen_idx_in_seen_unseen_vec):
                        text_embed = self.seen_vec[i].to(device) # 仍然基于原始 seen_vec 进行融合
                        fused_embed = text_embed # 默认是不融合
                        if original_cat_id in self.precomputed_visual_concept_prompts:
                            visual_prompts = self.precomputed_visual_concept_prompts[original_cat_id]
                            avg_visual = visual_prompts.mean(dim=0).to(device)
                            projected_visual = self.vp_proj(avg_visual)
                            if projected_visual.shape == text_embed.shape:
                                # fused_embed = (text_embed + projected_visual) / 2.0
                                # fused_embed = self.FFTBiVLFuse(avg_visual, text_embed, text_masks=None)
                                fused_embed = self.vp_fuse_module(projected_visual, text_embed)
                        # 修改目标：将融合结果直接写入 self.fused_seen_embeddings 的对应位置
                        self.fused_seen_embeddings[i] = fused_embed
                    print(f"[信息 __init__] 成功融合 {len(self.seen_idx_in_seen_unseen_vec)} 个可见类的提示到 self.fused_seen_embeddings。")
                else:
                    print(f"[警告 __init__] categories_in_seen_vec_order 长度与 seen_vec 长度不匹配，跳过融合。")
            else:
                print(f"[信息 __init__] categories_in_seen_vec_order 为空，跳过融合。")

    def _get_expected_seen_category_ids(self):
        """ 获取模型训练时所有期望的可见类的原始 category_id 列表。
            这个列表的顺序应该与 self.seen_vec 的行顺序一致。
        """
        # 你必须根据你的项目正确实现这个方法！
        if hasattr(self, 'categories_in_seen_vec_order') and self.categories_in_seen_vec_order:
            # 如果你已经在 __init__ 中正确设置了这个属性（例如，从 train_metadata）
            return self.categories_in_seen_vec_order
        
        # --- 下面是一个更通用的尝试，但你可能需要适配 ---
        # 尝试从 self.train_metadata 构建 (这部分逻辑应该在 __init__ 的早期完成并存为属性)
        # print("[警告] _get_expected_seen_category_ids: 正在尝试从元数据动态构建，建议在__init__中预先设置好 'self.categories_in_seen_vec_order'")
        if hasattr(self.train_metadata, 'json_file') and \
           self.train_metadata.json_file and \
           'categories' in self.train_metadata.json_file and \
           hasattr(self.train_metadata, 'seen_classes'): # seen_classes 是可见类名称列表，顺序应与seen_vec一致

            name_to_id_map = {cat['name']: cat['id'] for cat in self.train_metadata.json_file['categories']}
            expected_ids = []
            for class_name in self.train_metadata.seen_classes:
                if class_name in name_to_id_map:
                    expected_ids.append(name_to_id_map[class_name])
                else:
                    print(f"[错误] _get_expected_seen_category_ids: 类别名 '{class_name}' 在JSON的categories中未找到ID。")
                    return [] # 或者抛出错误
            if len(expected_ids) != self.seen_vec.size(0):
                 print(f"[错误] _get_expected_seen_category_ids: 生成的ID列表长度与seen_vec不符。")
                 return []
            return expected_ids
        
        print("[错误] _get_expected_seen_category_ids: 无法从元数据构建期望的可见类ID列表。")
        return []

    def _initialize_and_encode_visual_prompts(self, prompts_info_path: str, image_base_path: str):
        """
        在模型初始化时调用，加载预选的视觉提示，并使用当前模型的backbone, neck,
        和 VisualPromptsEncoder 来预计算它们的嵌入。
        """
        try:
            with open(prompts_info_path, 'r') as f:
                preselected_prompts_data = json.load(f)
        except Exception as e:
            print(f"无法加载或解析预选的视觉提示JSON文件 {prompts_info_path}: {e}")
            return None

        # 将模型组件设置为评估模式，因为我们只是在做前向传播，不希望BN层等行为异常
        original_mode = {}
        components_to_eval = {'backbone': self.backbone, 'neck': self.neck, 'visual_prompts_encoder': self.visual_prompts_encoder}
        for name, component in components_to_eval.items():
            if component is not None:
                original_mode[name] = component.training
                component.eval()
        
        encoded_prompts_dict = {}
        processed_count = 0

        for category_id_str, prompts_list_for_class in preselected_prompts_data.items():
            category_id = int(category_id_str) # JSON的键通常是字符串
            class_concept_prompts_list = []
            
            print(f"为类别 {category_id} 预计算视觉提示嵌入 ({len(prompts_list_for_class)} 个)...")
            for i, prompt_info in enumerate(prompts_list_for_class):
                file_name = prompt_info["file_name"]
                # 你需要根据你的数据集结构来构建完整的图像路径
                image_path = os.path.join(image_base_path, file_name) # 例如 "datasets/uecfood256/1.jpg"
                                                                    # 这个路径可能需要根据你的 zsd_builtin.py 中的 path_dir_image 逻辑调整

                if not os.path.exists(image_path):
                    print(f"[Warning!!] 类别 {category_id} 的提示 {i} 对应的图像文件未找到: {image_path}")
                    continue

                # 准备该图像的多尺度特征 (bs=1)
                prepared_features = self._prepare_image_and_multiscale_features(image_path)
                if prepared_features is None:
                    print(f"[Warning!!] 无法为图像 {image_path} 准备特征。")
                    continue
                
                (ms_feat_flat, ms_mask_flat, ms_pos_flat, ms_shapes, ms_lvl_start_idx) = prepared_features

                # 准备视觉提示坐标 (bs=1, num_prompts=1, coord_dim=4)
                norm_cxcywh_coords = torch.tensor([prompt_info["bbox_cxcywh_norm"]], device=self.device, dtype=torch.float32)
                visual_prompts_coords_input = norm_cxcywh_coords.unsqueeze(0) # (1, 1, 4)

                with torch.no_grad():
                    concept_prompt_vector = self.visual_prompts_encoder(
                        visual_prompts_coords=visual_prompts_coords_input,
                        multi_scale_features=ms_feat_flat,
                        multi_scale_masks=ms_mask_flat,
                        multi_scale_pos_embeds=ms_pos_flat,
                        multi_scale_spatial_shapes=ms_shapes,
                        multi_scale_level_start_index=ms_lvl_start_idx
                    ) # 输出形状 (1, 1, vp_d_model)
                class_concept_prompts_list.append(concept_prompt_vector.squeeze(0).squeeze(0).to('cuda:0')) # 移到CPU以节省GPU内存
                processed_count += 1
                if processed_count % 100 == 0 :
                    print(f"已处理 {processed_count} 个视觉提示...")

            if class_concept_prompts_list:
                # 如果某个类别实际处理的提示少于32个（因为图像丢失等），你需要决定如何处理
                # 选项1: 只使用有效的，数量可能少于32
                # 选项2: 用零向量填充到32个
                # 这里我们假设都成功了，或者你已经在生成JSON时确保了这一点
                if len(class_concept_prompts_list) < 32 and len(class_concept_prompts_list) > 0 :
                     print(f"[Warning!!] 类别 {category_id} 只成功编码了 {len(class_concept_prompts_list)} 个视觉提示，期望32个。")
                     # 可以考虑填充到32个
                     while len(class_concept_prompts_list) < 32:
                         class_concept_prompts_list.append(torch.zeros_like(class_concept_prompts_list[0]))

                if len(class_concept_prompts_list) == 32: # 确保是32个
                    encoded_prompts_dict[category_id] = torch.stack(class_concept_prompts_list).to(self.device) # (32, vp_d_model)
                elif not class_concept_prompts_list: # 如果一个都没成功
                    print(f"[Warning!!] 类别 {category_id} 未能成功编码任何视觉提示。")
            else:
                print(f"[Warning!!] 类别 {category_id} 在预选JSON中，但未能处理任何提示。")
        
        # 恢复模型组件到原始的训练/评估模式
        for name, component in components_to_eval.items():
            if component is not None:
                component.train(original_mode[name])

        if not encoded_prompts_dict:
            print("[错误] 未能成功预计算任何类别的视觉提示嵌入。")
            return None
            
        return encoded_prompts_dict

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        
        # 使用在 __init__ 中准备好的 self.final_concept_vec
        # concept_vec_to_use = self.final_concept_vec.clone()
        # concept_vec_to_use = self.fused_seen_embeddings.clone().detach()

        concept_vec_to_use = self.get_word_vec().clone().detach()
        # word_vec = self.get_word_vec()
        # self.transformer.word_vec = word_vec
        # self.transformer.decoder.word_vec = word_vec
        # self.criterion.word_vec = word_vec

        self.transformer.word_vec = concept_vec_to_use
        self.transformer.decoder.word_vec = concept_vec_to_use
        self.criterion.word_vec = concept_vec_to_use

        # --- 图像特征提取 ---
        # (与你之前的 forward 方法一致)
        if self.training:
            batch_size, _, H_tensor, W_tensor = images.tensor.shape
            # img_masks 用于 Transformer Encoder 中的 padding mask
            # 0 表示有效区域, 1 表示 padding
            img_masks = images.tensor.new_ones(batch_size, H_tensor, W_tensor, dtype=torch.bool)
            for img_id, size in enumerate(images.image_sizes): # image_sizes 是原始高宽
                img_h_eff, img_w_eff = size # 使用原始尺寸来确定有效区域
                img_masks[img_id, :img_h_eff, :img_w_eff] = 0 # 有效区域为 False
        else: # eval
            img_masks = images.tensor.new_zeros(images.tensor.shape[0], images.tensor.shape[2], images.tensor.shape[3], dtype=torch.bool)

        features = self.backbone(images.tensor)
        multi_level_image_feats = self.neck(features) # List of Tensors

        multi_level_masks_for_transformer = []
        multi_level_pos_embeds_for_transformer = []
        for feat in multi_level_image_feats:
            mask_resized = F.interpolate(img_masks.unsqueeze(1).float(), size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            multi_level_masks_for_transformer.append(mask_resized)
            multi_level_pos_embeds_for_transformer.append(self.position_embedding(mask_resized))
        # --- 图像特征提取结束 ---

        # --- Denoising and Query Preparation ---
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances) # 你需要实现或确保此方法存在
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                 targets, self.dn_number, self.label_noise_ratio, self.box_noise_scale,
                 self.num_queries, self.num_classes, self.embed_dim, self.label_enc)
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta, targets = None, None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)
        # --- Denoising and Query Preparation End ---


        (inter_states, init_reference, inter_references, enc_state, enc_reference) = self.transformer(
            multi_level_image_feats,
            multi_level_masks_for_transformer,
            multi_level_pos_embeds_for_transformer,
            query_embeds,
            attn_masks=[attn_mask, None, None], # Decoder self-attn, Enc-Dec cross-attn, Enc self-attn (None)
        )
        
        inter_states[0] += self.label_enc.weight[0,0]*0.0 # hack

        # --- 计算 Decoder 输出 ---
        outputs_classes_list = []
        outputs_coords_list = []
        outputs_embeddings_list = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0: reference = init_reference
            else: reference = inter_references[lvl-1]
            reference = inverse_sigmoid(reference) # 参考点反向sigmoid
            
            current_decoder_state = inter_states[lvl]
            outputs_embedding = self.class_embed[lvl](current_decoder_state) # (bs, num_queries, text_embed_dim)
            
            if lvl == inter_states.shape[0] - 1 and self.training: # 最后一层输出的统计
                storage = get_event_storage()
                if hasattr(storage, 'iter') and storage.iter > 0:
                    storage.put_scalar("feat_norm_decoder_out", outputs_embedding.norm(dim=-1).mean())
            
            if self.norm_features: # 特征归一化
                outputs_embedding = outputs_embedding / (outputs_embedding.norm(dim=-1, keepdim=True) + 1e-7)
            
            # 使用 self.final_concept_vec (可能已融合) 进行分类
            outputs_class = self.tau * outputs_embedding @ concept_vec_to_use.t() # (bs, num_queries, total_num_classes)
            
            tmp = self.bbox_embed[lvl](current_decoder_state) # (bs, num_queries, 4)
            if reference.shape[-1] == 4: # cxcywh format
                tmp += reference
            else: # cxcy format
                assert reference.shape[-1] == 2
                tmp[...,:2] += reference
            outputs_coord = tmp.sigmoid() # (bs, num_queries, 4) -> 归一化到 [0,1]
            
            outputs_classes_list.append(outputs_class)
            outputs_coords_list.append(outputs_coord)
            outputs_embeddings_list.append(outputs_embedding) # 用于loss计算或后续分析

        outputs_class = torch.stack(outputs_classes_list)     # (num_decoder_layers, bs, num_query, total_num_classes)
        outputs_coord = torch.stack(outputs_coords_list)       # (num_decoder_layers, bs, num_query, 4)
        outputs_embedding = torch.stack(outputs_embeddings_list) # (num_decoder_layers, bs, num_query, text_embed_dim)

        if dn_meta is not None and self.training: # DN后处理仅在训练时
            outputs_class, outputs_coord, outputs_embedding = self.dn_post_process(
                outputs_class, outputs_coord, outputs_embedding, dn_meta)

        output = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
            "pred_embeddings": outputs_embedding[-1],
        }
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord, outputs_embedding)
        
        # prepare two stage output
        interm_coord = enc_reference
        interm_embedding = self.transformer.decoder.class_embed[-1](enc_state)
        if self.norm_features:
            interm_embedding = interm_embedding / interm_embedding.norm(dim=-1, keepdim=True)
        interm_class = self.tau * interm_embedding @ concept_vec_to_use.t()
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord, "pred_embeddings": interm_embedding}

        if self.training:
            loss_dict = self.criterion(output, targets, dn_meta) # targets 来自 prepare_for_cdn
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict: loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            # 推理时的后处理
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            # image_sizes 应该来自 self.preprocess_image 返回的 ImageList 对象
            results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        
    def visualize_training(self, batched_inputs, results):
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_box = 20

        for input, results_per_image in zip(batched_inputs, results):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(boxes=results_per_image.pred_boxes[:max_vis_box].tensor.detach().cpu().numpy())
            pred_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, pred_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted boxes"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_embedding):
        return [
            {
                "pred_logits": a,
                "pred_boxes": b,
                "pred_embeddings": c,
            }
            for a, b, c in zip(outputs_class[:-1], outputs_coord[:-1], outputs_embedding[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc,
    ):
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1)  # half of bbox prob
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        input_label_embed = label_enc(m)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_padding * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, outputs_embedding, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            output_known_embedding = outputs_embedding[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]
            outputs_embedding = outputs_embedding[:, :, padding_size:, :]

            out = {
                "pred_logits": output_known_class[-1],
                "pred_boxes": output_known_coord[-1],
                "pred_embeddings": output_known_embedding[-1],
            }
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord, output_known_embedding)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord, outputs_embedding

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes):
        assert len(box_cls) == len(image_sizes)
        results = []

        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1)
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            box_pred_per_image = box_cxcywh_to_xyxy(box_pred_per_image)

            if self.nms_iou_threshold < 1.0:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, self.nms_iou_threshold)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(image_size)
            result.pred_boxes = Boxes(box_pred_per_image)
            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def get_word_vec(self):
        if self.training:
            return self.fused_seen_embeddings
        else: # 测试时
            if self.test_mode == "unseen":
                return self.unseen_vec
            elif self.test_mode == "seen":
                return self.seen_vec # 测试已见类时，使用原始 seen_vec
            elif self.test_mode == "seen_unseen":
                # GZSD模式
                if hasattr(self, 'seen_unseen_vec_for_eval') and self.seen_unseen_vec_for_eval is not None: # 确保它已在 __init__ 中创建和赋值
                    return self.seen_unseen_vec_for_eval
                else: # 如果 seen_unseen_vec_for_eval 未在 __init__ 中正确初始化，则执行回退逻辑
                    # print("[警告] get_word_vec: seen_unseen_vec_for_eval 未初始化，将临时构建。")
                    
                    # 【关键修改点】: 回退逻辑中使用融合后的已见类嵌入
                    # 确保 self.fused_seen_embeddings 和 self.unseen_vec 此时是可用的
                    if self.fused_seen_embeddings is None or self.unseen_vec is None:
                        # 这是一个更深层的问题，意味着 __init__ 中有严重错误
                        raise ValueError("get_word_vec: fused_seen_embeddings 或 unseen_vec 未初始化，无法构建 GZSD 向量。")

                    concatenated_vec = torch.cat((self.fused_seen_embeddings, self.unseen_vec), dim=0)
                    
                    if hasattr(self.test_metadata, 'classes_idx'):
                        # 确保 self.test_metadata.classes_idx 也是可用的
                        if self.test_metadata.classes_idx is None:
                             print("[警告] get_word_vec (fallback): test_metadata.classes_idx 为 None，返回未排序的拼接向量。")
                             return concatenated_vec
                        try:
                            return concatenated_vec[self.test_metadata.classes_idx]
                        except IndexError as e:
                            print(f"[错误] get_word_vec (fallback): 使用 classes_idx 索引拼接向量时出错: {e}。拼接向量形状: {concatenated_vec.shape}, classes_idx 最大值: {max(self.test_metadata.classes_idx) if self.test_metadata.classes_idx else 'N/A'}")
                            return concatenated_vec # 出错则返回未排序的
                    return concatenated_vec # 无排序的回退
            else:
                raise ValueError(f"无效的 test_mode: {self.test_mode}")
    def get_word_vec_ori(self):
        if self.training:
            return self.seen_vec

        if self.test_mode == "unseen":
            return self.unseen_vec
        elif self.test_mode == "seen":
            return self.seen_vec
        else:
            return self.seen_unseen_vec

class VectorFusionX_MHA(nn.Module):
    """
    用于融合两个单一特征向量的跨模态多头注意力模块 (X-MHA)。
    
    它接收一个一维视觉特征向量作为查询(Query)，一个一维文本特征向量作为键(Key)和值(Value)，
    并输出一个经过文本信息增强（或调制）的、与输入视觉向量维度相同的向量。
    """
    def __init__(self, visual_dim, text_dim, embed_dim=512, num_heads=8):
        """
        初始化函数。

        参数:
            visual_dim (int): 输入视觉特征向量的维度。
            text_dim (int): 输入文本特征向量的维度。
            embed_dim (int): 注意力机制内部的统一嵌入维度。
            num_heads (int): 多头注意力的头的数量。
        """
        super(VectorFusionX_MHA, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. 投影层：将不同维度的输入投影到统一的嵌入维度
        self.q_proj = nn.Linear(visual_dim, embed_dim)
        self.k_proj = nn.Linear(text_dim, embed_dim)
        self.v_proj = nn.Linear(text_dim, embed_dim)

        # 2. 核心注意力层
        # 输入格式为 (序列长度, 批量大小, 嵌入维度)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

        # 3. 输出层
        self.out_proj = nn.Linear(embed_dim, visual_dim)

    def forward(self, visual_vec, text_vec):
        """
        前向传播。

        参数:
            visual_vec (torch.Tensor): 一维视觉特征张量，形状为 (visual_dim,)。
            text_vec (torch.Tensor): 一维文本特征张量，形状为 (text_dim,)。
        
        返回:
            torch.Tensor: 融合后的一维视觉特征张量，形状为 (visual_dim,)。
        """
        # --- 步骤 1: 为注意力机制准备输入形状 ---
        # nn.MultiheadAttention 期望输入包含批量和序列维度。
        # 我们将当前操作视为 batch_size=1, sequence_length=1。
        # visual_vec: (256,) -> (1, 1, 256) (序列长度=1, 批量=1, 维度=256)
        # text_vec:   (512,) -> (1, 1, 512) (序列长度=1, 批量=1, 维度=512)
        q_in = visual_vec.unsqueeze(0).unsqueeze(0)
        k_in = text_vec.unsqueeze(0).unsqueeze(0)
        v_in = k_in # Key 和 Value 通常来自同一来源

        # --- 步骤 2: 将视觉和文本向量投影到统一的嵌入空间 ---
        # Q 来自视觉, K 和 V 来自文本
        q = self.q_proj(q_in)  # 形状: (1, 1, embed_dim)
        k = self.k_proj(k_in)  # 形状: (1, 1, embed_dim)
        v = self.v_proj(v_in)  # 形状: (1, 1, embed_dim)
        
        # --- 步骤 3: 执行跨模态多头注意力 ---
        # 视觉Query (长度为1的序列) 去关注 文本Key/Value (长度为1的序列)
        attn_output, attn_weights = self.attention(query=q, key=k, value=v)
        # attn_output 形状: (1, 1, embed_dim)
        
        # --- 步骤 4: 整理输出并进行输出投影 ---
        # 移除多余的序列和批量维度
        attn_output = attn_output.squeeze(0).squeeze(0) # 形状变为: (embed_dim,)
        output = self.out_proj(attn_output)             # 形状变为: (visual_dim,)

        # --- 步骤 5: 残差连接 (可选但推荐) ---
        # 将注意力模块学习到的“修正量”加回到原始的视觉向量上
        output = visual_vec + output

        return output