import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List # 增加了 List

# 假设 detrex.layers 中的 MultiScaleDeformableAttention 已经可用
from detrex.layers import MultiScaleDeformableAttention
# 或者你的本地 MultiScaleDeformableAttention 实现
# from .path_to_your_ms_deform_attn import MultiScaleDeformableAttention

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if num_layers == 1:
            print(f"[DEBUG MLP Init] Creating Single Linear layer: in={input_dim}, out={output_dim}")
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            # 对于第一个 nn.Linear(n, k)
            n_first = input_dim
            k_first = h[0] # 也就是 hidden_dim
            print(f"[DEBUG MLP Init] Creating First Linear layer of Multi-layer: in={n_first}, out={k_first}")
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

# Aggregator 模块 (MLP - Optional GlobalAvgPool - MLP)
class Aggregator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, use_pool=False):
        super().__init__()
        self.use_pool = use_pool
        # 第一个MLP：将拼接后的特征进行初步处理
        self.mlp1 = MLP(input_dim, hidden_dim, hidden_dim, 2) # 2层MLP
        
        if self.use_pool:
            # 池化层：作用于 num_prompts 维度，将所有 prompts 的特征平均
            # 这会将 (bs, num_prompts, hidden_dim) -> (bs, 1, hidden_dim)
            # 如果不希望减少 prompt 数量，则不使用此池化或使用不同的池化
            # CP-DETR的图示中，Aggregator的输出似乎仍然是多个prompts，
            # 所以这里的池化可能不是全局池化所有prompts。
            # 如果池化是针对每个prompt内部的特征（例如，如果拼接的是空间特征），则方式不同。
            # 鉴于我们拼接的是不同encoder_layer的输出，这里的池化可能不直接适用。
            # 我们先假设池化是可选的，并且如果使用，是全局平均池化 num_prompts 维度。
            # 但更可能的情况是，CP-DETR的池化是针对单个prompt的特征进行的，
            # 或者Aggregator的结构比MLP-Pool-MLP更复杂。
            # 为了保持输出 (bs, num_prompts, d_model)，池化不能减少 num_prompts。
            # 另一种可能是，池化作用于拼接的特征维度上（不常见）。
            # 我们先实现一个不减少prompt数量的Aggregator：MLP1 -> MLP2
            # 如果要严格对应 MLP-Pool-MLP 并且保持 prompt 数量，
            # Pool可能是作用于每个prompt的拼接特征内部，例如AdaptiveAvgPool1d作用于最后一个维度
            # self.pool = nn.AdaptiveAvgPool1d(output_size=some_fixed_size) #作用于特征维度
            pass # 暂时不定义特定的池化，除非明确其作用方式

        # 第二个MLP：输出最终的特征维度
        self.mlp2 = MLP(hidden_dim, hidden_dim, output_dim, 2) # 再一个2层MLP

    def forward(self, x):
        # x: (bs, num_prompts, input_dim) - input_dim 是 d_model * num_encoder_layers
        
        # 通过第一个MLP
        x = self.mlp1(x) # (bs, num_prompts, hidden_dim)
        
        # 可选的池化操作
        if self.use_pool:
            # 这里的池化需要小心设计以保持 (bs, num_prompts, new_dim) 的形状
            # 例如，如果池化是用来压缩拼接的层特征的：
            # x_permuted = x.permute(0, 2, 1) # -> (bs, hidden_dim, num_prompts)
            # pooled_x = self.pool(x_permuted) # -> (bs, new_hidden_dim, num_prompts)
            # x = pooled_x.permute(0, 2, 1) # -> (bs, num_prompts, new_hidden_dim)
            # 但CP-DETR的图示更像是每个prompt独立通过MLP-Pool-MLP。
            # 为了简单，我们这里的Aggregator先不实现复杂的池化。
            # 如果论文的AvgPool指的是在每个prompt内部对特征进行操作，
            # 那么MLP的输入输出维度可能需要调整。
            # 假设这里的AvgPool是全局作用于每个prompt的特征向量的，例如在时间或序列维度（这里是num_prompts）
            # 如果是这样，它会把所有prompt聚合成一个，这可能不是期望的。
            # 若AvgPool是作用于拼接的特征维度，例如：
            # x (bs, num_prompts, hidden_dim) -> pool -> (bs, num_prompts, pooled_hidden_dim)
            # nn.AdaptiveAvgPool1d(target_output_size)可以做到，但需要permute
            pass # 暂时跳过池化实现细节

        # 通过第二个MLP
        x = self.mlp2(x) # (bs, num_prompts, output_dim)
        return x


class VisualPromptsEncoder(nn.Module):
    def __init__(self,
                 d_model=512,
                 num_encoder_layers=3, # CP-DETR中提到3层
                 num_heads=8,
                 input_coord_dim=4,
                 num_feature_levels=4,
                 dropout=0.1,
                 use_aggregator=True, # 新增：是否使用Aggregator
                 aggregator_hidden_dim=None # Aggregator中MLP的隐藏维度
                ):
        super().__init__()
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers
        self.num_feature_levels = num_feature_levels
        self.input_coord_dim = input_coord_dim
        self.use_aggregator = use_aggregator

        if self.input_coord_dim != 4:
            print(f"[提示] VisualPromptsEncoder 主要设计为处理4D坐标 (cx,cy,w,h)。"
                  f"当前 input_coord_dim={self.input_coord_dim}。请确保输入匹配。")
            if self.input_coord_dim not in [2,4]:
                 raise ValueError(f"input_coord_dim must be 2 or 4, got {self.input_coord_dim}")

        self.coordinate_embed = MLP(self.input_coord_dim, d_model, d_model, 3)
        self.ref_point_pos_embed_head = MLP(d_model, d_model, d_model, 2)
        self.encoder_layers = nn.ModuleList([
            VisualPromptEncoderLayer(
                d_model=d_model,
                d_ffn=d_model * 2,
                dropout=dropout,
                n_heads=num_heads,
                n_levels=num_feature_levels
            ) for _ in range(num_encoder_layers)])

        if self.use_aggregator:
            if self.num_encoder_layers == 0 :
                print("[警告] num_encoder_layers 为 0，但 use_aggregator 为 True。Aggregator将不会被使用。")
                self.aggregator = None
            else:
                agg_input_dim = d_model * num_encoder_layers
                _aggregator_hidden_dim = aggregator_hidden_dim if aggregator_hidden_dim is not None else d_model * 2
                print(f"[DEBUG VPE Init] d_model: {d_model}, num_encoder_layers: {num_encoder_layers}")
                print(f"[DEBUG VPE Init] Aggregator input_dim (expected): {agg_input_dim}")
                print(f"[DEBUG VPE Init] Aggregator hidden_dim: {_aggregator_hidden_dim}")
                print(f"[DEBUG VPE Init] Aggregator output_dim (to be d_model): {d_model}")
                # Aggregator的输出维度通常还是 d_model
                self.aggregator = Aggregator(input_dim=agg_input_dim,
                                             hidden_dim=_aggregator_hidden_dim,
                                             output_dim=d_model,
                                             use_pool=False) # Pool的实现比较微妙，先禁用
        else:
            self.aggregator = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 你可能需要为Aggregator中的MLP也进行特定的初始化

    def get_sine_pos_embed_from_coordinates(self, coords_normalized_2d, temperature=10000):
        if coords_normalized_2d.size(-1) != 2:
            raise ValueError("Input coordinates for sine embedding must be 2D (x,y).")
        num_pos_feats_per_dim = self.d_model // 2
        dim_t = torch.arange(num_pos_feats_per_dim, dtype=torch.float32, device=coords_normalized_2d.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats_per_dim)
        pos_x = coords_normalized_2d[..., 0:1] / dim_t.view(1, 1, -1)
        pos_y = coords_normalized_2d[..., 1:2] / dim_t.view(1, 1, -1)
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        return torch.cat((pos_x, pos_y), dim=-1)


    def forward(self, visual_prompts_coords, multi_scale_features, multi_scale_masks,
                multi_scale_pos_embeds, multi_scale_spatial_shapes, multi_scale_level_start_index
               ):
        bs, num_prompts, coord_actual_dim = visual_prompts_coords.shape
        if coord_actual_dim != self.input_coord_dim:
            raise ValueError(f"输入坐标维度 {coord_actual_dim} 与模型期望的 input_coord_dim {self.input_coord_dim} 不匹配")

        current_prompt_features = self.coordinate_embed(visual_prompts_coords)

        if self.input_coord_dim == 4:
            reference_points_xy = visual_prompts_coords[..., :2].clone().detach()
        elif self.input_coord_dim == 2:
            reference_points_xy = visual_prompts_coords.clone().detach()
        else:
            raise NotImplementedError(f"Coordinate dimension {self.input_coord_dim} not handled.")
        
        reference_points_for_attn = reference_points_xy.unsqueeze(2).repeat(1, 1, self.num_feature_levels, 1)
        query_pos_sine_embed = self.get_sine_pos_embed_from_coordinates(reference_points_xy)
        query_pos_embed = self.ref_point_pos_embed_head(query_pos_sine_embed)

        if self.num_encoder_layers == 0: # 如果没有encoder layer，直接返回坐标嵌入
            return current_prompt_features

        intermediate_layer_outputs: List[torch.Tensor] = [] # 类型提示

        for layer in self.encoder_layers:
            current_prompt_features = layer(
                query=current_prompt_features,
                query_pos=query_pos_embed,
                value=multi_scale_features,
                value_pos=multi_scale_pos_embeds,
                value_padding_mask=multi_scale_masks,
                reference_points=reference_points_for_attn,
                spatial_shapes=multi_scale_spatial_shapes,
                level_start_index=multi_scale_level_start_index
            )
            if self.use_aggregator and self.aggregator is not None: # 仅当使用aggregator时收集
                intermediate_layer_outputs.append(current_prompt_features)

        if self.use_aggregator and self.aggregator is not None and intermediate_layer_outputs:
            # 将所有中间层的输出在特征维度上拼接
            # (bs, num_prompts, d_model) * num_layers -> (bs, num_prompts, d_model * num_layers)
            concatenated_outputs = torch.cat(intermediate_layer_outputs, dim=-1)
            
            # 通过Aggregator处理
            final_concept_prompts = self.aggregator(concatenated_outputs)
            return final_concept_prompts
        else:
            # 如果不使用Aggregator，或者没有encoder_layers，则返回最后一层的输出（或初始嵌入）
            return current_prompt_features

class VisualPromptEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 d_ffn=512,
                 dropout=0.1,
                 n_heads=8,
                 n_levels=4,
                 n_points=4):
        super().__init__()
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            num_levels=n_levels,
            num_points=n_points,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = MLP(d_model, d_ffn, d_model, 2)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, query_pos, value, value_pos, value_padding_mask,
                reference_points, spatial_shapes, level_start_index):
        q_with_pos = query + query_pos

        attn_output = self.cross_attn(
            query=q_with_pos.to('cuda:0'),
            value=value.to('cuda:0'),
            key_padding_mask=value_padding_mask.to('cuda:0'),
            reference_points=reference_points.to('cuda:0'),
            spatial_shapes=spatial_shapes.to('cuda:0'),
            level_start_index=level_start_index.to('cuda:0')
        )
        query = query + self.dropout1(attn_output) # type: ignore
        query = self.norm1(query)

        ffn_output = self.ffn(query)
        query = query + self.dropout2(ffn_output) # type: ignore
        query = self.norm2(query)
        return query