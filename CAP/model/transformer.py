import torch
import torch.nn as nn


# class PPGTransformerEncoder(nn.Module):
#     def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=3, dropout=0.1, flag=False):
#         super().__init__()

#         self.input_proj = nn.Linear(input_dim, d_model)

#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.norm = nn.LayerNorm(d_model)
#         self.downstream = flag
#         self.seq_len = 1000

#         if self.downstream:
#             # 回归任务：用全连接层输出1个值
#             self.pooling = nn.AdaptiveAvgPool1d(1)  # 可选 MaxPool 或 Attention Pool
#             self.regressor = nn.Sequential(
#                 nn.Linear(d_model, d_model // 2),
#                 nn.ReLU(),
#                 nn.Linear(d_model // 2, 1)
#             )

#     def forward(self, x, key_padding_mask):
#         """
#         x: [B, C=1, T] -> expected as [B, T, C]
#         """
#         if x.dim() == 2:
#             x = x.unsqueeze(1)
#         x = x.transpose(1, 2)  # [B, T, C=1]
#         x = self.input_proj(x)  # [B, T, d_model]
#         x = self.transformer_encoder(x)  # [B, T, d_model]
#         x = self.norm(x)

#         if self.downstream:
#             # x: [B, T, d_model] -> [B, d_model, T] for pooling
#             x = x.transpose(1, 2)
#             x = self.pooling(x).squeeze(-1)  # [B, d_model]
#             x = self.regressor(x)  # [B, 1]
#             return x

#         return x.transpose(1, 2)


class PPGTransformerEncoder(nn.Module):
    def __init__(self, input_dim=1, d_model=128, nhead=4, num_layers=3, dropout=0.1, flag=False):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.downstream = flag
        self.seq_len = 1000

        if self.downstream:
            # 回归任务：用全连接层输出1个值
            self.pooling = nn.AdaptiveAvgPool1d(1)  # 可选 MaxPool 或 Attention Pool
            self.regressor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1)
            )

    def forward(self, x, key_padding_mask=None):
        """
        x: [B, C=1, T] -> expected as [B, T, C]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.transpose(1, 2)  # [B, T, C=1]
        x = self.input_proj(x)  # [B, T, d_model]
        
        # 在这里传入 key_padding_mask
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)  # [B, T, d_model]
        x = self.norm(x)

        # 这里用 mask 把 pad 位抹成 0
        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).type_as(x)  # [B,T,1] True=有效
            x = x * valid

        if self.downstream:
            x = x.transpose(1, 2)                  # [B, d_model, T]
            x = self.pooling(x).squeeze(-1)        # [B, d_model]
            x = self.regressor(x)                  # [B, 1]
            return x

        return x.transpose(1, 2) 
    
class PPGTransformerEncoder(nn.Module):
    def __init__(self, input_dim=3, d_model=128, nhead=4, num_layers=3, dropout=0.1, flag=False):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.downstream = flag

    def _get_derivatives(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # 计算一阶导 (VPG) 和 二阶导 (APPG)
        vpg = torch.gradient(x, dim=-1)[0] 
        appg = torch.gradient(vpg, dim=-1)[0]
        return torch.cat([x, vpg, appg], dim=1) 

    def forward(self, x, key_padding_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x_multi = self._get_derivatives(x) 
        x_multi = x_multi.transpose(1, 2)  
        x = self.input_proj(x_multi)       
        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        if key_padding_mask is not None:
            valid = (~key_padding_mask).unsqueeze(-1).type_as(x)
            x = x * valid
        return x.transpose(1, 2) 

# class PPGDownstreamModel(nn.Module):
#     def __init__(self, encoder, d_model, head_hidden=None, dropout=0.1):
#         super().__init__()
#         self.encoder = encoder
#         self.d_model = 512
#         self.head_hidden = self.d_model // 2
            
#         self.pooling = nn.AdaptiveAvgPool1d(1)

#         self.head = nn.Sequential(
#             nn.Linear(self.d_model, self.head_hidden),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(self.head_hidden, 1)
#         )
#         for p in self.encoder.parameters():
#             p.requires_grad = False

#     def forward(self, x, key_padding_mask=None):
#         feats = self.encoder(x, key_padding_mask=key_padding_mask)
#         # print(feats.shape) # (B,d_model,L)
#         pooled = self.pooling(feats).squeeze(-1)
#         # print(pooled.shape) # (B,d_model)
#         out = self.head(pooled) 
#         # print(out.shape) # (B,1)
#         return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleneckAdapter(nn.Module):
    # 轻量 Adapter：在特征维做残差瓶颈变换
    def __init__(self, d_model, bottleneck=64, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        # 初始化让 adapter 一开始接近 0，避免刚开始把特征扰乱太大
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        # x: (B, L, C)
        return x + self.drop(self.up(self.act(self.down(x))))


class AttnPooling(nn.Module):
    # attention pooling：学习每个时间点权重，然后加权求和
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, key_padding_mask=None):
        # x: (B, L, C)
        # key_padding_mask: (B, L) True=pad
        logits = self.score(x).squeeze(-1)  # (B, L)

        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask, -1e9)

        w = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, L, 1)
        pooled = (x * w).sum(dim=1)  # (B, C)
        return pooled


class PPGDownstreamModel(nn.Module):
    def __init__(
        self,
        encoder,
        d_model=512,
        head_hidden=None,
        dropout=0.1,
        adapter_bottleneck=64,
        use_adapter=True,
        use_attn_pool=True,
        freeze_encoder=True
    ):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.use_adapter = use_adapter
        self.use_attn_pool = use_attn_pool

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if head_hidden is None:
            head_hidden = max(64, self.d_model // 2)

        # adapter & pooling 都在 (B,L,C) 格式下做
        self.adapter = BottleneckAdapter(self.d_model, bottleneck=adapter_bottleneck, dropout=dropout) if use_adapter else nn.Identity()
        self.attn_pool = AttnPooling(self.d_model, dropout=dropout) if use_attn_pool else None

        # 作为退路：不开 attention pooling 就用 mean pooling
        self.pool_norm = nn.LayerNorm(self.d_model)

        self.head = nn.Sequential(
            nn.Linear(self.d_model, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1)
        )

    def _to_blc(self, feats):
        # 兼容 feats: (B,C,L) 或 (B,L,C)，统一成 (B,L,C)
        if feats.dim() != 3:
            raise ValueError(f"Expected feats dim=3, got {feats.shape}")

        if feats.shape[1] == self.d_model:
            # (B,C,L) -> (B,L,C)
            return feats.transpose(1, 2)
        if feats.shape[-1] == self.d_model:
            # (B,L,C)
            return feats
        raise ValueError(f"Cannot match d_model={self.d_model} with feats shape {feats.shape}")

    def forward(self, x, key_padding_mask=None):
        # x = self.encoder(x, key_padding_mask=key_padding_mask)  # (B,C,L) or (B,L,C)
        # x = self._to_blc(x)                                 # (B,L,C)

        x = self.adapter(x)                                 # (B,L,C)

        if self.use_attn_pool:
            pooled = self.attn_pool(feats, key_padding_mask=key_padding_mask)  # (B,C)
        else:
            # mean pooling with mask
            if key_padding_mask is None:
                pooled = feats.mean(dim=1)
            else:
                valid = (~key_padding_mask).float().unsqueeze(-1)  # (B,L,1)
                denom = valid.sum(dim=1).clamp(min=1.0)
                pooled = (feats * valid).sum(dim=1) / denom

        pooled = self.pool_norm(pooled)  # (B,C)
        out = self.head(pooled)          # (B,1)
        return out



import torch
import torch.nn as nn
import torch.nn.functional as F

class PPGDownstreamModel(nn.Module):
    def __init__(self, encoder,seq_len=1000, d_model=512,in_channels=1, hidden=128, dropout=0.1):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),

            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x, key_padding_mask=None):
        # x: (B,L) or (B,1,L) or (B,C,L)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,L)
        # 如果 x 是 (B,L,C) 这种，就转成 (B,C,L)
        if x.dim() == 3 and x.shape[1] != 1 and x.shape[1] != 3:
            # 通常你不会给这种格式；这里不强行猜，留给你自己保证输入
            pass

        h = self.feat(x)            # (B,hidden, L')
        h = self.pool(h).squeeze(-1) # (B,hidden)
        out = self.head(h)          # (B,1)
        return out