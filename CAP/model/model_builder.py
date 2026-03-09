import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def normalize(x, dim=-1):
    return x / (x.norm(dim=dim, keepdim=True) + 1e-8)

# --- 1. 辅助组件：注意力池化 ---
class AttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.size()
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        attn_mask = None
        if key_padding_mask is not None:
            cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=x.device)
            attn_mask = torch.cat([cls_mask, key_padding_mask], dim=1)
        x_cls, _ = self.attn(x[:, :1], x, x, key_padding_mask=attn_mask)
        x_cls = self.output_proj(x_cls)
        return x_cls.squeeze(1), _

# --- 2. 核心：增强型 Patch Encoder ---
class PPGPatchEncoder(nn.Module):
    def __init__(self, num_leads, d_model, nheads, num_layers):
        super().__init__()
        self.conv_stem = nn.ModuleList([
            nn.Conv1d(num_leads, d_model // 4, kernel_size=k, padding=k//2)
            for k in [7, 15, 31, 63]
        ])
        self.stem_norm = nn.GroupNorm(8, d_model)
        self.stem_act = nn.GELU()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nheads, batch_first=True, dropout=0.1, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask=None):
        stem_outs = [conv(x) for conv in self.conv_stem]
        x = torch.cat(stem_outs, dim=1)
        x = self.stem_act(self.stem_norm(x))
        x = x.permute(0, 2, 1) 
        
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).type_as(x)
            x_masked = x * valid_mask
            patch_vec = torch.sum(x_masked, dim=1) / torch.sum(valid_mask, dim=1).clamp(min=1e-5)
        else:
            patch_vec = torch.mean(x, dim=1)
        return patch_vec, x 

# --- 3. MAE 解码器 ---
class PPGReconDecoder(nn.Module):
    def __init__(self, d_model, num_leads):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_leads)
        )

    def forward(self, x):
        return self.decoder(x).permute(0, 2, 1)

# --- 4. 主模型：PPGCLIP ---
class PPGCLIP(nn.Module):
    def __init__(self, network_config, target_points=1200, target_duration=30, raw_fs=125):
        super(PPGCLIP, self).__init__()
        self.target_points = target_points
        self.target_duration = target_duration
        self.raw_fs = raw_fs
        self.target_fs = target_points / target_duration
        
        self.d_model = network_config['d_model']
        self.proj_out = network_config['projection_head']['projection_size']
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']

        self.ppg_encoder = PPGPatchEncoder(
            num_leads=network_config['num_leads'],
            d_model=self.d_model, nheads=network_config['nheads'], num_layers=network_config['num_layers']
        )
        self.recon_decoder = PPGReconDecoder(self.d_model, network_config['num_leads'])

        # 对比学习组件
        self.global_aggregator = AttentionPool(self.d_model, network_config['nheads'], self.proj_out)
        
        # 【修改】SimCSE 应该共用一个投影头，利用 Dropout 的随机性产生正样本对
        self.simcse_proj = nn.Linear(self.proj_out, self.proj_out, bias=False)
        self.dropout = nn.Dropout(p=0.1)

        # 文本端
        path = '/public/home/ai_user_1/DC/hcy/MedCPT-Query-Encoder'
        self.lm_model = AutoModel.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden), nn.GELU(), nn.Linear(self.proj_hidden, self.proj_out)
        )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))
        self.mask_ratio = 0.5 

    def _compute_clip_loss(self, q, k):
        # 如果 Batch Size 为 1，对比学习无法进行，直接返回 0
        if q.size(0) <= 1:
            return torch.tensor(0.0, device=q.device, requires_grad=True)

        q, k = normalize(q), normalize(k)
        scale = self.logit_scale.clamp(0, 4.6052).exp()
        logits = scale * q @ k.t()
        labels = torch.arange(len(logits), device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2

    def _resample_and_patch(self, ppg, mask):
        B, C, L_raw = ppg.shape
        L_target_total = int((L_raw / self.raw_fs) * self.target_fs)
        ppg_resampled = F.interpolate(ppg, size=L_target_total, mode='linear', align_corners=False)
        mask_resampled = F.interpolate(mask.unsqueeze(1).float(), size=L_target_total, mode='nearest').squeeze(1).bool()
        
        num_patches = L_target_total // self.target_points
        if num_patches > 0:
            x = ppg_resampled[:, :, :num_patches * self.target_points].view(B, C, num_patches, self.target_points)
            x = x.permute(0, 2, 1, 3).contiguous().view(B * num_patches, C, self.target_points)
            m = mask_resampled[:, :num_patches * self.target_points].view(B, num_patches, self.target_points)
            p_inv = torch.all(m, dim=2)
            m = m.view(B * num_patches, self.target_points)
        else:
            x = F.pad(ppg_resampled, (0, self.target_points - L_target_total))
            m = F.pad(mask_resampled, (0, self.target_points - L_target_total), value=True)
            p_inv = torch.all(m, dim=1, keepdim=True)
            num_patches = 1
        return x, m, num_patches, p_inv

    def forward(self, ppg, key_padding_mask, input_ids, attention_mask):
        B = ppg.shape[0]
        # 0. 标准化
        ppg = (ppg - ppg.mean(dim=-1, keepdim=True)) / (ppg.std(dim=-1, keepdim=True) + 1e-8)

        # 1. 物理对齐
        patch_inputs, patch_masks, num_patches, patch_is_all_invalid = self._resample_and_patch(ppg, key_padding_mask)

        # 2. MAE 掩码逻辑
        if self.training:
            noise = torch.rand(patch_inputs.shape[0], patch_inputs.shape[2], device=ppg.device)
            mask_pts = noise < self.mask_ratio
            combined_mask = patch_masks | mask_pts
            masked_inputs = patch_inputs.masked_fill(mask_pts.unsqueeze(1), 0.0)
        else:
            combined_mask = patch_masks
            masked_inputs = patch_inputs

        # 3. 编码与重建
        patch_vecs, seq_feats = self.ppg_encoder(masked_inputs, key_padding_mask=combined_mask)
        recon_ppg = self.recon_decoder(seq_feats)
        loss_mse = F.mse_loss(recon_ppg, patch_inputs)

        # 4. 全局表征
        patch_vecs_seq = patch_vecs.view(B, num_patches, self.d_model)
        proj_ppg_raw, _ = self.global_aggregator(patch_vecs_seq, key_padding_mask=patch_is_all_invalid)
        
        # 5. 【修复 UMA】：使用同参数层 + 不同 Dropout 实现 SimCSE
        # 这样 ppg_emb1 和 ppg_emb2 就在同一个特征空间，但有细微扰动
        ppg_emb1 = self.dropout(self.simcse_proj(proj_ppg_raw))
        ppg_emb2 = self.dropout(self.simcse_proj(proj_ppg_raw))
        loss_uma = self._compute_clip_loss(ppg_emb1, ppg_emb2)

        # 6. 文本对齐
        text_emb = self.lm_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        proj_text = normalize(self.proj_t(text_emb))

        return {
            'mse_loss': loss_mse,
            'uma_loss': loss_uma,
            'proj_ppg_emb': normalize(proj_ppg_raw),
            'proj_text_emb': proj_text
        }

    def _tokenize(self, text):
        return self.tokenizer.batch_encode_plus(
            text, add_special_tokens=True, truncation=True, 
            max_length=256, padding='max_length', return_tensors='pt'
        )