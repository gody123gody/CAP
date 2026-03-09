import torch
import torch.nn as nn
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer
from model import transformer

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

class AttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(self, x):
        # x: [B, C, T] -> [B, T, C]
        x = x.permute(0, 2, 1)
        B, T, C = x.size()
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+T, C]
        x, attn = self.attn(x[:, :1], x, x, average_attn_weights=True)
        x = self.output_proj(x)
        return x.squeeze(1), attn  # squeeze dim 1 (not 0)

class PPGCLIPv1(torch.nn.Module):
    def __init__(self, network_config, flag = False):
        super(PPGCLIPv1, self).__init__()
        
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        self.num_leads = network_config['num_leads']
        self.flag = flag
        self.d_model = network_config['d_model']
        self.nheads = network_config['nheads']
        self.num_layers = network_config['num_layers']

        self.downconv = nn.Conv1d(in_channels=self.d_model, out_channels=self.proj_out, kernel_size=1)
        self.att_pool_head = AttentionPool(embed_dim=self.proj_out, num_heads=self.nheads, output_dim=self.proj_out)

        self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
        self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)

        self.ppg_encoder = transformer.PPGTransformerEncoder(
            input_dim=self.num_leads,
            d_model=self.d_model,  # 可调参数
            nhead=self.nheads,
            num_layers=self.num_layers,
            flag=self.flag
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        # text encoder
        self.lm_model = AutoModel.from_pretrained(
            '/public/home/ai_user_1/DC/hcy/MedCPT-Query-Encoder')
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/public/home/ai_user_1/DC/hcy/MedCPT-Query-Encoder')
        
        # text projector
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )
        
        
    def _tokenize(self, text):
        tokenizer_output = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=text,
                                                            add_special_tokens=True,
                                                            truncation=True,
                                                            max_length=256,
                                                            padding='max_length',
                                                            return_tensors='pt')

        return tokenizer_output
    
    @torch.no_grad()
    def ext_ppg_emb(self, ppg, key_padding_mask):

        ppg_emb = self.ppg_encoder(ppg, key_padding_mask)
        ppg_emb = self.downconv(ppg_emb)
        proj_ppg_emb, att_map = self.att_pool_head(ppg_emb)
        proj_ppg_emb = proj_ppg_emb.view(proj_ppg_emb.shape[0], -1)

        return proj_ppg_emb
    
    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        text_emb = self.lm_model(input_ids=input_ids,
                                 attention_mask=attention_mask).pooler_output
        return text_emb
    
    def forward(self, ppg, key_padding_mask, input_ids, attention_mask):
        ppg_emb = self.ppg_encoder(ppg, key_padding_mask)

        ppg_emb = self.downconv(ppg_emb)
        proj_ppg_emb, _ = self.att_pool_head(ppg_emb)
        proj_ppg_emb = proj_ppg_emb.view(proj_ppg_emb.shape[0], -1)

        ppg_emb = self.avgpool(ppg_emb).view(ppg_emb.shape[0], -1)
        ppg_emb1 = self.dropout1(self.linear1(ppg_emb))
        ppg_emb2 = self.dropout2(self.linear2(ppg_emb))
        

        proj_ppg_emb = normalize(proj_ppg_emb, dim=-1)

        text_emb = self.get_text_emb(input_ids, attention_mask)
        proj_text_emb = self.proj_t(text_emb.contiguous())
        proj_text_emb = normalize(proj_text_emb, dim=-1)

        return {'ppg_emb': [ppg_emb1, ppg_emb2],
                'proj_ppg_emb': [proj_ppg_emb],
                'proj_text_emb': [proj_text_emb]}


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. 生理感知 Transformer 编码器 (保持不变)
# ==========================================
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

# ==========================================
# 2. 注意力池化层 (保持不变)
# ==========================================
class AttentionPool(nn.Module):
    def __init__(self, embed_dim, num_heads, output_dim=None):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_proj = nn.Linear(embed_dim, output_dim or embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        B, T, C = x.size()
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  
        x_out, attn_map = self.attn(x[:, :1], x, x)
        x_out = self.output_proj(x_out)
        return x_out.squeeze(1), attn_map

# ==========================================
# 3. 完整的 PPG-CLIP 多模态框架 (已补全 _tokenize)
# ==========================================
class PPGCLIP(nn.Module):
    def __init__(self, network_config, flag=False):
        super(PPGCLIP, self).__init__()
        
        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']
        self.d_model = network_config['d_model']
        self.nheads = network_config['nheads']
        self.num_layers = network_config['num_layers']
        self.flag = flag

        # PPG 支路 (输入固定为 3: Raw, VPG, APPG)
        self.ppg_encoder = PPGTransformerEncoder(
            input_dim=3, 
            d_model=self.d_model,
            nhead=self.nheads,
            num_layers=self.num_layers,
            flag=self.flag
        )
        
        self.downconv = nn.Conv1d(in_channels=self.d_model, out_channels=self.proj_out, kernel_size=1)
        self.att_pool_head = AttentionPool(embed_dim=self.proj_out, num_heads=self.nheads, output_dim=self.proj_out)

        self.linear1 = nn.Linear(self.proj_out, self.proj_out, bias=False)
        self.linear2 = nn.Linear(self.proj_out, self.proj_out, bias=False)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

        # 文本支路
        model_path = network_config.get('text_model_path', '/public/home/ai_user_1/DC/hcy/MedCPT-Query-Encoder')
        self.lm_model = AutoModel.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.proj_t = nn.Sequential(
            nn.Linear(768, self.proj_hidden),
            nn.GELU(),
            nn.Linear(self.proj_hidden, self.proj_out),
        )

    # ------------------------------------------
    # 核心修复：补全预训练脚本需要的 _tokenize 方法
    # ------------------------------------------
    def _tokenize(self, text):
        """
        供外部 trainer 调用，将原始文本转换为 tensor
        """
        tokenizer_output = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=text,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
            padding='max_length',
            return_tensors='pt'
        )
        return tokenizer_output

    @torch.no_grad()
    def get_text_emb(self, input_ids, attention_mask):
        outputs = self.lm_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    def forward(self, ppg, key_padding_mask, input_ids, attention_mask):
        # 1. PPG 编码
        ppg_feat = self.ppg_encoder(ppg, key_padding_mask)
        
        # 2. 跨模态表征 (用于 CLIP 对比)
        ppg_reduced = self.downconv(ppg_feat)
        proj_ppg_emb_raw, _ = self.att_pool_head(ppg_reduced)
        proj_ppg_emb = normalize(proj_ppg_emb_raw, dim=-1)

        # 3. 内部对比表征 (SimCSE style)
        # 注意：这里我们对投影后的特征再做一次线性映射和 Dropout
        ppg_emb1 = self.dropout1(self.linear1(proj_ppg_emb_raw))
        ppg_emb2 = self.dropout2(self.linear2(proj_ppg_emb_raw))

        # 4. 文本编码
        text_raw_emb = self.get_text_emb(input_ids, attention_mask)
        proj_text_emb = self.proj_t(text_raw_emb)
        proj_text_emb = normalize(proj_text_emb, dim=-1)

        return {
            'ppg_emb': [ppg_emb1, ppg_emb2],  # 对应你原逻辑中的 simcse 对比
            'proj_ppg_emb': [proj_ppg_emb],   # 对应 CLIP 损失
            'proj_text_emb': [proj_text_emb]  # 对应 CLIP 损失
        }

# ==========================================
# 使用示例与配置
# ==========================================
# if __name__ == "__main__":
#     config = {
#         'projection_head': {'mlp_hidden_size': 512, 'projection_size': 256},
#         'd_model': 128,
#         'nheads': 4,
#         'num_layers': 3,
#         'num_leads': 1, # 外部输入的原始通道数
#         'text_model_path': 'ncbi/MedCPT-Query-Encoder' # 示例路径
#     }

#     model = PPGCLIP(config)

#     # 模拟数据
#     batch_size = 4
#     seq_len = 1000
#     dummy_ppg = torch.randn(batch_size, 1, seq_len)
#     dummy_mask = torch.zeros(batch_size, seq_len).bool() # False 表示有效
#     dummy_ids = torch.randint(0, 1000, (batch_size, 128))
#     dummy_attn = torch.ones(batch_size, 128)

#     output = model(dummy_ppg, dummy_mask, dummy_ids, dummy_attn)
#     print("PPG Embedding Shape:", output['proj_ppg_emb'].shape) # [4, 256]
#     print("Text Embedding Shape:", output['proj_text_emb'].shape) # [4, 256]
