# package import
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # 过滤AMP弃用警告
import torch
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as torch_dist
import torch.distributed as dist
import numpy as np
import yaml as yaml
from utils.utils_loss import clip_loss
from torch.nn.utils.rnn import pad_sequence

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

from torch.nn.utils.rnn import pad_sequence


def collate_ppg_only(batch):
    # 1) PPG 动态 padding
    ppg_1d = [b["ppg"].squeeze(0) for b in batch]             # list of [L_i]
    lengths = torch.tensor([t.size(0) for t in ppg_1d], dtype=torch.long)
    Lmax = int(lengths.max().item())
    ppg_padded = pad_sequence(ppg_1d, batch_first=True, padding_value=0.0)  # [B, Lmax]
    ppg_padded = ppg_padded.unsqueeze(1).contiguous()                        # [B, 1, Lmax]
    ppg_mask_valid = torch.arange(Lmax).unsqueeze(0) < lengths.unsqueeze(1)  # [B, Lmax], True=有效

    # 2) 文本保持为原始字符串列表（绝不 pad_sequence）
    assert (ppg_mask_valid.sum(dim=1) == lengths).all()

    texts = [b["txt"] for b in batch]  # list[str]

    return {
        "ppg": ppg_padded.float(),
        "ppg_mask": ppg_mask_valid,     # True=有效；喂给 Transformer 前记得取反
        "txt": texts                  # list[str]
    }


class trainer:
    def __init__(self, model,
                 optimizer, device, model_name,lead, tp,td,**args):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.train_batch_size = args['batch_size']
        self.max_epochs = args['max_epochs']
        self.num_workers = args['num_workers']
        self.checkpoint_interval = args['checkpoint_interval']
        self.val_batch_size = args['val_batch_size']
        self.tp = tp
        self.td = td
        self.lead = lead

    # traing process
    def pretrain(self, train_dataset, val_dataset, args_dataset):
        
        train_loader = DataLoader(
                            train_dataset,
                            batch_size=self.train_batch_size,
                            shuffle=True,
                            num_workers=self.num_workers,
                            collate_fn=collate_ppg_only,
                            pin_memory=True,
                            drop_last=True
                        )
        
        val_loader = DataLoader(val_dataset, batch_size=self.val_batch_size,
                                num_workers=self.num_workers,
                                drop_last=True, shuffle=False)

        model_checkpoints_folder = os.path.join(f'../checkpoints/tp{self.tp}_td{self.td}/')
        if not os.path.exists(model_checkpoints_folder):
            print('create directory "{}" for save checkpoint!'.format(
                model_checkpoints_folder))
            print('---------------------------')
            os.makedirs(model_checkpoints_folder)
        else:
            print('directory "{}" existing for save checkpoint!'.format(
                model_checkpoints_folder))


        if os.path.exists(model_checkpoints_folder + self.model_name+'_checkpoint.pth'):
            ckpt = torch.load(model_checkpoints_folder + self.model_name+'_checkpoint.pth',
                              map_location='cpu')
            start_epoch = ckpt['epoch']
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print('continue training successful!')
        else:
            start_epoch = 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=1e-8,
        )
        niter = 1

        skip_scheduler = False
        scaler = GradScaler()

        sens_total = []
        spec_total = []
        auc_total = []

        best_auc = 0

        print(f'🚀 Pretraining Start!')
        for epoch_counter in range(start_epoch, self.max_epochs + 1):
            epoch_loss = 0
            epoch_acc1 = []
            epoch_acc5 = []
            self.model.train()

            for data in train_loader:
                self.model.train()

                report = data['txt']
                ppg = data['ppg'].to(torch.float32).to(self.device).contiguous()
                ppg_mask = data['ppg_mask'].to(torch.float32).to(self.device).contiguous()
                key_padding_mask = (~ppg_mask.bool())  # [B, L], True=pad

                self.optimizer.zero_grad()

                with autocast():
                    # 1. 文本编码
                    report_tokenize_output = unwrap_model(self.model)._tokenize(report)
                    input_ids = report_tokenize_output.input_ids.to(self.device).contiguous()
                    attention_mask = report_tokenize_output.attention_mask.to(self.device).contiguous()

                    # 2. 模型前向传播
                    output_dict = self.model(ppg, key_padding_mask, input_ids, attention_mask)

                    proj_ppg_emb = output_dict['proj_ppg_emb']
                    proj_text_emb = output_dict['proj_text_emb']
                    # ppg_emb = output_dict['ppg_emb']
                    # 直接拿到内部算好的标量 Loss (已经是多卡平均后的)
                    uma_loss = output_dict['uma_loss'].mean()
                    mse_loss = output_dict['mse_loss'].mean()
     
                    # --- Loss 1: CMA (Global PPG <-> Text 对齐) ---
                    cma_loss, acc1, acc5 = clip_loss(proj_ppg_emb, proj_text_emb, device=self.device)

                    # 总 Loss 权重分配（可以根据实验微调权重）
                    # 给 pma_loss 0.5 的权重，防止局部特征过于抢戏，干扰全局语义
                    loss =  0.1* cma_loss + 1 * uma_loss + 1 * mse_loss

                # 打印日志
                if niter % 10 == 0:
                    metrics = {
                        'loss': loss.item(),
                        'acc1': acc1.item(),
                        'acc5': acc5.item(),
                        'cma_loss': cma_loss.item(),
                        'uma_loss': uma_loss.item(),
                        'mse_loss': mse_loss.item()
                    }

                    log_message = (
                        f"[Epoch {epoch_counter}][Step {niter}] "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Acc1: {metrics['acc1']:.2f}% | "
                        f"CMA: {metrics['cma_loss']:.4f} | "
                        f"UMA: {metrics['uma_loss']:.4f} | "
                        f"MSE: {metrics['mse_loss']:.4f}"
                    )
                    print(log_message)

                epoch_loss += loss.item()
                epoch_acc1.append(acc1.item())
                epoch_acc5.append(acc5.item())

                # 反向传播更新
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if not skip_scheduler:
                    scheduler.step()

                niter += 1


            torch.save(self.model.state_dict(),
                        model_checkpoints_folder + self.model_name+f'_{epoch_counter}_ckpt.pth')
            torch.save(unwrap_model(self.model).ppg_encoder.state_dict(),
                        model_checkpoints_folder + self.model_name+f'_{epoch_counter}_encoder.pth')                
                
            if epoch_counter % self.checkpoint_interval == 0:
                self.save_checkpoints(epoch_counter, model_checkpoints_folder + self.model_name + f'_{epoch_counter}_ckpt.pth')
            
        if self.checkpoint_interval != 1:
            torch.save(unwrap_model(self.model).ppg_encoder.state_dict(),
                    model_checkpoints_folder + self.model_name + '_final_encoder.pth')
            torch.save(self.model.state_dict(),
                    model_checkpoints_folder + self.model_name + '_final_total.pth')


    def val(self, loader):
        print('📊 Validation Start')
        self.model.eval()
        val_cma_loss = 0
        val_uma_loss = 0
        val_loss = 0
        val_epoch_acc1 = []
        val_epoch_acc5 = []
        
        for data in loader:
    
            report = data['txt']
            # print(report)
            ppg = data['ppg'].to(torch.float32).to(self.device).contiguous()
            
            report_tokenize_output = unwrap_model(self.model)._tokenize(report)

            input_ids = report_tokenize_output.input_ids.to(
                self.device).contiguous()
            attention_mask = report_tokenize_output.attention_mask.to(
                self.device).contiguous()
            
            with torch.no_grad():
                output_dict = self.model(ppg, input_ids, attention_mask) 
                ppg_emb, proj_ppg_emb, proj_text_emb = output_dict['ppg_emb'],\
                                                            output_dict['proj_ppg_emb'],\
                                                            output_dict['proj_text_emb']


                world_size = torch_dist.get_world_size()
                with torch.no_grad():
                    agg_proj_img_emb = [torch.zeros_like(proj_ppg_emb[0]) for _ in range(world_size)]
                    agg_proj_text_emb = [torch.zeros_like(proj_text_emb[0]) for _ in range(world_size)]
                    
                    dist.all_gather(agg_proj_img_emb, proj_ppg_emb[0])
                    dist.all_gather(agg_proj_text_emb, proj_text_emb[0])
                    
                    agg_proj_ppg_emb1 = [torch.zeros_like(ppg_emb[0]) for _ in range(world_size)]
                    agg_proj_ppg_emb2 = [torch.zeros_like(ppg_emb[1]) for _ in range(world_size)]
                    dist.all_gather(agg_proj_ppg_emb1, ppg_emb[0])
                    dist.all_gather(agg_proj_ppg_emb2, ppg_emb[1])
                    # get current rank
                    rank = torch_dist.get_rank()

                agg_proj_img_emb[rank] = proj_ppg_emb[0]
                agg_proj_text_emb[rank] = proj_text_emb[0]
                
                agg_proj_ppg_emb1[rank] = ppg_emb[0]
                agg_proj_ppg_emb2[rank] = ppg_emb[1]

                agg_proj_img_emb = torch.cat(agg_proj_img_emb, dim=0)
                agg_proj_text_emb = torch.cat(agg_proj_text_emb, dim=0)

                agg_proj_ppg_emb1 = torch.cat(agg_proj_ppg_emb1, dim=0)
                agg_proj_ppg_emb2 = torch.cat(agg_proj_ppg_emb2, dim=0)

                cma_loss, acc1, acc5 = clip_loss(agg_proj_img_emb, agg_proj_text_emb, device=self.device)
                uma_loss, _, _ = clip_loss(agg_proj_ppg_emb1, agg_proj_ppg_emb2, device=self.device)
                loss = cma_loss + uma_loss

                # accumalate loss for logging
                val_cma_loss += cma_loss.item()
                val_uma_loss += uma_loss.item()
                val_loss += loss.item()
                val_epoch_acc1.append(acc1.item())
                val_epoch_acc5.append(acc5.item())
        
        val_metrics = {
            'loss': loss.item(),
            'acc1': acc1.item(),
            'acc5': acc5.item(),
            'cma_loss': cma_loss.item(),
            'uma_loss': uma_loss.item()
        }
        val_cma_loss = val_cma_loss/len(val_epoch_acc1)
        val_uma_loss = val_uma_loss/len(val_epoch_acc1)
        val_loss = val_loss/len(val_epoch_acc1)
        val_epoch_acc1 = np.array(val_epoch_acc1).mean()
        val_epoch_acc5 = np.array(val_epoch_acc5).mean()

        val_log = (
            "[Validation] "
            f"Loss: {val_metrics['loss']:.4f} | "
            f"Acc1: {val_metrics['acc1']:.2f}% | "
            f"Acc5: {val_metrics['acc5']:.2f}% | "
            f"CMA: {val_metrics['cma_loss']:.4f} | "
            f"UMA: {val_metrics['uma_loss']:.4f}"
        )
        print(val_log)
        return val_log

    def save_checkpoints(self, epoch, PATH):

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()},
            PATH)

