import random
import torch
import os
import numpy as np
import torch
import yaml
import argparse

from exp import exp_trainer
from data_loader import pretrain_dataset
from model import model_builder

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def main():
    parser = argparse.ArgumentParser(description='PPG Clip')
    parser.add_argument('--lead', type=str, default='1', help='lead count')
    parser.add_argument('--pretrain', type=str, default='Renmin', help='pretrain dataset')
    parser.add_argument('--seed', type=int, default='42', help='Random Seed')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--dataset', type=str, default='mimic', help='[mimic, ed]')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--target_points', type=int, default=1000, help='tp')
    parser.add_argument('--target_duration', type=int, default=25, help='td')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='device ids of multile gpus')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    config_name = f'/public/home/ai_user_1/DC/hcy/PPG_Clip/config/config_{args.dataset}.yaml'
    print(config_name)

    if args.use_gpu:
        if args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = [int(id_) for id_ in args.devices.split(',')]
            args.device_ids = device_ids
            args.gpu = device_ids[0]
            device = torch.device(f"cuda:{args.gpu}")
            print(f"🟢 Using multiple GPUs: {args.device_ids}")
        else:
            device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            print(f"🟢 Using single GPU: {args.gpu}")
    else:
        device = torch.device("cpu")
        print("🟡 Using CPU")

    if not args.use_multi_gpu:
        torch.cuda.set_device(device)

    torch.cuda.empty_cache()

    config = yaml.load(open(config_name, "r"), Loader=yaml.FullLoader)
    
    if args.dataset == 'mimic':
        train_dataset = pretrain_dataset.train_MIMIC_Dataset()
        val_dataset = pretrain_dataset.val_Dataset()
    elif args.dataset == 'ed':
        train_dataset = pretrain_dataset.train_ED_Dataset()
        val_dataset = pretrain_dataset.val_Dataset()


    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    model = model_builder.PPGCLIP(config['network'],target_points=args.target_points, target_duration=args.target_duration)

    if config['network']['free_layers'] is not None:
        for layer_idx in range(int(config['network']['free_layers'])):
            for param in list(model.lm_model.encoder.layer[layer_idx].parameters()):
                param.requires_grad = False

    model = model.to(device)

    if args.use_multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        print(f"✅ Model wrapped with DataParallel, device_ids: {args.device_ids}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        **config['optimizer']['params'],
        betas=(0.9, 0.999)
    )

    trainer = exp_trainer.trainer(model=model,
                                  optimizer=optimizer,
                                  device=device,
                                  model_name=config['dataset']['dataset_name'],
                                  lead=int(args.lead),
                                  tp=args.target_points,
                                  td=args.target_duration,
                                  **config['trainer'])
    
    trainer.pretrain(train_dataset, val_dataset, args.dataset)


if __name__ == '__main__':
    main()
