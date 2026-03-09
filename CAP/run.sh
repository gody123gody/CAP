nohup python finetune.py --dataset fangchan --gpu_id 0 > finetune_fangchan_gpu0.log 2>&1 &

nohup python finetune.py --dataset xueya --gpu_id 1 > finetune_xueya_gpu1.log 2>&1 &

nohup python finetune.py --dataset xintiao --gpu_id 2 > finetune_xintiao_gpu2.log 2>&1 &

nohup python finetune.py --dataset huxipinlu --gpu_id 3 > finetune_huxipinlu_gpu3.log 2>&1 &

python finetune.py --dataset fangchan --gpu_id 0 --batch_size 8 --lr 0.01

python finetune.py --dataset xintiao --gpu_id 1 --batch_size 8 --lr 0.01

python finetune.py --dataset huxipinlu --gpu_id 2 --batch_size 8 --lr 0.01

python finetune.py --dataset xueya --gpu_id 0 --batch_size 8 --lr 0.01

python finetune.py --dataset xueya --gpu_id 1 --batch_size 8 --lr 0.01
