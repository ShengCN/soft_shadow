CUDA_VISIBLE_DEVICES=0 python train_relight_ssn.py --multi_gpu --batch_size=24 --workers=48 --norm='batch_norm' --use_schedule --bilinear --ibl_num=6 --lr=1e-3
