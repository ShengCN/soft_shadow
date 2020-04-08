CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --multi_gpu --batch_size=72 --workers=48 --norm='batch_norm' --use_schedule --bilinear --ibl_num=12 --lr=1e-3 --scale_ibl
