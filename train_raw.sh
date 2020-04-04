CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --multi_gpu --batch_size=68 --workers=48 --norm='batch_norm' --use_schedule --bilinear --prelu --ibl_num=6 --lr=1e-3
