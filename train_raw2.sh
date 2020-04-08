CUDA_VISIBLE_DEVICES=2 python train_relight_ssn.py --batch_size=20 --workers=12 --norm='batch_norm' --use_schedule --bilinear --ibl_num=6 --lr=1e-3 --new_exp --exp_name='scaled_ibl' --scale_ibl
