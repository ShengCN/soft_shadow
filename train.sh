CUDA_VISIBLE_DEVICES=0,1 python train_relight_ssn.py --save --exp_name='group_norm' --batch_size=4 --multi_gpu --workers=72 --norm='group_norm' --bilinear --resume --weight_file='group_norm_15-May-07-45-PM.pt' --relearn --lr=5e-4 --timers=10 --use_schedule --patience=5
