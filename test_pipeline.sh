CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --exp_name='coordconv' --norm='group_norm' --vis_port=8002 --coordconv --lr=5e-6 --multi_gpu --batch_size=1 --need_train --timers=10 --flip
