CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --exp_name='psp' --psp --norm='group_norm' --vis_port=8002 --baseline --lr=1e-3 --multi_gpu --batch_size=66 --need_train --timers=10 --use_schedule

