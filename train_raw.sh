CUDA_VISIBLE_DEVICES=0 python train_relight_ssn.py --exp_name='general_obj' --multi_gpu --batch_size=22 --workers=22 --norm='batch_norm' --use_schedule --patience=10 --resume --weight_file="general_obj_03-August-05-30-AM.pt" --lr=1e-5 --timers=1000 --need_train --baseline --small_ds

