CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --vis_port=8002 --need_train --relearn --baseline --workers=48 --batch_size=72 --timers=10 --exp_name='general_base' --ds_folder='dataset/general_ds' --lr=1e-3
