CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --multi_gpu --vis_port=8002 --relearn --from_baseline --touch_loss --workers=72 --batch_size=4 --timers=20 --exp_name='general_touchloss' --ds_folder='dataset/general_ds' --lr=1e-3 
