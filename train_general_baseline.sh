CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --multi_gpu --vis_port=8002 --relearn --input_channel=1 --workers=72 --batch_size=72 --timers=40 --exp_name='general_baseline' --ds_folder='dataset/general_ds' --lr=1e-3 --resume --weight_file="human_baseline.pt"
