CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --multi_gpu --vis_port=8002 --relearn --workers=72 --batch_size=72 --timers=60 --exp_name='human_baseline' --ds_folder='dataset/human_ds' --resume --weight_file="human_baseline.pt" --lr=1e-3
