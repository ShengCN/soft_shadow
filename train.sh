CUDA_VISIBLE_DEVICES=0,1,2 python train_relight_ssn.py --multi_gpu --batch_size=68 --workers=48 --use_schedule --bilinear --prelu --ibl_num=3 --resume --weight_file='flipped.pt' --lr=1e-5
