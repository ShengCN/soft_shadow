import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import utils
import time
from tqdm import tqdm
import numpy as np
import os
import datetime

from ssn.ssn_dataset import SSN_Dataset
# from ssn.ssn_submodule import Contract
from ssn.ssn import Relight_SSN
from utils.net_utils import save_model, get_lr, set_lr
from utils.visdom_utils import visdom_plot_loss, visdom_relight_results, visdom_log, visdom_show_batch, visdom_show_light
from params import params as options, parse_params
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# parse args
params = parse_params()
print("Params: {}".format(params))
if params.new_exp:
    exp = 1
else:
    exp = 0
exp_name = params.exp_name

""" https://discuss.pytorch.org/t/changing-the-weight-decay-on-bias-using-named-parameters/19132/3
"""
def set_model_optimizer(model, weight_decay):
    optim_params = []
    for key, value in model.named_parameters(): 
        if not value.requires_grad: continue # frozen weights		
            
        if key[-4:] == 'bias':
            optim_params += [{'params': value,'weight_decay':0.0}]
        else:
            optim_params += [{'params': value,'weight_decay':weight_decay}]
    
    # import pdb; pdb.set_trace()
    optimizer = optim.Adam(optim_params, 
                           lr=params.lr, 
                           betas=(params.beta1, 0.999), 
                           eps=1e-5)
    # optimizer = optim.SGD(optim_params, lr=params.lr,  momentum=0.9)
    return optimizer
    
def reconstruct_loss(gt_img, pred_img):
    """ M * (I-I') """
    return torch.norm(gt_img-pred_img, 2)

def get_grid_img(tensor_img):
    return utils.make_grid(tensor_img).detach().cpu().unsqueeze(0)

def visdom_plot_img(I_t, predicted_img, mask, L_t, is_training=True, save_batch=False):
    batch_size = min(I_t.shape[0], 4)

    vis_predicted_img = get_grid_img(predicted_img[:batch_size])
    vis_predicted_img_gt = get_grid_img(I_t[:batch_size])
    # import pdb; pdb.set_trace()

    if save_batch:
        vis_predicted_img_np = np.clip(vis_predicted_img[0].detach().cpu().numpy().transpose((1,2,0)), 0.0, 1.0)
        vis_predicted_img_gt_np = np.clip(vis_predicted_img_gt[0].detach().cpu().numpy().transpose((1,2,0)), 0.0, 1.0)
        saving_folder = 'training_result'
        pred_fname, gt_fname = os.path.join(saving_folder, 'predict_{}.png'.format(datetime.datetime.now())), os.path.join(saving_folder,'gt_{}.png'.format(datetime.datetime.now()))
        plt.imsave(pred_fname, vis_predicted_img_np, cmap='gray')
        plt.imsave(gt_fname, vis_predicted_img_gt_np, cmap='gray')

    vis_shadow_img = torch.cat((vis_predicted_img_gt,
                                vis_predicted_img))
    if is_training:
        win_prefix = 'train'
    else:
        win_prefix = 'valid'
    vis_mask = get_grid_img(mask[:batch_size])
    visdom_show_batch(vis_mask, win_name="{} masks".format(win_prefix), exp=exp, normalize=False)
    visdom_show_batch(vis_shadow_img, win_name="{} shadow gt vs. inference".format(win_prefix), nrow=1,exp=exp, normalize=False)

    if not params.new_ibl:
        visdom_show_batch(get_grid_img(L_t[:batch_size]), win_name='{} light'.format(win_prefix), exp=exp, normalize=True)

def training_iteration(model, train_dataloder, optimizer, train_loss, epoch_num):
    # training
    cur_epoch_loss = 0.0
    model.train()
    
    with tqdm(total=len(train_dataloder) * params.timers) as t:
        t.set_description("Ep. {}".format(epoch_num))

        for j in range(params.timers):
            for i, (mask, light, shadow) in enumerate(train_dataloder):
                I_s, L_t, I_t = mask.to(device), light.to(device), shadow.to(device)

                optimizer.zero_grad()
                
                # predict
                predicted_img, predicted_src_light = model(I_s, L_t)

                # compute loss
                loss = reconstruct_loss(I_t, predicted_img)

                loss.backward()
                optimizer.step()
                
                cur_epoch_loss += loss.item()
                
                # visualize results
                if i % 10 == 0:
                    # divide_factor = 1.0 / torch.max(L_t) / 5.0
                    divide_factor = 1.0
                    visdom_plot_img(torch.clamp(I_t* divide_factor, 0.0, 1.0),
                                torch.clamp(predicted_img * divide_factor, 0.0, 1.0),
                                mask, L_t, save_batch=False)

                # keep tracking
                train_loss.append(loss.item()/np.sqrt(params.batch_size))
                visdom_plot_loss("train_total_loss", train_loss, exp)

                t.update()

    # Finish one epoch
    cur_epoch_loss /= (params.timers * len(train_dataloder) * np.sqrt(params.batch_size))
    return cur_epoch_loss

def validation_iteration(model, valid_dataloader, valid_loss, epoch_num):
    cur_epoch_loss = 0.0
    model.eval()

    cur_timer = 1
    with torch.no_grad():
        with tqdm(total=len(valid_dataloader) * params.timers) as t:
            t.set_description("(Validation)Ep. {} ".format(epoch_num))
            for j in range(cur_timer):
                for i, (mask, light, shadow) in enumerate(valid_dataloader):
                    I_s = mask.to(device)
                    L_t = light.to(device)
                    I_t = shadow.to(device)

                    # predict transfer
                    predicted_img, predicted_src_light = model(I_s, L_t)

                    # compute loss
                    loss = reconstruct_loss(I_t, predicted_img)

                    cur_epoch_loss += loss.item()

                    # visualize results
                    if i % 10 == 0:
                        # divide_factor = 1.0 / torch.max(L_t) / 5.0
                        divide_factor = 1.0
                        visdom_plot_img(torch.clamp(I_t * divide_factor, 0.0, 1.0),
                                        torch.clamp(predicted_img * divide_factor, 0.0, 1.0),
                                        mask, L_t, False)

                    # keep tracking
                    valid_loss.append(loss.item()/np.sqrt(params.batch_size))

                    visdom_plot_loss("valid_total_loss", valid_loss, exp)
                    t.update()

    # Finish one epoch
    cur_epoch_loss /= (params.timers * len(valid_dataloader) * np.sqrt(cur_timer)) 
    return cur_epoch_loss

def train(params):
    # history logs
    best_valid_loss = float('inf')
    log_info = ""
    hist_train_loss = []
    hist_valid_loss = []

    # dataset
    ds_csv = "/home/ysheng/Dataset/new_dataset/meta_data.csv"
    train_set = SSN_Dataset(ds_csv, True)
    train_dataloder = DataLoader(train_set, batch_size= min(len(train_set), params.batch_size), shuffle=True, num_workers=params.workers, drop_last=True)
    valid_set = SSN_Dataset(ds_csv, False)
    valid_dataloader = DataLoader(valid_set, batch_size= min(len(valid_set), params.batch_size), shuffle=False, num_workers=params.workers, drop_last=True)

    # model & optimizer & scheduler & loss function
    model = Relight_SSN(1, 1)    # input is mask + human
    model.to(device)    
    optimizer = set_model_optimizer(model, params.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=params.patience)
    
#     import pdb;pdb.set_trace()
    
    # resume from last saved points
    if params.resume:
        # print("Not implemented yet, remember to implement")
        # best_weight = "weights/cross entropy loss_04-December-07-56-PM.pt"
        best_weight = os.path.join("weights", params.weight_file)
        checkpoint = torch.load(best_weight, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint['best_loss']
        hist_train_loss = checkpoint['hist_train_loss']
        hist_valid_loss = checkpoint['hist_valid_loss']
        print("resuming from: {}".format(best_weight))
        del checkpoint
    
    if params.relearn:
        best_valid_loss = float('inf')
    
    print(torch.cuda.device_count())
    # test multiple GPUs
    if torch.cuda.device_count() > 1 and params.multi_gpu:
        print("Let's use ", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    set_lr(optimizer, params.lr)

    print("Current LR: {}".format(get_lr(optimizer)))

    # training states
    train_loss, valid_loss = [], []
    
    # training iterations
    for epoch in range(params.epochs):
        # training
        cur_train_loss = training_iteration(model, train_dataloder, optimizer, train_loss, epoch)

        # validation
        cur_valid_loss = validation_iteration(model, valid_dataloader, valid_loss, epoch)
        
        if params.use_schedule:
            scheduler.step(cur_valid_loss)

        log_info += "Current epoch: {} Learning Rate: {}  <br>".format(epoch, get_lr(optimizer))
        visdom_log(log_info, exp=exp)

        hist_train_loss.append(cur_train_loss)
        hist_valid_loss.append(cur_valid_loss)

        visdom_plot_loss("history train loss", hist_train_loss, exp=exp)
        visdom_plot_loss("history valid loss", hist_valid_loss, exp=exp)

        log_info += "Epoch: {} training loss: {}, valid loss: {}  <br>".format(epoch, cur_train_loss, cur_valid_loss)
        # save results
        if best_valid_loss > cur_valid_loss:
            log_info += "<br> ---------- Exp: {} Find better loss: {} at {} --------  <br>".format(exp_name, cur_valid_loss, datetime.datetime.now())
            visdom_log(log_info, exp=exp)

            best_valid_loss = cur_valid_loss
            global_params = options().get_params()
            save_model("weights", model, optimizer, epoch, best_valid_loss, exp_name, hist_train_loss, hist_valid_loss, global_params)

    print("Training finished")

if __name__ == "__main__":    
    # trainig
    train(params)