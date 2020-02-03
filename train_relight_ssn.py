import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
from tqdm import tqdm
import numpy as np
import os
import math

from ssn.ssn_dataset import SSN_Dataset
# from ssn.ssn_submodule import Contract
from ssn.ssn import Relight_SSN
from utils.net_utils import save_model, get_lr, set_lr
from utils.visdom_utils import visdom_plot_loss, visdom_relight_results, visdom_log, visdom_show_batch, visdom_show_light
from params import params as options, parse_params

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
    
def data_augmentation():
    """ Transforms passed to training and validating """
    train_trnfs = transforms.Compose([
        transforms.ToTensor()
    ])

    valid_trnfs = transforms.Compose([
        transforms.ToTensor()
    ])

    return train_trnfs, valid_trnfs


def get_spherical_weight():
    filename = "spherical_weight.pt"
    if os.path.exists(filename):
        weight = torch.load(filename)
    else:
        weight = torch.zeros(16, 32)
        # todo, vectorize this process
        for h in range(weight.size()[0]):
            weight[h, :] = abs(math.sin(h / 16.0 * 3.1415926)) + 0.001
        torch.save(weight, filename)

    return weight

def spherical_loss(gt_img, pred_img):
    """ latitude-longtitude loss """
    weight = get_spherical_weight()
    weight = weight.to(device)
    gt_img = torch.log(1.0 + gt_img)
    pred_img = torch.log(1.0 + pred_img)
    pred_img = pred_img.view(-1, 1, 16, 32)
    diff = torch.norm(weight * (gt_img - pred_img), 2)

    return diff * diff

def reconstruct_loss(gt_img, pred_img):
    """ M * (I-I') """
    return torch.norm((gt_img - pred_img), 2)

def loss_functions(ty, sl, sy, gt_ty, gt_sl, gt_sy):
    """ Loss Function:
          a. bottle-neck output light 
          b. reconstruct output original shadow
          c. reconstruct output new shadowsche
    """
    bn_light_loss = spherical_loss(gt_sl, sl)  # light loss
    # gt_sy_mask, gt_sy_shadow = decouple_image(gt_sy)
    # sy_mask, sy_shadow = decouple_image(sy)
    # gt_ty_mask, gt_ty_shadow = decouple_image(gt_ty)
    # ty_mask, ty_shadow = decouple_image(ty)

    # recon_ori_mask_loss = reconstruct_loss(gt_sy_mask, sy_mask)  # self supervision loss
    # recon_ori_shadow_loss = reconstruct_loss(gt_sy_shadow, sy_shadow)  # self supervision loss

    # recon_nov_mask_loss = reconstruct_loss(gt_ty_mask, ty_mask)  # new view loss
    # recon_nov_shadow_loss = reconstruct_loss(gt_ty_shadow, ty_shadow)  # self supervision loss

    recon_ori_loss = reconstruct_loss(gt_ty, ty)
    recon_nov_loss = reconstruct_loss(gt_sy, sy)
    # return bn_light_loss, recon_ori_mask_loss + recon_ori_shadow_loss, recon_nov_mask_loss + recon_nov_shadow_loss
    return bn_light_loss, recon_nov_loss, recon_ori_loss

def training_iteration(model, train_dataloder, optimizer, train_loss, epoch_num):
    # training
    cur_epoch_loss = 0.0
    model.train()
    
    with tqdm(total=len(train_dataloder) * params.timers) as t:
        t.set_description("Ep. {}".format(epoch_num))

        vis_predicted_img = torch.zeros(1)
        for j in range(params.timers):
            for i, (mask, light, shadow) in enumerate(train_dataloder):
        
                # concatenate human mask and shadow mask
                # I_s = torch.cat((mask, shadow), dim=1).to(device)
                # I_t = torch.cat((nov_mask, nov_shadow), dim=1).to(device)
                # import pdb; pdb.set_trace()
                
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
                    vis_predicted_img = torch.clamp(predicted_img, 0.0, 1.0).clone().cpu()
                    vis_predicted_img_gt = I_t.detach().cpu()

                    batch_size, c, h, w = vis_predicted_img.size()

                    random_batch = min(4, batch_size)
                    vis_shadow_img = torch.cat((vis_predicted_img_gt[0:random_batch, :, :, :].view(random_batch, 1, h, w), vis_predicted_img[0:random_batch, :, :, :].view(random_batch,1,h, w)))

                    torchvision.utils.save_image(predicted_img[0:random_batch, 0, :, :].view(random_batch, 1,h, w), "{}_shadow.png".format(exp_name), nrow=4)
                    visdom_show_batch(mask[:random_batch,:,:,:], win_name="train masks", exp=exp)
                    visdom_show_batch(vis_shadow_img, win_name="train shadow gt vs. inference", exp=exp)
                    
                # keep tracking
                train_loss.append(loss.item() / params.batch_size)

                visdom_plot_loss("train_total_loss", train_loss, exp)

                t.update()

    # Finish one epoch
    cur_epoch_loss /= (params.timers * len(train_dataloder) * params.batch_size)
    return cur_epoch_loss

def validation_iteration(model, valid_dataloader, valid_loss, epoch_num):
    cur_epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        with tqdm(total=len(valid_dataloader) * params.timers) as t:
            t.set_description("(Validation)Ep. {} ".format(epoch_num))
            vis_predicted_img = torch.zeros(1)

            for j in range(params.timers):
                for i, (mask, light, shadow) in enumerate(valid_dataloader):

                    # concatenate human mask and shadow mask
#                     I_s = torch.cat((mask, shadow), dim=1).to(device)
#                     L_t = nov_light.to(device)
#                     I_t = torch.cat((nov_mask, nov_shadow), dim=1).to(device)
#                     L_s = light.to(device)

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
                        vis_predicted_img = torch.clamp(predicted_img, 0.0, 1.0).clone().cpu()
                        vis_predicted_img_gt = I_t.detach().cpu()

                        batch_size, c, h, w = vis_predicted_img.size()
                        random_batch = min(4, batch_size)
                        vis_shadow_img = torch.cat((vis_predicted_img_gt[0:random_batch, :, :, :].view(random_batch, 1,
                                                                                                       h, w),
                                                    vis_predicted_img[0:random_batch, :, :, :].view(random_batch, 1, h,
                                                                                                    w)))
                        torchvision.utils.save_image(predicted_img[0:random_batch, :, :, :].view(random_batch, 1, h, w),
                                                     "valid_{}_shadow.png".format(exp_name), nrow=4,
                                                     normalize=True)
                        
                        visdom_show_batch(mask[:random_batch,:,:,:], win_name="valid masks", exp=exp)
                        visdom_show_batch(vis_shadow_img, win_name="valid shadow gt vs. inference", exp=exp)
                        
                    # keep tracking
                    valid_loss.append(loss.item() / params.batch_size)

                    visdom_plot_loss("valid_total_loss", valid_loss, exp)
                    t.update()

    # Finish one epoch
    cur_epoch_loss /= (params.timers * len(valid_dataloader) * params.batch_size) 
    return cur_epoch_loss

def train(params):
    # history logs
    best_valid_loss = float('inf')
    log_info = ""
    hist_train_loss = []
    hist_valid_loss = []

    # transforms
    # train_trnfs, valid_trnfs = data_augmentation()

    # data set
    ds_csv = "~/Dataset/soft_shadow/train/metadata.csv"
    train_set = SSN_Dataset(ds_csv, True)
    train_dataloder = DataLoader(train_set, batch_size=params.batch_size, shuffle=True, num_workers=params.workers, drop_last=True)
    valid_set = SSN_Dataset(ds_csv, False)
    valid_dataloader = DataLoader(valid_set, batch_size=params.batch_size, shuffle=False, num_workers=params.workers, drop_last=True)

    # model & optimizer & scheduler & loss function
    model = Relight_SSN(1, 1)    # input is mask + human
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(params.beta1, 0.999), eps=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    # resume from last saved points
    if params.resume:
        # print("Not implemented yet, remember to implement")
        # best_weight = "weights/cross entropy loss_04-December-07-56-PM.pt"
        best_weight = os.path.join("weights", params.weight_file)
        checkpoint = torch.load(best_weight)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_valid_loss = checkpoint['best_loss']
        hist_train_loss = checkpoint['hist_train_loss']
        hist_valid_loss = checkpoint['hist_valid_loss']
        print("resuming from: {}".format(best_weight))

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
            import datetime

            log_info += "<br> ---------- Exp: {} Find better loss: {} at {} --------  <br>".format(exp_name, cur_valid_loss, datetime.datetime.now())

            visdom_log(log_info, exp=exp)

            best_valid_loss = cur_valid_loss
            save_model("weights", model, optimizer, epoch, best_valid_loss, exp_name, hist_train_loss, hist_valid_loss, params.multi_gpu)


    print("Training finished")

if __name__ == "__main__":    
    # trainig
    train(params)