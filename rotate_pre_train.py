import os
import sys
import shutil
import random
import math
import csv
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import roc_auc_score, roc_curve, auc



def data_iter(rand_shuf, img_dir, img_name, features, labels,
              images_per_gpu, rgb_mean, rgb_std, pad_val,
              image_shape):
    
    if rand_shuf:
        num_samples = len(img_name)
        indices = list(range(num_samples))
        random.shuffle(indices)
    else: # For validation
        num_samples = len(img_name)
        indices = list(range(num_samples))      

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])


    for i in range(0, num_samples, images_per_gpu):
        rotate_index_0 = torch.randint(0, len(rotate_list), (images_per_gpu,1))
        rotate_index_1 = torch.randint(0, len(rotate_list), (images_per_gpu,1))

        if i + images_per_gpu > num_samples: 
            batch_indices = indices[i:] + indices[: images_per_gpu - num_samples + i]
        else:
            batch_indices = indices[i: i + images_per_gpu]   
        imgs_hub_0 = []
        imgs_hub_1 = []
        for k in range(images_per_gpu):
            j = batch_indices[k]
            img_tongue = transforms(img_array[rotate_index_0[k]][img_name[j]])
            imgs_hub_0.append(img_tongue)
            img_tongue = transforms(img_array[rotate_index_1[k]][img_name[j]])
            imgs_hub_1.append(img_tongue)
        imgs_tensor_0 = torch.stack(imgs_hub_0, dim=0) # !!!!!!!!!!!
        imgs_tensor_1 = torch.stack(imgs_hub_1, dim=0)
        yield imgs_tensor_0, imgs_tensor_1, rotate_index_0 - rotate_index_1 + 0.0 # images, coordinates, features, labels

def train(net, train_img_name, train_features, train_labels, 
          valid_img_name, valid_features, valid_labels,
          num_epochs, learning_rate, weight_decay, batch_size, fold_i):
    save_file_name = work_dir + f'fold_{fold_i}_'

    best_epoch, save_i = -1, 0
    best_metric = [0, 0, 0, 0, 0]
    train_ls, valid_ls = [], []

    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.LambdaLR(optimizer, 
                                      lr_lambda=lambda epoch: adjust_learning_rate(epoch, warmup_factor, warmup_epochs))     

    for epoch in range(num_epochs):        
        print('Epoch: ',epoch, '  Batch Size = ', batch_size, f"  lr = {optimizer.param_groups[0]['lr']:.2e}")
        train_ls_batch, valid_ls_batch = [], []
        net.train()
        optimizer.zero_grad()
        i = 0
        for imgs_tensor_0, imgs_tensor_1, labels in data_iter(True, img_dir, train_img_name, train_features, train_labels,
                                                                images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
            X0, X1, y = imgs_tensor_0.to(device), imgs_tensor_1.to(device), labels.to(device)
            logit = net(X0, 0) - net(X1, 0)
            l = loss(logit, y).mean() # mean loss of images_per_gpu
            train_ls_batch.append(l.item())
            (l/mini_batch_num).backward() # add grad
            i += 1
            if i >= mini_batch_num:
                optimizer.step()
                optimizer.zero_grad()
                i = 0
                print(l.item(), end="\r")              
        if i:
            optimizer.zero_grad()
        scheduler.step()

        train_ls.append(sum(train_ls_batch)/len(train_ls_batch))
        print('train_loss = ', train_ls[-1])

        with torch.no_grad():
            net.eval()
            samples_num = len(valid_img_name)
            pred_hub = []
            label_hub = []
            for imgs_tensor_0, imgs_tensor_1, labels in data_iter(False, img_dir, valid_img_name, valid_features, valid_labels,
                                                                    images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
                X0, X1, y = imgs_tensor_0.to(device), imgs_tensor_1.to(device), labels.to(device)
                logit = net(X0, 0) - net(X1, 0)
                l = loss(logit, y).mean()
                samples_num -= images_per_gpu
                if samples_num < 0:
                    last_num = images_per_gpu + samples_num
                    logit = logit[:last_num]
                    y = y[:last_num]
                    l = loss(logit, y).mean()

                valid_ls_batch.append(l.item())
                pred_hub = pred_hub + logit.detach().to('cpu').tolist()
                label_hub = label_hub + y.detach().to('cpu').tolist()

        valid_ls.append(sum(valid_ls_batch)/len(valid_ls_batch))
        results_tensor = torch.tensor(pred_hub)
        labels_tensor = torch.tensor(label_hub)
        MSE = (results_tensor - labels_tensor).pow(2).mean(0).item()
        RMSE = (results_tensor - labels_tensor).pow(2).mean(0).pow(0.5).item()
        MAE = (results_tensor - labels_tensor).abs().mean(0).item()
        MAPE = ((results_tensor - labels_tensor).abs() / (labels_tensor.abs() + 1e-6)).mean(0).item()
        R2 = (1 - (results_tensor - labels_tensor).pow(2).sum(0) / ((labels_tensor - labels_tensor.mean(0)).pow(2).sum(0))).item()

        print('-----------------------------------------------------------------')
        print('MSE =  ', MSE)       
        print('RMSE = ', RMSE)       
        print('MAE =  ', MAE) 
        print('MAPE = ', MAPE)
        print('R2 =   ', R2)

        if valid_ls.index(min(valid_ls))==len(valid_ls)-1: # if now epoch is best according to valid_loss
            best_metric = [MSE, RMSE, MAE, MAPE, R2]
            net.to('cpu')
            best_epoch = epoch
            torch.save(net.state_dict(), pre_rorate_file_name) # save params
            print(f'Saved best params of epoch_{best_epoch}')
            net.to(device)
        print('Best epoch is: ', best_epoch)

        # save params
        save_i += 1
        if save_i == save_interval:
            net.to('cpu')
            torch.save(net.state_dict(), save_file_name+f'epoch_{epoch:03}.params') # save params
            print('Save parameters of Epoch: ', epoch)
            save_i = 0
            net.to(device)
        print('-------------------------------------')
    
    output_ls = [[x, y] for x, y in zip(train_ls, valid_ls)]
    with open(save_file_name + 'train_and_valid_lose.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows([['train_lose', 'valid_lose']] + output_ls)
    return best_metric

def get_k_fold_data(k, i, X, Xf, y):
    assert k > 1
    fold_size = len(X) // k
    X_train, Xf_train, y_train = None, None, None
    indices = list(range(len(X)))
    random.shuffle(indices) 
    for j in range(k):
        idx = indices[slice(j * fold_size, (j + 1) * fold_size)] # the indices for this fold
        X_part = []
        for index in idx:
             X_part.append(X[index])
        Xf_part = Xf[idx]
        y_part = y[idx]
        if j == i:
            X_valid, Xf_valid, y_valid = X_part, Xf_part, y_part
        elif X_train is None:
            X_train, Xf_train, y_train = X_part, Xf_part, y_part
        else:
            X_train = X_train + X_part
            Xf_train = torch.cat([Xf_train, Xf_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    
    return X_train, Xf_train, y_train, X_valid, Xf_valid, y_valid

def k_fold(k, X_train, Xf_train, y_train, num_epochs, 
           learning_rate, weight_decay, batch_size):
    metrics_hub = []
    i = 0
    data = get_k_fold_data(k, i, X_train, Xf_train, y_train)
    net = model.get_net(num_features, num_labels, drop_rate)
    metrics = train(net, *data, num_epochs, learning_rate,
                                weight_decay, batch_size, i)
    metrics_hub.append(metrics)
    print(f'Results for Flod {i}')
    print([['Precision', 'Recall', 'F1', 'Accuracy', 'Auc'], metrics])

    metrics_array = np.asarray(metrics_hub)
    mean_metrics = metrics_array.mean(axis=0).tolist()
    return mean_metrics


# Define the learning rate adjustment function
def adjust_learning_rate(epoch, warmup_factor, warmup_epochs):
    max_lr = 1.0
    min_lr = warmup_factor * max_lr
    if epoch < warmup_epochs:
        # Preheat stage: linearly increase the learning rate
        return min_lr + (max_lr - min_lr) * epoch / warmup_epochs
    else:
        # Trionometric function learning rate decay strategy
        t = epoch - warmup_epochs
        cycle_length = num_epochs
        return min_lr + (max_lr - min_lr) * (1 + math.cos(math.pi * t / cycle_length)) / 2

from models import Mffkan_RotatePre as model # 1

if __name__ == "__main__":

    data_path = './Tongue-FLD/Indicator_and_Label.csv'
    work_dir = './test_work_dir/'
    img_dir = './Tongue-FLD/Tongue_Images_rotated/'
    pre_rorate_file_name = './pre_rotate_params.params'

    num_images = 2000 # Reduce this value if your memory is limited
    lr = 0.001
    weight_decay = 0.0001
    batch_size = 64
    images_per_gpu = 2 # 
    drop_rate = 0.10 
    fold_num = 5
    num_epochs = 20
    reg_loss_rate_active = 0.1
    reg_loss_rate_entropy = 0.1
    image_shape = (448, 448) # (448, 448) # (224, 224)
    device = 'cuda'
    
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    warmup_epochs, warmup_factor = 5, 0.01
    save_interval = int(num_epochs/10) if int(num_epochs/10) else 1 # num_epochs/10

    rgb_mean = torch.tensor([123.675, 116.28, 103.53])/255 # COCO dataset
    rgb_std = torch.tensor([58.395, 57.12, 57.375])/255
    pad_val = [0, 0, 0]# images masked with dark

    train_data = pd.read_csv(data_path).iloc[:num_images]
    # get images' name
    rotate_list = os.listdir(img_dir)
    train_img_name = list(train_data.iloc[:,0])
    train_img_name = [[rotare + '/' + image for image in train_img_name] for rotare in rotate_list]
    # get features
    all_features = train_data.iloc[:, 1:-1]
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=False, dtype=int)
    all_features = torch.tensor(all_features.values, dtype=torch.float32)

    # get labels
    all_labels = train_data.iloc[:, -1]
    all_labels = pd.get_dummies(all_labels, dummy_na=False, dtype=int)
    all_labels = torch.tensor(all_labels.values, dtype=torch.float32)

    num_features = all_features.shape[1]
    num_labels = 1

    img_array = []
    for train_img_name_sub in train_img_name:
        img_array_sub = []
        for img_name in train_img_name_sub:
            img_array_sub.append(Image.open(img_dir + img_name).convert('RGB'))
        img_array.append(img_array_sub)
    train_img_name = list(range(len(train_img_name[0])))
                    
    mini_batch_num = batch_size/images_per_gpu
    if mini_batch_num!=round(mini_batch_num):
        mini_batch_num = int(mini_batch_num + 1)

    loss = nn.MSELoss(reduction='none')

    mean_metrics = k_fold(fold_num, train_img_name, all_features, all_labels,
                                num_epochs, lr, weight_decay, batch_size)
    
    print('Five flods mean results:')
    print([['MSE', 'RMSE', 'MAE', 'MAPE', 'R2'], mean_metrics])


