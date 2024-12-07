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
        # Progressive sampling
        num_classes = labels.shape[1]
        num_unique = labels.shape[0]
        index_hub_class = []
        for i in range(num_classes): 
            index_hub_class.append(torch.where(labels[:, i])[0].tolist())
        [random.shuffle(indices_class) for indices_class in index_hub_class]
        num_minor_class = min([len(indices_class) for indices_class in index_hub_class])
        class_sampling = []
        for i in range(num_classes):
            class_sampling.append(index_hub_class[i][:num_minor_class])
        class_sampling_indices = np.asarray(class_sampling).transpose().flatten().tolist()
        random_sampling_indices = list(range(num_unique))
        random.shuffle(random_sampling_indices)
        random_sampling_indices = random_sampling_indices[:num_classes*num_minor_class]
        # Progressive
        indices = np.asarray([class_sampling_indices, random_sampling_indices]).transpose().flatten().tolist()
        num_samples = len(indices)      
    else: # For validation
        num_samples = len(img_name)
        indices = list(range(num_samples))        

    # Define the transformation pipeline
    if rand_shuf:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(4, fill=pad_val),
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    else:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_shape),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])

    # Iterate over the dataset in batches
    for i in range(0, num_samples, images_per_gpu):
        if i + images_per_gpu > num_samples:
            batch_indices = indices[i:] + indices[: images_per_gpu - num_samples + i]
        else:
            batch_indices = indices[i: i + images_per_gpu]
        imgs_hub = []
        for j in batch_indices:
            img_tongue = transforms(Image.open(img_dir+img_name[j]).convert('RGB'))
            imgs_hub.append(img_tongue)
        imgs_tensor = torch.stack(imgs_hub, dim=0)
        yield imgs_tensor, features[batch_indices], labels[batch_indices], batch_indices


def train(net, train_img_name, train_indicator, train_labels, 
          valid_img_name, valid_indicator, valid_labels,
          num_epochs, learning_rate, weight_decay, batch_size, fold_i):
    save_file_name = work_dir + f'fold_{fold_i}_'

    best_epoch, save_i = -1, 0
    best_metric = [0, 0, 0, 0, 0, 0, 0]
    train_ls, valid_ls = [], []

    net.to(device)

    opt_parameters = [] # Collect parameters to optimize
    for param in net.IE.base.layerKAN.parameters():
        opt_parameters.append(param)
    for param in net.DE.parameters():
        opt_parameters.append(param)
    for param in net.FFC.parameters():
        opt_parameters.append(param)
    # Freeze layers
    for param in net.MEC.parameters():
        param.requires_grad = False
    for param in net.IE.base.layer1.parameters():
        param.requires_grad = False
    for param in net.IE.base.layer2.parameters():
        param.requires_grad = False
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(opt_parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.LambdaLR(optimizer, 
                                      lr_lambda=lambda epoch: adjust_learning_rate(epoch, warmup_factor, warmup_epochs))     

    net.train()
    for epoch in range(num_epochs):        
        print('Epoch: ',epoch, '  Batch Size = ', batch_size, f"  lr = {optimizer.param_groups[0]['lr']:.2e}")
        train_ls_batch = []
        optimizer.zero_grad()
        i = 0
        for img_tensor, feature_tensor, labels, _ in data_iter(True, img_dir, train_img_name, train_indicator, train_labels,
                                                                images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
            X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels.to(device)
            logits, _, _, _ = net(X, Xf)

            cls_loss = moi_loss(logits, y)
            reg_loss = net.regularization_loss(reg_loss_rate_active, reg_loss_rate_entropy)
            l = cls_loss + reg_loss
            train_ls_batch.append(l.item())
            (l/mini_batch_num).backward() # Accumulate gradients
            # print(l.item())
            i += 1
            if i >= mini_batch_num:
                optimizer.step()
                optimizer.zero_grad()
                i = 0
                print(l.item(), end="\r")              
        if i:
            optimizer.step()
        scheduler.step() # adjust learning rate
        print('train_loss = ', sum(train_ls_batch)/len(train_ls_batch))


    net.eval()
    logit_hub = []
    index_hub = []
    label_hub = []
    with torch.no_grad():
        for img_tensor, feature_tensor, labels, batch_index in data_iter(False, img_dir, train_img_name, train_indicator, train_labels,
                                                                            images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
            X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels
            logits, _, _, _ = net(X, Xf)
            logit_hub.append(logits.detach().to('cpu'))
            index_hub.extend(batch_index) 
            label_hub.append(labels) 

    label_tensor = torch.cat(label_hub, dim=0)
    label_tensor = label_tensor.argmax(dim=1)
    # Group indices by class
    classes_index = [torch.where(label_tensor==i)[0] for i in range(num_labels)]
    logit_tensor = torch.cat(logit_hub, dim=0)
    # Calculate uncertainty
    uncertainty_tensor = moi_uncertianty(logit_tensor.to(device)).detach().to('cpu')
    classes_uncertainty = [uncertainty_tensor[classes_index[i]] for i in range(num_labels)]
    # Determine the mim number of hard examples to select for each class
    min_class_num = min([classes_index[i].shape[0] for i in range(num_labels)])
    classes_hard_num = [max(min_class_num, int(classes_index[i].shape[0] * min_hard_rate)) for i in range(num_labels)]

    # Select hard examples
    classes_index_hub = [[index_hub[j] for j in classes_index[i]] for i in range(num_labels)]
    hard_index = []
    for i in range(num_labels):
        now_index_hub = classes_index_hub[i]
        hard_random_index = torch.where(classes_uncertainty[i] > uncertainty_threshold)[0]
        if hard_random_index.size(0) < classes_hard_num[i]:
            _, hard_random_index = torch.topk(classes_uncertainty[i], k=classes_hard_num[i])
        hard_index.extend([now_index_hub[j] for j in hard_random_index])
    hard_train_img_name = [train_img_name[i] for i in hard_index]
    hard_train_features = train_indicator[hard_index]
    hard_train_labels = train_labels[hard_index]


    for param in net.MEC.parameters():
        param.requires_grad = True
    for param in net.IE.parameters():
        param.requires_grad = False
    for param in net.DE.parameters():
        param.requires_grad = False
    for param in net.FFC.parameters():
        param.requires_grad = False

    hard_opt_parameters = [] # Collect parameters to optimize
    for param in net.MEC.parameters():
        hard_opt_parameters.append(param)
    
    net.to('cpu')
    torch.cuda.empty_cache()
    net.to(device)

    hard_optimizer = torch.optim.AdamW(hard_opt_parameters, lr=learning_rate, weight_decay=weight_decay)
    # Create a learning rate scheduler object
    hard_scheduler = lr_scheduler.LambdaLR(hard_optimizer, 
                                      lr_lambda=lambda epoch: adjust_learning_rate(epoch, warmup_factor, warmup_epochs))   


    for epoch in range(num_epochs):        
        print('Epoch: ',epoch, '  Batch Size = ', batch_size, f"  lr = {hard_optimizer.param_groups[0]['lr']:.2e}")
        train_ls_batch, valid_ls_batch = [], []
        net.train()

        hard_optimizer.zero_grad()
        i = 0
        for img_tensor, feature_tensor, labels, _ in data_iter(True, img_dir, hard_train_img_name, hard_train_features, hard_train_labels,
                                                                images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
            X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels.to(device)
            _, _, distance, _ = net(X, Xf)

            cls_loss = (distance * y).mean()
            reg_loss = net.regularization_loss(reg_loss_rate_active, reg_loss_rate_entropy)
            l = cls_loss + reg_loss
            train_ls_batch.append(cls_loss.item())
            (l/mini_batch_num).backward() # add grad
            # print(l.item())
            i += 1
            if i >= mini_batch_num:
                hard_optimizer.step()
                hard_optimizer.zero_grad()
                i = 0
                print(l.item(), end="\r")              
        if i:
            hard_optimizer.step()
        hard_scheduler.step() # adjust learning rate
        train_ls.append(sum(train_ls_batch)/len(train_ls_batch))
        print('train_loss = ', train_ls[-1])

        with torch.no_grad():
            net.eval()
            label_hub, yhat_hub = [], []
            hard_num_total = 0
            samples_num = len(valid_img_name)
            val_result = torch.zeros((num_labels, 3), dtype=int) # [TP, FN, FP]
            val_matrix = torch.zeros((num_labels, num_labels), dtype=int)
            for img_tensor, feature_tensor, labels, _ in data_iter(False, img_dir, valid_img_name, valid_indicator, valid_labels,
                                                                    images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
                X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels
                FCC_out, MEC_out, distance, _ = net(X, Xf)
                cls_loss = (distance * y.to(device)).mean() # MEC loss
                reg_loss = net.regularization_loss(reg_loss_rate_active, reg_loss_rate_entropy)
                l = cls_loss + reg_loss      
                valid_ls_batch.append(cls_loss.item())       

            valid_ls.append(sum(valid_ls_batch)/len(valid_ls_batch))
            print('valid_loss = ', valid_ls[-1])

        if valid_ls.index(min(valid_ls))==len(valid_ls)-1: # if now epoch is best according to valid_loss
            net.to('cpu')
            if best_epoch >= 0:
                os.remove(save_file_name+f'best_epoch_{best_epoch:03}.params') # delete old
            best_epoch = epoch
            best_params_name = save_file_name+f'best_epoch_{best_epoch:03}.params'
            torch.save(net.state_dict(), best_params_name) # save params
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
    
    net.to('cpu')
    torch.cuda.empty_cache()
    best_params = torch.load(best_params_name)
    net.load_state_dict(best_params, strict=True)
    net.to(device)
    with torch.no_grad():
        net.eval()
        label_hub, yhat_hub = [], []
        hard_num_total = 0
        samples_num = len(valid_img_name)
        val_result = torch.zeros((num_labels, 3), dtype=int) # [TP, FN, FP]
        val_matrix = torch.zeros((num_labels, num_labels), dtype=int)
        for img_tensor, feature_tensor, labels, _ in data_iter(False, img_dir, valid_img_name, valid_indicator, valid_labels,
                                                                images_per_gpu, rgb_mean, rgb_std, pad_val, image_shape): 
            X, Xf, y = img_tensor.to(device), feature_tensor.to(device), labels
            FCC_out, MEC_out, distance, _ = net(X, Xf)

            uncertainty_tensor = moi_uncertianty(FCC_out)
            hard_index_val = torch.where(uncertainty_tensor > uncertainty_threshold)[0]
            hard_num = hard_index_val.size(0)
            if hard_num:
                hard_num_total += hard_num
                FCC_out[hard_index_val] = MEC_out[hard_index_val]

            y_hat = FCC_out.detach().to('cpu')
            samples_num -= images_per_gpu
            if samples_num < 0:
                last_num = images_per_gpu + samples_num
                y_hat = y_hat[:last_num]
                y = y[:last_num]

            label_hub = label_hub + y.argmax(dim=1).tolist()
            yhat_hub = yhat_hub + y_hat.tolist()
            pred = y_hat.squeeze(1).argmax(dim=1)
            for i in range(pred.shape[0]):
                yi = y[i].argmax() # label index
                pi = pred[i] # prediction index
                val_matrix[pi, yi] += 1 
                if y[i, pi]: # TP
                    val_result[pi][0] = val_result[pi][0] + 1
                else:
                    val_result[yi][1] = val_result[yi][1] + 1
                    val_result[pi][2] = val_result[pi][2] + 1
    
        pred_array = np.asarray(yhat_hub)
        pred_exp = np.exp(pred_array - np.max(pred_array, axis=1, keepdims=True))
        pred_softmax = pred_exp / pred_exp.sum(axis=1, keepdims=True)
        if num_labels == 2:
            Auc = roc_auc_score(label_hub, pred_softmax[:, 1])
        else:
            Auc = roc_auc_score(label_hub, pred_softmax, multi_class='ovr')
        Accuracy = (val_result[:, 0].sum() / len(yhat_hub)).item()
        Precision = (val_result[:,0] / (val_result[:,0]+val_result[:,2] + 1e-6))
        Recall = val_result[:,0]/(val_result[:,0]+val_result[:,1] + 1e-6)
        F1 = (2 * (Precision * Recall) / (Precision + Recall)).mean().item()
        Precision = Precision.mean().item()
        Recall = Recall.mean().item()
        MAE = np.abs((pred_array.argmax(1) - np.asarray(label_hub))).mean()
        RMSE = np.sqrt(((pred_array.argmax(1) - np.asarray(label_hub)) ** 2).mean())
        print('Accuracy   = ', Accuracy)        
        print('Precision  = ', Precision)                
        print('Recall     = ', Recall)
        print('F1-score   = ', F1)
        print('AUC        = ', Auc)    
        print('MAE        = ', MAE)
        print('RMSE       = ', RMSE)
        print('Confusion Matrix: ')
        print(val_matrix)
        best_metric = [Accuracy, Precision, Recall, F1, Auc, MAE, RMSE]

    output_ls = [[x, y] for x, y in zip(train_ls, valid_ls)]
    with open(save_file_name + 'train_and_valid_lose.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows([['train_lose', 'valid_lose']] + output_ls)
    return best_metric

def get_k_fold_data(k, i, indices, X_img, X_ind, Y_lable): # Get training and validation data for k-fold cross-validation.
    assert k > 1
    fold_size = len(X_img) // k
    X_img_train, X_ind_train, y_train = None, None, None
    for j in range(k):
        idx = indices[slice(j * fold_size, (j + 1) * fold_size)] # the indices for this fold
        X_part = []
        for index in idx:
             X_part.append(X_img[index])
        Xf_part = X_ind[idx]
        y_part = Y_lable[idx]
        if j == i:
            X_img_valid, X_ind_valid, y_valid = X_part, Xf_part, y_part
        elif X_img_train is None:
            X_img_train, X_ind_train, y_train = X_part, Xf_part, y_part
        else:
            X_img_train = X_img_train + X_part
            X_ind_train = torch.cat([X_ind_train, Xf_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    
    return X_img_train, X_ind_train, y_train, X_img_valid, X_ind_valid, y_valid

def k_fold(k, X_img, X_ind, Y_label, num_epochs, 
           learning_rate, weight_decay, batch_size):
    indices = list(range(len(X_img)))
    random.shuffle(indices) 
    metrics_hub = []
    for i in range(k):
        data = get_k_fold_data(k, i, indices, X_img, X_ind, Y_label)
        net = model.get_net(num_features, num_labels, drop_rate)
        # load rotation prediction trained params
        rotate_pretrain_params = torch.load(pre_rorate_file)
        net.load_state_dict(rotate_pretrain_params, strict=False)

        metrics = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size, i)
        metrics_hub.append(metrics)
        print(f'Results for Flod {i}')
        print([['Accuracy', 'Precision', 'Recall', 'F1', 'Auc', 'MAE', 'RMSE'], metrics])

    metrics_array = np.asarray(metrics_hub)
    mean_metrics = metrics_array.mean(axis=0).tolist()
    return mean_metrics

def moi_loss(logit, y): # Calculate and return the loss value after applying the MOI loss function.
    probability = torch.softmax(logit, dim=1)
    y_center = (y * class_index).sum(dim=1, keepdim=True)
    rad = class_index - y_center

    cross_entropy = - (y * torch.log(probability)).sum(1).mean()
    moi = (rad * rad * probability).sum(1).mean()
    return alpha * moi + belta * cross_entropy

def moi_uncertianty(logit): # Calculate and return the uncertainty value after applying the MOI uncertainty function.
    probability = torch.softmax(logit, dim=1)
    center = (probability * class_index).sum(dim=1, keepdim=True)
    rad = class_index - center
    moi = (rad * rad * probability).sum(1)
    return moi


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

from models import Mffkan as model # 1

if __name__ == "__main__":

    data_path = './Tongue-FLD/Indicator_and_Label.csv'
    work_dir = './test_work_dir/'
    img_dir = './Tongue-FLD/Tongue_Images/'
    pre_rorate_file = './pre_rotate_params.params'
    device = 'cuda'

    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 60
    images_per_gpu = 6 # 12 
    drop_rate = 0.1
    fold_num = 5
    num_epochs = 50

    # # Define loss weights
    alpha = 0.5 # 0.75 # 0.25
    belta = 1.0 - alpha
    class_distance = 0.5
    reg_loss_rate_active = 0.1
    reg_loss_rate_entropy = 0.1

    # hard threshold
    min_hard_rate = 0.5
    uncertainty_rate = 0.5
 
    warmup_epochs, warmup_factor = 5, 0.01
    save_interval = int(num_epochs/10) if int(num_epochs/10) else 1 # num_epochs/10

    image_shape = (448, 448) # (224, 224)
    rgb_mean = torch.tensor([123.675, 116.28, 103.53])/255 # COCO dataset
    rgb_std = torch.tensor([58.395, 57.12, 57.375])/255
    pad_val = [0, 0, 0] # images masked with dark

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    # load data
    all_data = pd.read_csv(data_path)
    # get images' name
    all_img_name = list(all_data.iloc[:,0])
    # get features
    all_features = all_data.iloc[:, 1:-1]
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    # normalize numeric features
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=False, dtype=int)
    all_features = torch.tensor(all_features.values, dtype=torch.float32)
    # get labels
    all_labels = all_data.iloc[:, -1]
    all_labels = pd.get_dummies(all_labels, dummy_na=False, dtype=int)
    all_labels = torch.tensor(all_labels.values, dtype=torch.float32)
    # get the number of features and labels
    num_features = all_features.shape[1]
    num_labels = all_labels.shape[1]

    # calculate the number of mini-batches based on batch size and images per GPU
    mini_batch_num = batch_size/images_per_gpu
    if mini_batch_num!=round(mini_batch_num):
        mini_batch_num = int(mini_batch_num + 1)

    # prepare index tensor for labels
    class_index = torch.tensor([class_distance * i for i in range(num_labels)]).unsqueeze(0).to(device)
    # calculate uncertainty threshold based on uncertainty rate and model uncertainty
    uncertainty_threshold = uncertainty_rate * moi_uncertianty((torch.ones((1, num_labels))/num_labels).to(device)) # math.log(num_labels)
    uncertainty_threshold = uncertainty_threshold.item()
    # perform k-fold cross-validation and get mean metrics
    mean_metrics = k_fold(fold_num, all_img_name, all_features, all_labels,
                                num_epochs, learning_rate, weight_decay, batch_size)
    
    print('Five flods mean results:')
    print([['Accuracy', 'Precision', 'Recall', 'F1', 'Auc', 'MAE', 'RMSE'], mean_metrics])


