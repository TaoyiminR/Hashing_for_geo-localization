seed = 123
import numpy as np
from sympy import arg
np.random.seed(seed)
import random as rn
rn.seed(seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
# torch.cuda.set_device(6)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from utils.config import args
import time
from datetime import datetime

import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True

import nets as models
# from utils.preprocess import *
from utils.bar_show import progress_bar
import pdb
from src.cmdataset import CMDataset
import os
import scipy
import scipy.spatial
import scipy.io as sio
import torch.nn as nn
import src.utils as utils
from NCE.NCEAverage import NCEAverage
from NCE.NCECriterion import NCESoftmaxLoss
from torch.nn.utils.clip_grad import clip_grad_norm_
import torchsnooper
# --pretrain --arch resnet18

device_ids = [1, 2]
teacher_device_id = [1, 2]
best_acc = 0  # best test accuracy
start_epoch = 0
data_type = 'CVACT'

args.log_dir = os.path.join(args.root_dir, 'logs', args.log_name)
args.ckpt_dir = os.path.join(args.root_dir, 'ckpt', args.pretrain_dir)

args.data_name = 'cvact_fea'
args.bit = 512
# 平衡参数
args.alpha = 0.7
args.num_hiden_layers = [3, 2]
args.margin = 0.2
args.max_epochs = 10
args.train_batch_size = 32
args.shift = 0.01
args.optimizer = 'SGD'
# args.lr = 0.0001

# 从断点继续训练
# args.resume = 'vgg11_512_best_checkpoint.t7'

os.makedirs(args.log_dir, exist_ok=True)
os.makedirs(args.ckpt_dir, exist_ok=True)

def main():
    print('===> Preparing data ..')
        # build data
    train_dataset = CMDataset(
        args.data_name,
        return_index=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    retrieval_dataset = CMDataset(
        args.data_name,
        partition='retrieval'
    )
    retrieval_loader = torch.utils.data.DataLoader(
        dataset=retrieval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    test_dataset = CMDataset(
        args.data_name,
        partition='test'
    )
    query_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    print('===> Building Network..')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if 'fea' in args.data_name:
        image_model = models.__dict__['ImageNet'](y_dim=train_dataset.imgs.shape[1], bit=args.bit, hiden_layer=args.num_hiden_layers[0]).to(device)
        text_model = models.__dict__['ImageNet_grd'](y_dim=train_dataset.imgs.shape[1], bit=args.bit, hiden_layer=args.num_hiden_layers[0]).to(device)
        backbone = None

    else:
        # satview image net
        backbone = models.__dict__[args.arch](pretrained=args.pretrain, feature=True).to(device)
        fea_net = models.__dict__['ImageNet'](y_dim=768 if 'vgg' in args.arch.lower() else (512 if args.arch == 'resnet18' or args.arch == 'resnet34' else 2048), bit=args.bit, hiden_layer=args.num_hiden_layers[0]).to(device)
        image_model = nn.Sequential(backbone, fea_net)

        # grdview image net
        backbone = models.__dict__[args.arch](pretrained=args.pretrain, feature=True).to(device)
        fea_net = models.__dict__['ImageNet_grd'](y_dim=768 if 'vgg' in args.arch.lower() else (512 if args.arch == 'resnet18' or args.arch == 'resnet34' else 2048), bit=args.bit, hiden_layer=args.num_hiden_layers[0]).to(device)
        text_model = nn.Sequential(backbone, fea_net)

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    wd = args.wd
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=wd)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=wd)
    if args.ls == 'cos':
        lr_schedu = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=0, last_epoch=-1)
    else:
        lr_schedu = optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], gamma=0.1)

    summary_writer = SummaryWriter(args.log_dir)

    if args.resume:
        ckpt = torch.load(os.path.join(args.ckpt_dir, args.resume))
        image_model.load_state_dict(ckpt['image_model_state_dict'])
        text_model.load_state_dict(ckpt['text_model_state_dict'])
        ckpt['optimizer_state_dict']['param_groups'][0]['lr'] = args.lr
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        print('===> Load last checkpoint data')
    else:
        start_epoch = 0
        print('===> Start from scratch')

    def set_train(is_warmup=False):
        image_model.train()
        if is_warmup and backbone:
            backbone.eval()
            backbone.requires_grad_(False)
        elif backbone:
            backbone.requires_grad_(True)
        text_model.train()

    def set_eval():
        image_model.eval()
        text_model.eval()

    criterion = utils.ContrastiveLoss(args.margin, shift=args.shift)
    n_data = len(train_loader.dataset)
    contrast = NCEAverage(args.bit, n_data, args.K, args.T, args.momentum)
    contrast = contrast.to(device)
    criterion_contrast = NCESoftmaxLoss().to(device)
    criterion_contrast = criterion_contrast

    def train(epoch):
        print('\nEpoch: %d / %d' % (epoch, args.max_epochs))
        set_train(epoch < args.warmup_epoch)
        train_loss, correct, total = 0., 0., 0.
        for batch_idx, (idx, images, texts, _) in enumerate(train_loader):
            images, texts, idx = [img.to(device) for img in images], [txt.to(device) for txt in texts], [idx.to(device)]
            images_outputs = [image_model(im) for im in images]
            texts_outputs = [text_model(txt.float()) for txt in texts]

            out_l, out_ab = contrast(torch.cat(images_outputs), torch.cat(texts_outputs), torch.cat(idx * len(images)), idx=None, epoch=epoch-args.warmup_epoch)
            l_loss = criterion_contrast(out_l)
            ab_loss = criterion_contrast(out_ab)
            Lc = l_loss + ab_loss
            Lr = criterion(torch.cat(images_outputs), torch.cat(texts_outputs))
            loss = Lc * args.alpha + Lr * (1. - args.alpha)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(parameters, 1.)
            optimizer.step()
            train_loss += loss.item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | LR: %g'
                         % (train_loss / (batch_idx + 1), optimizer.param_groups[0]['lr']))

            if batch_idx % args.log_interval == 0:  #every log_interval mini_batches...
                summary_writer.add_scalar('Loss/train', train_loss / (batch_idx + 1), epoch * len(train_loader) + batch_idx)
                summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch * len(train_loader) + batch_idx)

    def eval(data_loader):
        imgs, txts, labs = [], [], []
        with torch.no_grad():
            for batch_idx, (images, texts, targets) in enumerate(data_loader):
                images, texts, targets = [img.to(device) for img in images], [txt.to(device) for txt in texts], targets.to(device)

                images_outputs = [image_model(im) for im in images]
                texts_outputs = [text_model(txt.float()) for txt in texts]

                imgs += images_outputs
                txts += texts_outputs
                labs.append(targets)

            imgs = torch.cat(imgs).sign_().cpu().numpy()
            txts = torch.cat(txts).sign_().cpu().numpy()
            labs = torch.cat(labs).cpu().numpy()

        save_data_path = './result' + '/' + data_type + '/'
        if not os.path.exists(save_data_path):
            os.makedirs(save_data_path)
        sio.savemat(save_data_path + data_type + '_hc.mat', {'sat_hc': imgs, 'grd_hc': txts})
        return imgs, txts, labs

    def test(epoch, is_eval=True):
        # pass
        global best_acc
        set_eval()
        # switch to evaluate mode
        (retrieval_imgs, retrieval_txts, retrieval_labs) = eval(retrieval_loader)
        # query_imgs:sat //query_txts:grd
        # retrival_imgs:sat // query_txts:grd
        if is_eval:
            query_imgs, query_txts, query_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
            retrieval_imgs, retrieval_txts, retrieval_labs = retrieval_imgs[0: 2000], retrieval_txts[0: 2000], retrieval_labs[0: 2000]
        else:
            (query_imgs, query_txts, query_labs) = eval(query_loader)
            retrieval_imgs, retrieval_txts = query_imgs, query_txts

        i2t = recall_evaluation(retrieval_txts, query_imgs, topK=1)
        t2i = recall_evaluation(retrieval_imgs, query_txts, topK=1)

        i2t_5 = recall_evaluation(retrieval_txts, query_imgs, topK=5)
        i2t_10 = recall_evaluation(retrieval_txts, query_imgs, topK=10)
        i2t_1p = recall_evaluation(retrieval_txts, query_imgs, topK=89)

        print('top1,5,10,1p', i2t, i2t_5, i2t_10, i2t_1p)

        avg = (i2t + t2i) / 2.
        print('%s\nImg2Txt: %g \t Txt2Img: %g \t Avg: %g' % ('Evaluation' if is_eval else 'Test', i2t, t2i, (i2t + t2i) / 2.))
        if avg > best_acc:
            print('Saving..')
            state = {
                'image_model_state_dict': image_model.state_dict(),
                'text_model_state_dict': text_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'Avg': avg,
                'Img2Txt': i2t,
                'Txt2Img': t2i,
                'epoch': epoch,
            }
            torch.save(state, os.path.join(args.ckpt_dir, '%s_%d_best_checkpoint.t7' % (args.arch, args.bit)))
            best_acc = avg
        return i2t, t2i

    lr_schedu.step(start_epoch)
    for epoch in range(start_epoch, args.max_epochs):
        train(epoch)
        lr_schedu.step(epoch)
        i2t, t2i = test(epoch)
        avg = (i2t + t2i) / 2.
        if avg == best_acc:
            image_model_state_dict = image_model.state_dict()
            image_model_state_dict = {key: image_model_state_dict[key].clone() for key in image_model_state_dict}
            text_model_state_dict = text_model.state_dict()
            text_model_state_dict = {key: text_model_state_dict[key].clone() for key in text_model_state_dict}
        test(epoch, is_eval=False)

    chp = torch.load(os.path.join(args.ckpt_dir, '%s_%d_best_checkpoint.t7' % (args.arch, args.bit)))
    image_model.load_state_dict(image_model_state_dict)
    text_model.load_state_dict(text_model_state_dict)
    test(chp['epoch'], is_eval=False)
    summary_writer.close()
    # pdb.set_trace()

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def calculate_hamming(B1, B2):
    dist_array = np.zeros([B1.shape[0], B2.shape[0]])
    iter = B1.shape[0]
    q = B2.shape[1]  # max inner product value
    for i in range(iter):
        distH = 0.5 * (q - np.dot(B1[i, :], B2.transpose()))
        dist_array[i, :] = distH
    return dist_array

def recall_evaluation(retrival, query, topK):
    dist = calculate_hamming(query, retrival)
    accu = 0.
    data_amount = 0.
    for i in range(dist.shape[0]):
        dist_gt = dist[i, i]
        pred = np.sum(dist[i, :] < dist_gt)
        if pred < topK:
            accu += 1.
        data_amount += 1.
    accu /= data_amount
    return accu

if __name__ == '__main__':
    main()

