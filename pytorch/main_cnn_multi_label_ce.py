import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets
import torchvision.transforms as transforms

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *

import pretrainedmodels

import dataset_processing
#from gen_labels import get_labels

from resnet_224 import *
#import resnext_101_64x4d 

from mean_teacher import data

LOG = logging.getLogger('main')

args = None
best_prec1 = 0
global_step = 0

DATA_PATH = '/misc/lmbraid19/mittal/dense_prediction/forked/mean-teacher/pytorch/dataset/VOC_2012_class/'
TRAIN_DATA = 'train_10582/train_all'
TEST_DATA = 'val'
TRAIN_IMG_FILE = 'train_img.txt'
TEST_IMG_FILE = 'val_img.txt'
TRAIN_LABEL_FILE = 'train_label.txt'
TEST_LABEL_FILE = 'val_label.txt'

#train_labels = get_labels('train')
#test_labels = get_labels('test')

def main(context):
    global global_step
    global best_prec1

    checkpoint_path = context.transient_dir
    training_log = context.create_train_log("training")
    validation_log = context.create_train_log("validation")
    ema_validation_log = context.create_train_log("ema_validation")

    #dataset_config = datasets.__dict__[args.dataset]()
    #num_classes = dataset_config.pop('num_classes')
    #train_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
    train_loader, eval_loader, train_eval_loader = create_data_loaders()
    
    def create_model(ema=False):
        LOG.info("=> creating {pretrained}{ema}model '{arch}'".format(
            pretrained='pre-trained ' if args.pretrained else '',
            ema='EMA ' if ema else '',
            arch=args.arch))
       
        ''' 
        model = resnext_101_64x4d.resnext_101_64x4d
        model.load_state_dict(torch.load('resnext_101_64x4d.pth'))
        model.eval() 
        #model = torch.load('resnext_101_64x4d.pth')
        #model.load_state_dict(checkpoint['state_dict'])   
        #model.cuda()
        #model = torch.load
        #torch.load('resnext_101_64x4d.pth')['state_dict']       
        #model = torch.nn.DataParallel(model).cuda()
        model.cuda()

        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        model = model_factory(**model_params)
        model = nn.DataParallel(model).cuda()
        #model.cuda()
        
        #model_name = 'resnext50_32x4d'
        #model_name = 'resnext101_64x4d'
        ''' 
        model = resnet101(pretrained=True)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        model.module.fc = nn.Sequential(*list(model.module.fc.children())[:-1], nn.Linear(2048, 21))
        #self.dis.module.fc = nn.Linear(4096, 7)
        model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    LOG.info(parameters_string(model))

   
    ''' 
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    '''
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
     
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), "=> no checkpoint found at '{}'".format(args.resume)
        LOG.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        LOG.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    cudnn.benchmark = True
    '''
    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        #print ("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model:")
        #print ("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return
    '''
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, epoch, training_log)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            #print ("Evaluating the primary model:")
            prec1 = validate(eval_loader, 'val', model, validation_log, global_step, epoch + 1)
            LOG.info("Evaluating the EMA model:")
            #print ("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, 'ema', ema_model, ema_validation_log, global_step, epoch + 1)
            LOG.info("Evaluating the primary train model:")
            #print ("Evaluating the primary model:")
            #prec_train = validate(train_eval_loader, 'trainval', model, validation_log, global_step, epoch + 1)
            
    
            LOG.info("--- validation in %s seconds ---" % (time.time() - start_time))
            is_best = ema_prec1 > best_prec1
            best_prec1 = max(ema_prec1, best_prec1)
        else:
            is_best = False

        if args.checkpoint_epochs and (epoch + 1) % args.checkpoint_epochs == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'global_step': global_step,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, checkpoint_path, epoch + 1)


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders():
    channel_stats = dict(mean=[.485, .456, .406],
                         std=[.229, .224, .225])

    #transform_train = transforms.Compose([
    transform_train = data.TransformTwice(transforms.Compose([
        #transforms.Resize(224),
        #data.RandomTranslateWithReflect(4),
        #transforms.RandomRotation(10),
        transforms.Resize(size=(224, 224), interpolation=2),
        #transforms.RandomResizedCrop(224, scale =(0.8, 1.2)),
        transforms.RandomCrop(224, padding=8),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        #transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        #transforms.Resize(224),
        transforms.Normalize(**channel_stats)
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))

    '''
         transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    '''
        
    transform_test = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
        

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])
   
    ''' 
    dataset = dataset_processing.DatasetProcessing(
        DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, transform_train)

    #dataset_test = dataset_processing.DatasetProcessing(
        #DATA_PATH, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, transform_test)
    
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size= args.batch_size,
                                               num_workers=args.workers,
                                               pin_memory=True)
    
    if args.labels:
        labeled_idxs, unlabeled_idxs = data.relabel_dataset_ml(dataset, args.percent)
        #print (len(labeled_idxs))
        #print (len(unlabeled_idxs))

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        print ('two stream samples')
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)
     
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)
     
    eval_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)
    
        
    '''
    
    datadir = '/misc/lmbraid19/mittal/dense_prediction/forked/mean-teacher/pytorch/dataset/VOC_2012_class/'
    traindir = os.path.join(datadir, args.train_subdir)
    evaldir = os.path.join(datadir, args.eval_subdir)

    assert_exactly_one([args.exclude_unlabeled, args.labeled_batch_size])

    dataset = torchvision.datasets.ImageFolder(traindir, transform_train)
    dataset, labels = data.change_labels(dataset, DATA_PATH, TRAIN_IMG_FILE)
     
    #if args.labels:
    labeled_idxs, unlabeled_idxs, dataset = data.relabel_dataset_ml(dataset, labels, args.percent)

    print (len(labeled_idxs))
    print (len(unlabeled_idxs))

    if args.exclude_unlabeled:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    elif args.labeled_batch_size:
        batch_sampler = data.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, args.batch_size, args.labeled_batch_size)
    else:
        assert False, "labeled batch size {}".format(args.labeled_batch_size)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_sampler=batch_sampler,
                                               num_workers=args.workers,
                                               pin_memory=True)
    

    dataset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, transform_test)
    
    eval_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)
    
    dataset = dataset_processing.DatasetProcessing(
        DATA_PATH, TRAIN_DATA, TRAIN_IMG_FILE, TRAIN_LABEL_FILE, transform_test)
    
    train_eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)
    
    return train_loader, eval_loader, train_eval_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def weighted_binary_cross_entropy(output, target):

    # including Augmented dataset total 10582 training images
    #norm_weights = [0.04810572, 0.05631423, 0.04025868, 0.0606461,  0.03975122, 0.07221978, 0.02468032, 0.02824117, 0.02311268, 0.10630102, 0.04630077, 0.02389089, 0.06378061, 0.05768775, 0.0068309, 0.05437236, 0.09460791, 0.04373247, 0.05642619, 0.05005709, 0.0] # based on frequency
    
    #norm_weights = [0.04810572, 0.05631423, 0.04025868, 0.0606461,  0.04, 0.07221978, 0.04, 0.04, 0.04, 0.07, 0.04630077, 0.04, 0.06378061, 0.05768775, 0.04, 0.05437236, 0.07, 0.04373247, 0.05642619, 0.05005709, 0.0] # min value 0.02
    #norm_weights = [0.05062529, 0.06853885, 0.04242881, 0.05711571, 0.05120719, 0.05711571, 0.04480489, 0.04400783, 0.04010152, 0.06960977, 0.05432958, 0.04, 0.06551508, 0.05500031, 0.04, 0.05432958, 0.07071469, 0.0479035, 0.053675, 0.05303602, 0.0] # from pytorch multi label main.py

    #norm_weights = [0.05251095, 0.05438634, 0.04060847, 0.07710469, 0.09371186, 0.03929852, 0.02559357, 0.01296015, 0.04088101, 0.09091449, 0.04545724, 0.01659747, 0.05076059, 0.03955371, 0.0057195 , 0.07251513, 0.08344206, 0.04685593, 0.035831  , 0.07428379, 0.0 ]  # based on area

    #norm_weights = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0]
    norm_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
    
    #norm_weights = [0.05062529, 0.06853885, 0.04242881, 0.05711571, 0.05120719, 0.05711571, 0.03480489, 0.03400783, 0.03010152, 0.06960977, 0.05432958, 0.03681839, 0.06551508, 0.05500031, 0.01007924, 0.05432958, 0.07071469, 0.0479035, 0.053675, 0.05303602, 0.0]
    '''
    weights = [0.95454545, 1.29230769, 0.8, 1.07692308, 0.96551724, 1.07692308, 0.65625, 0.64122137, 0.56756757, 1.3125, 1.02439024, 0.69421488, 1.23529412, 1.03703704, 0.19004525, 1.02439024, 1.33333333, 0.90322581, 1.01204819, 1., 0.05737705]
    
    '''
    loss = 0.0
    count = 0
        
    for i in range(target.size(0)):
        for j in range(target.size(1)):
            if target[i][j] != -1:
                loss += (1-norm_weights[j]) * target[i][j] * torch.log(torch.clamp(output[i][j], min=1e-12, max=1.0)) + \
                        norm_weights[j] * (1 - target[i][j]) * torch.log(torch.clamp(1 - output[i][j], min=1e-12, max=1.0))
                #loss += target[i][j] * torch.log(torch.clamp(output[i][j], min=1e-12, max=1.0)) + \
                        #(1 - target[i][j]) * torch.log(torch.clamp(1 - output[i][j], min=1e-12, max=1.0))
                count +=1

    loss = loss / count
    
    return torch.neg(loss)

def entropy_loss(output, weights=None):
    loss = output * torch.log(output + 1e-12) 
    return torch.neg(torch.mean(loss))
    
def l1_penalty(var):
    return 0.01*torch.mean(torch.abs(var).sum())

def fm_loss(outputs, targets):
    fm_l = torch.mean(torch.abs(torch.mean(outputs, 0) - torch.mean(targets, 0)))
    return fm_l

def weigh_the_outputs(outputs):
    w_outputs = 1/(1+ torch.exp(6-12*outputs))
    return w_outputs

def BCE_Loss_1 (criterion, input, target):
    count = 0
    for i in range(input.size(0)):
        if target.data[i][0] == -1:
            target.data[i] = input.data[i]
        else:
            count +=1
    
        #print ('input: ',input.data[i])
        #print ('target: ', target.data[i])
    
    return criterion(input, target) 

def train(train_loader, model, ema_model, optimizer, epoch, log):
    global global_step

    #class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    #class_criterion = nn.BCELoss(size_average=False).cuda()
    #weights = torch.ones(args.batch_size, 20)*0.2
    #class_criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    '''
    if epoch%5 == 0: 
        filename_raw = 'output_train_raw_' + str(epoch) + '.txt'
        filename_bin = 'output_train_bin_' + str(epoch) + '.txt'
        f_raw = open(filename_raw, 'a')   
        f_bin = open(filename_bin, 'a')   
    '''
    if args.consistency_type == 'mse':
        consistency_criterion = losses.symmetric_mse_loss
        #consistency_criterion = F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes
        #consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    elif args.consistency_type == 'l1':
        consistency_criterion = nn.L1Loss(size_average=False)
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    
    for i, ((inputs, ema_input), target) in enumerate(train_loader):

        labeled_batch_idxs = []
   
        # measure data loading time
        meters.update('data_time', time.time() - end)

        adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        meters.update('lr', optimizer.param_groups[0]['lr'])

        input_var = Variable(inputs.cuda())
        ema_input_var = Variable(ema_input.cuda())
        target_var = Variable(target.cuda(async=True))

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        with torch.no_grad():
            ema_model_out, ema_feat = ema_model(ema_input_var)
        model_out, cons_feat= model(input_var)

         

        if isinstance(model_out, Variable):
            assert args.logit_distance_cost < 0
            logit1 = model_out
            ema_logit = ema_model_out
        else:
            assert len(model_out) == 2
            assert len(ema_model_out) == 2
            logit1, logit2 = model_out
            ema_logit, _ = ema_model_out

        '''
        thres = torch.ones(ema_logit.size(0), 21)*0.5
        thres = thres.cuda()

        cond = torch.ge(ema_logit, thres)
        cond = cond.type(torch.FloatTensor).cuda()
        ema_logit = Variable(cond.detach().data, requires_grad=False)
        '''
        ema_logit = Variable(ema_logit.detach().data, requires_grad=False)
        
 
        ema_feat = Variable(ema_feat.detach().data, requires_grad=False)
        cons_feat = Variable(cons_feat.detach().data, requires_grad=False)

        
        if args.logit_distance_cost >= 0:
            class_logit, cons_logit = logit1, logit2
            res_loss = args.logit_distance_cost * residual_logit_criterion(class_logit, cons_logit) / minibatch_size
            meters.update('res_loss', res_loss.item())
        else:
            class_logit, cons_logit = logit1, logit1
            res_loss = 0
        
        target_var = target_var.type(torch.FloatTensor).cuda()
            
        class_loss = weighted_binary_cross_entropy(class_logit, target_var)
        #cons_loss = weighted_binary_cross_entropy(cons_logit, ema_logit)
        meters.update('class_loss', class_loss.item())
       
        ema_logit.cuda()
  
        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            meters.update('cons_weight', consistency_weight)
            #consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit)/minibatch_size 
            #consistency_loss = consistency_weight * F.mse_loss(cons_logit, ema_logit, size_average=False) / (21*minibatch_size)
            consistency_loss = consistency_weight * torch.sum(((cons_logit - ema_logit)**2)*((ema_logit+1)**3))/ (21*minibatch_size*4) #with scaling towards 1
            #consistency_loss = consistency_weight * torch.sum((cons_logit - ema_logit)**2)/ (21*minibatch_size)
            #consistency_loss = consistency_weight * weighted_binary_cross_entropy(cons_logit, ema_logit)
            #consistency_loss = 0.05 * cons_loss 
            #consistency_loss = consistency_weight * weighted_binary_cross_entropy(cons_logit, ema_logit)
            #consistency_loss = consistency_weight * fm_loss(cons_feat, ema_feat)
            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)


        loss = class_loss + consistency_loss #+ l1_loss #+ l1_loss # ent_loss #+ res_loss
    
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())

        prec1, acc_zeros, acc_ones, acc = accuracy(class_logit.data, target_var.data, 'prec1',  topk=(1,))
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100. - prec1, labeled_minibatch_size)

        ema_prec1, ema_acc_zeros, ema_acc_ones, ema_acc = accuracy(ema_logit.data, target_var.data, 'ema_prec1', topk=(1,))
        meters.update('ema_top1', ema_prec1, labeled_minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1, labeled_minibatch_size)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print(
            'Epoch: [{0}][{1}/{2}]\t'
            'Time {meters[batch_time]:.3f}\t'
            'Data {meters[data_time]:.3f}\t'
            'Class {meters[class_loss]:.4f}\t'
            'Cons {meters[cons_loss]:.6f}\t'
            'Prec@1 {meters[top1]:.3f}'.format(
                epoch, i, len(train_loader), meters=meters))

def validate(eval_loader, mode,  model, log, global_step, epoch):
  
    if epoch%1 == 0:
        if mode=='val':
            filename_raw = 'output_val_17raw_' + str(epoch) + '.txt'
            filename_bin = 'output_val_17bin_' + str(epoch) + '.txt'
        if mode == 'ema':
            filename_raw = 'output_ema_17raw_' + str(epoch) + '.txt'
            filename_bin = 'output_ema_17bin_' + str(epoch) + '.txt'
            
        f_raw = open(filename_raw, 'a')   
        f_bin = open(filename_bin, 'a')   
 
    meters = AverageMeterSet()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    
    with torch.no_grad():
        for i, ((inputs), target) in enumerate(eval_loader):
            
            meters.update('data_time', time.time() - end)

            input_var = Variable(inputs.cuda())
            target_var = Variable(target.cuda())

            minibatch_size = len(target_var)
            labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum()
            assert labeled_minibatch_size > 0
            meters.update('labeled_minibatch_size', labeled_minibatch_size)

            # compute output
            output1, _  = model(input_var)
   
            if epoch%1 == 0:
                output_raw = output1.cpu().numpy()
                output_raw = np.roll(output_raw, 1)
                output_bin = (output_raw>0.5)*1
                np.savetxt(f_raw, output_raw, fmt='%f') 
                np.savetxt(f_bin, output_bin, fmt='%d') 
            
            class_loss = weighted_binary_cross_entropy(output1, target_var)
        
            # measure accuracy and record loss
            prec1, acc_zeros, acc_ones, acc = accuracy(output1.data, target_var.data, 'test',  topk=(1,))
            meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
            meters.update('top1', prec1, labeled_minibatch_size)
            meters.update('acc_zeros', acc_zeros, labeled_minibatch_size)
            meters.update('acc_ones', acc_ones, labeled_minibatch_size)
            meters.update('acc', acc, labeled_minibatch_size)
            meters.update('error1', 100.0 - prec1, labeled_minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                print (
                    'Test: [{0}/{1}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}'.format(
                        i, len(eval_loader), meters=meters))
    
    if epoch%1 == 0: 
        f_raw.close() 
        f_bin.close() 
    
    print (' * Prec@1 {top1.avg:.3f}'
          .format(top1=meters['top1']))
    print (' * Acc@zeros {acc_zeros.avg:.3f}'
          .format(acc_zeros=meters['acc_zeros']))
    print (' * Acc@ones {acc_ones.avg:.3f}'
          .format(acc_ones=meters['acc_ones']))
    print (' * Acc {acc.avg:.3f}'
          .format(acc=meters['acc']))
    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    LOG.info("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        LOG.info("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def accuracy(outputs, targets, phase,  topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    thres = torch.ones(targets.size(0), 21)*0.5
    thres = thres.cuda()

    cond = torch.ge(outputs, thres)
    
    count_label_ones = 0
    count_label_zeros = 0 
    correct_ones = 0
    correct_zeros = 0
    correct = 0
    total = 0  

    for i in range(targets.size(0)):
        for  j in range(21):
            if targets[i][j]==0:
                count_label_zeros +=1
            if targets[i][j]==1:
                count_label_ones +=1

    targets = targets.type(torch.ByteTensor).cuda()
    
    for i in range(targets.size(0)):
        for  j in range(21):
            if targets[i][j]==cond[i][j]:
                correct +=1
                if targets[i][j] == 0:
                    correct_zeros +=1
                elif targets[i][j] ==1:
                    correct_ones +=1
    total += targets.size(0)*21

    acc = (correct_zeros + correct_ones)*100.0/total

    avg_acc = (correct_ones/count_label_ones + correct_zeros/count_label_zeros)*100.0/2.0 
    
    acc_zeros = (100.*correct_zeros/count_label_zeros)
    acc_ones =  (100.*correct_ones/count_label_ones)
    
    
    '''
    if phase == 'test': 
        print ('Acc: %.3f%% (%d/%d) | zeros: (%d/%d) | ones: (%d/%d) | Avg Acc: %f | Acc zeros: %3f | Acc Ones: %3f |  Output zeros/ones: (%d/%d)'
                %  (100.*correct/total, correct, total, correct_zeros, count_label_zeros, correct_ones, count_label_ones, avg_acc, (100.*correct_zeros/count_label_zeros), (100.*correct_ones/count_label_ones), output_zeros, output_ones))
    '''
    return avg_acc, acc_zeros, acc_ones, acc

