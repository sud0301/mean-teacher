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
TRAIN_DATA = 'train'
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
    train_loader, eval_loader = create_data_loaders()
    
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
        model = resnet18(pretrained=True)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        model.module.fc = nn.Sequential(*list(model.module.fc.children())[:-1], nn.Linear(512, 21))
        #self.dis.module.fc = nn.Linear(4096, 7)
        model.cuda()

        '''
        model_name = 'resnet101'
        model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Sequential(nn.Linear(2048, 20), nn.Sigmoid())
        model = torch.nn.DataParallel(model).cuda()
        '''
        '''
        model = resnet18(pretrained=True)
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        model.module.fc = nn.Sequential(*list(model.module.fc.children())[:-1], nn.Linear(512, 20), nn.Sigmoid())
        #self.dis.module.fc = nn.Linear(4096, 7)
        model.cuda()   
        '''
        #cudnn.benchmark = False
        

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    LOG.info(parameters_string(model))

    
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    '''
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    ''' 
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

    if args.evaluate:
        LOG.info("Evaluating the primary model:")
        #print ("Evaluating the primary model:")
        validate(eval_loader, model, validation_log, global_step, args.start_epoch)
        LOG.info("Evaluating the EMA model:")
        #print ("Evaluating the EMA model:")
        validate(eval_loader, ema_model, ema_validation_log, global_step, args.start_epoch)
        return

    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()
        # train for one epoch
        train(train_loader, model, ema_model, optimizer, epoch, training_log)
        LOG.info("--- training epoch in %s seconds ---" % (time.time() - start_time))

        if args.evaluation_epochs and (epoch + 1) % args.evaluation_epochs == 0:
            start_time = time.time()
            LOG.info("Evaluating the primary model:")
            #print ("Evaluating the primary model:")
            prec1 = validate(eval_loader, model, validation_log, global_step, epoch + 1)
            LOG.info("Evaluating the EMA model:")
            #print ("Evaluating the EMA model:")
            ema_prec1 = validate(eval_loader, ema_model, ema_validation_log, global_step, epoch + 1)
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
    '''
    transform_test = transforms.Compose([
        transforms.Resize(size=(224, 224), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    '''
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    
    '''
    transform_train = data.TransformTwice(transforms.Compose([
        #transforms.RandomCrop(200),
        #data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    '''
    #transform_train = transforms.Compose([
    transform_train = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
        
    
    ''' 
    transform_train = transforms.Compose([
        #transforms.RandomRotation(10),
        transforms.Resize(size=(224, 224), interpolation=2),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    '''
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])
    ''' 
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    '''
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
    dataset, labels = data.change_labels(dataset, DATA_PATH, TRAIN_LABEL_FILE)
     
    #if args.labels:
    labeled_idxs, unlabeled_idxs, dataset = data.relabel_dataset_ml(dataset, labels, args.percent)

    print (len(labeled_idxs))

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
    
    #dataset_test = torchvision.datasets.ImageFolder(evaldir, transform_test)
    #dataset_test, labels = data.change_labels(dataset_test, DATA_PATH, TEST_LABEL_FILE)

    dataset_test = dataset_processing.DatasetProcessing(
        DATA_PATH, TEST_DATA, TEST_IMG_FILE, TEST_LABEL_FILE, transform_test)
    
    eval_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2 * args.workers,  # Needs images twice as fast
        pin_memory=True,
        drop_last=False)
    
    return train_loader, eval_loader


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def weighted_binary_cross_entropy(output, target, weights=None):

    norm_weights = [0.05062529, 0.06853885, 0.04242881, 0.05711571, 0.05120719, 0.05711571, 0.03480489, 0.03400783, 0.03010152, 0.06960977, 0.05432958, 0.03681839, 0.06551508, 0.05500031, 0.01007924, 0.05432958, 0.07071469, 0.0479035, 0.053675, 0.05303602, 0.00304305]
    '''
    weights = [0.95454545, 1.29230769, 0.8, 1.07692308, 0.96551724, 1.07692308, 0.65625, 0.64122137, 0.56756757, 1.3125, 1.02439024, 0.69421488, 1.23529412, 1.03703704, 0.19004525, 1.02439024, 1.33333333, 0.90322581, 1.01204819, 1., 0.05737705]
    '''
    loss = 0.0
    
    for i in range(target.size(0)):
        for j in range(target.size(1)):
            loss += (1-norm_weights[j]) * target[i][j] * torch.log(torch.clamp(output[i][j], min=1e-12, max=1.0)) + \
                    norm_weights[j] * (1 - target[i][j]) * torch.log(torch.clamp(1 - output[i][j], min=1e-12, max=1.0))

    loss = loss / (target.size(0)*target.size(1))
    '''    
    
    if weights is not None:
        assert len(weights) == 2
         
        #loss = weights[1] * (target * torch.log(torch.clamp(output, min=1e-12, max=1.0))) + \
               #weights[0] * ((1 - target) * torch.log(torch.clamp(1 - output, min=1e-12, max=1.0)))
        
        loss = weights[1] * ((1 - output)**4) * (target * torch.log(torch.clamp(output, min=1e-12, max=1.0))) + \
               weights[0] * (output**4) * ((1 - target) * torch.log(torch.clamp(1 - output, min=1e-12, max=1.0)))
        
    else:
        loss = target * ((1 - output)**2) * torch.log(output + 1e-12) + (1 - target) * (output**2) * torch.log(1 - output + 1e-12)
    '''
    
    return torch.neg(loss)

def entropy_loss(output, weights=None):
    loss = output * torch.log(output + 1e-12) 
    return torch.neg(torch.mean(loss))
    
def l1_penalty(var):
    return 0.01*torch.mean(torch.abs(var).sum())

def fm_loss(outputs, targets):
    fm_l = torch.mean(torch.abs(torch.mean(outputs, 0) - torch.mean(targets, 0)))
    return fm_l

def convert_ema_logit_to_target(ema_logit):
    for i in range(ema_logit.size(0)):
        for j in range(ema_logit.size(1)): 
            if ema_logit.data[i][j] > 0.5:   
                ema_logit.data[i][j] = 1.0
            else:
                ema_logit.data[i][j] = 0.0
    return ema_logit

def weighted_mse_loss(outputs, targets):
    for i in range(outputs.size(0)):
        for j in range(outputs.size(1)):
            if outputs.data[i][j] > 0.5:
                outputs.data[i][j] = 0.99
        
            if targets.data[i][j] > 0.5:
                targets.data[i][j] = 0.99
    return outputs, targets
        

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
    
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    elif args.consistency_type == 'l1':
        consistency_criterion = nn.L1Loss()
    else:
        assert False, args.consistency_type
    residual_logit_criterion = losses.symmetric_mse_loss

    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    
    for i, ((inputs, ema_input), target) in enumerate(train_loader):
       
        ''' 
        scale = torch.ones(args.batch_size, 20)*0.2
        scale = scale.type(torch.LongTensor)
        
        weights = target*scale
        #print (weights) 
        weights = weights.type(torch.FloatTensor).cuda()
        class_criterion.weight = weights       
        '''
        #ema_input = inputs
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
        
        #if epoch>30:
            #target_var = convert_ema_logit_to_target(ema_logit)
         
        target_var = target_var.type(torch.FloatTensor).cuda()
        #class_loss = BCE_Loss_1(class_criterion, class_logit, target_var)/minibatch_size 
        class_loss = weighted_binary_cross_entropy(class_logit, target_var, [0.1, 0.9])
        
        ent_loss = entropy_loss(class_logit)
        
        l1_loss = 0.01*l1_penalty(class_logit)
         
        meters.update('class_loss', class_loss.item())
        meters.update('ent_loss', ent_loss.item())
        meters.update('l1_loss', l1_loss.item())
       
        #ema_logit_cons = weigh_the_outputs(ema_logit)
        #cons_logit_cons = weigh_the_outputs(cons_logit)
        #cons_logit_w, ema_logit_w = weighted_mse_loss(cons_logit, ema_logit)
 
        ema_logit.cuda()
  
        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch)
            meters.update('cons_weight', consistency_weight)
            consistency_loss = consistency_weight * consistency_criterion(cons_logit, ema_logit)/minibatch_size 
            #consistency_loss = consistency_weight * weighted_binary_cross_entropy(cons_logit, ema_logit)
            #consistency_loss = consistency_weight * fm_loss(cons_feat, ema_feat)
            #print ('cons_loss: ', consistency_loss) 
            meters.update('cons_loss', consistency_loss.item())
        else:
            consistency_loss = 0
            meters.update('cons_loss', 0)


        loss = class_loss + consistency_loss #+ l1_loss #+ l1_loss # ent_loss #+ res_loss
    
        assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.item())
        #meters.update('loss', loss.data[0])

        prec1, acc_zeros, acc_ones = accuracy(class_logit.data, target_var.data, 'prec1',  topk=(1,))
        meters.update('top1', prec1, labeled_minibatch_size)
        meters.update('error1', 100. - prec1, labeled_minibatch_size)

        ema_prec1, ema_acc_zeros, ema_acc_ones = accuracy(ema_logit.data, target_var.data, 'ema_prec1', topk=(1,))
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
        '''
        if i % args.print_freq == 0:
            LOG.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Entropy {meters[ent_loss]:.4f}\t'
                'L1 {meters[l1_loss]:.4f}\t'
                'Cons {meters[cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(train_loader), meters=meters))
            log.record(epoch + i / len(train_loader), {
                'step': global_step,
                **meters.values(),
                **meters.averages(),
                **meters.sums()
            })
        '''
        if i % args.print_freq == 0:
            print(
            'Epoch: [{0}][{1}/{2}]\t'
            'Time {meters[batch_time]:.3f}\t'
            'Data {meters[data_time]:.3f}\t'
            'Class {meters[class_loss]:.4f}\t'
            'Entropy {meters[ent_loss]:.4f}\t'
            'L1 {meters[l1_loss]:.4f}\t'
            'Cons {meters[cons_loss]:.6f}\t'
            'Prec@1 {meters[top1]:.3f}'.format(
                epoch, i, len(train_loader), meters=meters))
        

def validate(eval_loader, model, log, global_step, epoch):
    #class_criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=NO_LABEL).cuda()
    #class_criterion = nn.BCELoss(size_average=False).cuda()
    #class_criterion = nn.BCELoss(size_average=False).cuda()
    #class_criterion = nn.BCEWithLogitsLoss(size_average=False).cuda()
    
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
        
            #class_loss = class_criterion(output1, target_var) / minibatch_size
            #class_loss = BCE_Loss_1(class_criterion, output1, target_var)/minibatch_size
            class_loss = weighted_binary_cross_entropy(output1, target_var, [0.1, 0.9])
        
            # measure accuracy and record loss
            prec1, acc_zeros, acc_ones = accuracy(output1.data, target_var.data, 'test',  topk=(1,))
            meters.update('class_loss', class_loss.item(), labeled_minibatch_size)
            meters.update('top1', prec1, labeled_minibatch_size)
            meters.update('acc_zeros', acc_zeros, labeled_minibatch_size)
            meters.update('acc_ones', acc_ones, labeled_minibatch_size)
            meters.update('error1', 100.0 - prec1, labeled_minibatch_size)

            # measure elapsed time
            meters.update('batch_time', time.time() - end)
            end = time.time()

            
            if i % args.print_freq == 0:
                ''' 
                LOG.info(
                    'Test: [{0}/{1}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@5 {meters[top5]:.3f}'.format(
                        i, len(eval_loader), meters=meters))
                print (
                    'Test: [{0}/{1}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}\t'
                    'Prec@5 {meters[top5]:.3f}'.format(
                        i, len(eval_loader), meters=meters))
                '''
                print (
                    'Test: [{0}/{1}]\t'
                    'Time {meters[batch_time]:.3f}\t'
                    'Data {meters[data_time]:.3f}\t'
                    'Class {meters[class_loss]:.4f}\t'
                    'Prec@1 {meters[top1]:.3f}'.format(
                        i, len(eval_loader), meters=meters))
    '''    
    LOG.info(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}'
          .format(top1=meters['top1'], top5=meters['top5']))
    log.record(epoch, {
        'step': global_step,
        **meters.values(),
        **meters.averages(),
        **meters.sums()
    })
    '''
    print (' * Prec@1 {top1.avg:.3f}'
          .format(top1=meters['top1']))
    print (' * Acc@zeros {acc_zeros.avg:.3f}'
          .format(acc_zeros=meters['acc_zeros']))
    print (' * Acc@ones {acc_ones.avg:.3f}'
          .format(acc_ones=meters['acc_ones']))
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
    #thres = Variable(thres.cuda())

    cond = torch.ge(outputs, thres)
    #targets = targets.type(torch.ByteTensor).cuda()
    
    count_label_ones = 0
    count_label_zeros = 0 
    correct_ones = 0
    correct_zeros = 0
    correct = 0
    total = 0  
    output_zeros = 0
    output_ones = 0

    for i in range(targets.size(0)):
        for  j in range(21):
            if cond[i][j] ==0:
                output_zeros += 1
            if cond[i][j] == 1:
                output_ones += 1
    
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

    avg_acc = (correct_ones/count_label_ones + correct_zeros/count_label_zeros)*100.0/2.0 
    
    acc_zeros = (100.*correct_zeros/count_label_zeros)
    acc_ones =  (100.*correct_ones/count_label_ones)
    '''
    if phase == 'test': 
        print ('Acc: %.3f%% (%d/%d) | zeros: (%d/%d) | ones: (%d/%d) | Avg Acc: %f | Acc zeros: %3f | Acc Ones: %3f |  Output zeros/ones: (%d/%d)'
                %  (100.*correct/total, correct, total, correct_zeros, count_label_zeros, correct_ones, count_label_ones, avg_acc, (100.*correct_zeros/count_label_zeros), (100.*correct_ones/count_label_ones), output_zeros, output_ones))
    '''
    return avg_acc, acc_zeros, acc_ones

    '''
    maxk = max(topk)
    labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    '''
