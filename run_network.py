# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import torchvision
from torchvision import transforms, models
from tensorboardX import SummaryWriter

import os, time
import numpy as np
from commons.VireoFoodDatasets import *
from commons.model import *
from commons.performance import Performance
from commons.util import auto_cuda, weight_init
from commons.config import get_args


class RunNetwork():
    def __init__(self, args):
        self.args = args

        self.model = eval(args.backbone_net)(d = args.inter_dim, out_size=args.label_num)
        if len(args.gpu_device) > 1:
            divices = list(map(int, args.gpu_device))
            self.model = torch.nn.DataParallel(self.model, device_ids=divices).cuda(int(args.gpu_device[0]))
        else:
            self.model = auto_cuda(self.model, args.use_cuda, args.gpu_device)
        self.saved_model_path = "./models/%s/model_%d_lr%.e_gamma%.1f_ss%d_e%d.pth"
        self._init_models_dir()
    
    def train(self):
        args = self.args
        if args.use_writer:
            writer = SummaryWriter(comment="-%s-%s"%(args.mode, args.saved_model_dir))
            writer.add_text('Train', self._get_format_args())
        
        model = self.model
        model.apply(weight_init)
        model.train()
        model.dropout = args.dropout

        optimizer = optim.Adam(model.parameters(), lr=args.lr)  
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size, args.lr_gamma)
        niters = 0
        best_val_performance = 0
        best_epoch = 0
        time1 = time.time()
        dataset = self._get_dataloder(p=torch.zeros(558))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                shuffle=True, num_workers=32)
        if 'EfficientNetB7' in args.backbone_net:
            accumulation_steps = 4
        else:
            accumulation_steps = 1

        for i in range(args.epoch):
            train_loss = 0.
            train_acc = 0.
            acc_all = 0 
            performance = Performance(args)
            loss_func_ingre = nn.BCELoss(reduction="sum")
            print(len(dataset))
            for iters, [img1, img2, labels1, labels2] in enumerate(dataloader, start=1):
                niters += 1
                
                img1 = auto_cuda(img1, args.use_cuda, args.gpu_device)
                labels1 = auto_cuda(labels1, args.use_cuda, args.gpu_device)

                if args.pure_dynamic:
                    pass
                    output_ingre = model(img1)
                    loss = loss_func_ingre(output_ingre, labels1)
                    label_mixed = labels1
                else:
                    if 'mixup-all-all' in args.mix_style:
                        batch_size = img1.size()[0]
                        index = auto_cuda(torch.randperm(batch_size), args.use_cuda, args.gpu_device)
                        img2 = img1[index]
                        labels2 = labels1[index]
                    else:
                        img2 = auto_cuda(img2, args.use_cuda, args.gpu_device)
                        labels2 = auto_cuda(labels2, args.use_cuda, args.gpu_device)

                    img, lam = self._mixup_data(img1, img2)
                    img, labels1, labels2 = map(Variable, (img, labels1, labels2))
                    
                    output_ingre = model(img)

                    loss, label_mixed = self._mixup_criterion(loss_func_ingre, output_ingre, 
                                                labels1, labels2, lam)
                                                
                loss_ingre = torch.div(loss, img1.size()[0])
    
                loss_ingre.backward()
                if (iters % accumulation_steps) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                performance.add_data(output_ingre, label_mixed)

                if iters % args.print_freq == 0:
                    time2 = time.time()
                    if args.use_writer:
                        writer.add_scalars(
                            'LongTail-%d/Train' % args.label_num, 
                            {'train_loss': loss_ingre.item(), 'lr': optimizer.param_groups[0]['lr']}, 
                            niters)
                    
                    str_ = 'epoch {} | iters {} | Loss {:.6f} | lr:{} | time {:.2f}'
                    print(str_.format(i+1, iters, loss_ingre.item(), 
                                    optimizer.param_groups[0]['lr'], time2-time1))
                    time1 = time.time()
                    
            if args.eval_mode == 'val':
                _p, _r, _micro_F1, _macro_F1 = performance.get_results()
                print("epoch {} | P {:.4f} | R {:.4f} | micro_F1 {:.4f} | macro_F1 {:.4f}".format(
                    i+1, _p, _r, _micro_F1, _macro_F1))
                p, r, micro_F1, macro_F1, F1_sum = self.eval_val(model)
                model.train()
            elif args.eval_mode == 'train':
                p, r, micro_F1, macro_F1 = performance.get_results()
                F1_sum = performance.F1_sum
            else:
                raise NotImplementedError("eval_mode error!")

            self._save_model(model, i)
            if macro_F1 > best_val_performance:
                best_val_performance = macro_F1
                best_epoch = i+1
                torch.save(model.state_dict(), 
                    "./models/%s/best_model.pth" % (args.saved_model_dir))
            
            print("epoch {} | P {:.4f} | R {:.4f} | micro_F1 {:.4f} | macro_F1 {:.4f} | best {} - {:.4f}".format(
                i+1, p, r, micro_F1, macro_F1, best_epoch, best_val_performance))
            if args.use_writer:
                writer.add_scalars('LongTail-%d/Performance' % args.label_num, 
                    {'Precision':p, 'Recall':r, 'micro_F1': micro_F1,'macro_F1': macro_F1}, 
                    i+1)
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), niters)
                # writer.add_pr_curve("pr", label_ingre[0], output_ingre[0], niters)

            dataset = self._get_dataloder(p=F1_sum, dynamic_mode=args.dynamic_mode)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                shuffle=True, num_workers=32)

        if args.use_writer:
            writer.close()
        print('done')
        return

    def eval_val(self, model):
        args = self.args

        if 'JapaneseFood100' in args.dataset_dir:
            self.preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.6007847, 0.50769776, 0.38684255], 
                                        std = [0.27148357, 0.28454104, 0.30570388])
                ])
        else:
            input_size_init = 512 if args.input_size == 448 else 256
            self.preprocess = transforms.Compose([
                transforms.Resize((input_size_init, input_size_init)),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # Get mean and std from 10000 samples of VIREOFood172.
                transforms.Normalize(mean = [0.6007847, 0.50769776, 0.38684255], 
                                    std = [0.27148357, 0.28454104, 0.30570388])
            ])

        dataset = FoodDataset(args.dataset_dir, transforms=self.preprocess, 
            tail_threshold=args.tail_threshold, head_threshold=args.head_threshold,
            mode='validation', mix_style=args.mix_style)
        dataloader = DataLoader(dataset, batch_size=args.batch_size//2, 
                shuffle=True, num_workers=32)

        model.eval()
        
        performance = Performance(args)
        for iters, [img, label_ingre] in enumerate(dataloader, start=1):
            img = auto_cuda(Variable(img), args.use_cuda, args.gpu_device)
            label_ingre = auto_cuda(Variable(label_ingre), args.use_cuda, args.gpu_device)
            output_ingre = model(img)
            performance.add_data(output_ingre, label_ingre)
        p, r, micro_F1, macro_F1 = performance.get_results()
        return p, r, micro_F1, macro_F1, performance.F1_sum

    def eval(self):
        args = self.args

        dataset = self._get_dataloder(p=None)
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                shuffle=True, num_workers=32)
        if args.load_model_num == 'best_model':
            model_path = "./models/%s/best_model.pth" % (args.saved_model_dir)
        else :
            load_model_num = int(args.load_model_num)
            model_num = args.epoch if load_model_num == -1 else load_model_num
            model_path = self.saved_model_path % (args.saved_model_dir, args.input_size, 
                    args.lr, args.lr_gamma, args.step_size, model_num)
        model = self.model

        # pretrained_model = torch.load(model_path, 
        #     map_location=lambda storage, loc: storage.cuda([0, 1, 3]))
        # pretrained_model = torch.load(model_path, map_location='cuda')
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model)
        # model.load_state_dict({k.replace('module.',''):v for k,v in pretrained_model.items()})
        # model = auto_cuda(model, args.use_cuda, args.gpu_device)
        model.eval()
        model.dropout = False
        
        performance = Performance(args)
        for iters, [img, label_ingre] in enumerate(dataloader, start=1):
            img = auto_cuda(Variable(img), args.use_cuda, args.gpu_device)
            label_ingre = auto_cuda(Variable(label_ingre), args.use_cuda, args.gpu_device)
            output_ingre = model(img)
            
            performance.add_data(output_ingre, label_ingre)
        p, r, micro_F1, macro_F1 = performance.get_results()
        return p, r, micro_F1, macro_F1, performance.F1_sum

    def _mixup_data(self, x1, x2, alpha=1.0):
        '''Returns mixed inputs and lambda'''
        args = self.args
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

        mixed_x = lam * x1 + (1 - lam) * x2
        return mixed_x, lam

    def _mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        args = self.args
        if args.labels_mixup_style == '1':
            y_mixed = y_a + y_b
            y_mixed[y_mixed==2] = 1
            return criterion(pred, y_mixed), y_mixed
        elif args.labels_mixup_style == 'lambda':
            loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
            y_mixed = lam * y_a + (1-lam)*y_b
            return loss, y_mixed
        else:
            raise NotImplementedError(args.labels_mixup_style)

    def _get_dataloder(self, p, dynamic_mode="linear"):
        args = self.args

        if 'JapaneseFood100' in args.dataset_dir:
            self.preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.6007847, 0.50769776, 0.38684255], 
                                        std = [0.27148357, 0.28454104, 0.30570388])
                ])
        else:
            input_size_init = 512 if args.input_size == 448 else 256
            self.preprocess = transforms.Compose([
                    transforms.Resize((input_size_init, input_size_init)),
                    transforms.RandomCrop(args.input_size) if args.mode == 'train' else \
                        transforms.CenterCrop(args.input_size),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    # Get mean and std from 10000 samples of VIREOFood172.
                    transforms.Normalize(mean = [0.6007847, 0.50769776, 0.38684255], 
                                        std = [0.27148357, 0.28454104, 0.30570388])
                ])
        dataset = FoodDataset(args.dataset_dir, transforms=self.preprocess, 
            tail_threshold=args.tail_threshold, head_threshold=args.head_threshold,
            mode=args.mode, mix_style=args.mix_style, p = p, dynamic_mode=dynamic_mode)
        
        return dataset

    def _save_model(self, model, i):
        args = self.args
        if i % args.save_freq == 0:
            torch.save(model.state_dict(), 
                self.saved_model_path % (args.saved_model_dir, args.input_size, 
                    args.lr, args.lr_gamma, args.step_size, i+1))
        return 

    def _get_format_args(self):
        format_args = "|parameter|value|\n| ---------- |:----------:|\n"
        args = vars(self.args)
        for param, value in args.items():
            format_args += "| %s | %s |\n" % (param, str(value))
        return format_args

    def _init_models_dir(self):
        if not os.path.exists('models'):
            os.mkdir('models')
        models_dir = os.path.join('models', self.args.saved_model_dir)
        if not os.path.exists(models_dir):
            os.mkdir(models_dir)
        return
