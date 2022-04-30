# coding: utf-8
import torch
from commons.util import auto_cuda

class Performance():
    def __init__(self, args):
        self.args = args
        self.TP_sum = torch.zeros(args.label_num)
        self.FP_sum = torch.zeros(args.label_num)
        self.FN_sum = torch.zeros(args.label_num)
        self.F1_sum = torch.zeros(args.label_num)
        zeros = torch.zeros(args.batch_size, args.label_num).type(torch.LongTensor)
        self.zeros = auto_cuda(zeros, args.use_cuda, args.gpu_device)
        ones  = torch.ones (args.batch_size, args.label_num).type(torch.LongTensor)
        self.ones  = auto_cuda(ones, args.use_cuda, args.gpu_device)

    def add_data(self, pred, label, topk=-1):
        args = self.args
        if args.batch_size == label.size(0):
            zeros, ones = self.zeros, self.ones
        else:
            zeros = torch.zeros(label.size()).type(torch.LongTensor)
            zeros = auto_cuda(zeros, args.use_cuda, args.gpu_device)
            ones  = torch.ones (label.size()).type(torch.LongTensor)
            ones  = auto_cuda(ones, args.use_cuda, args.gpu_device)

        if topk <= 0 :
            pred[pred >= 0.5] = 1
            pred[pred <  0.5] = 0
        else:
            values, indices = pred.topk(3, dim=1, largest=True, sorted=True)
            for idx in range(pred.shape[0]):
                pred[idx][ pred[idx] >= values[idx][2] ] = 1
                pred[idx][ pred[idx] <  values[idx][2] ] = 0
                
        TP = ((pred==ones )&(label==ones )).sum(0)
        FP = ((pred==ones )&(label==zeros)).sum(0)
        FN = ((pred==zeros)&(label==ones )).sum(0)
        self.TP_sum = torch.add(self.TP_sum, TP.clone().cpu())
        self.FP_sum = torch.add(self.FP_sum, FP.clone().cpu())
        self.FN_sum = torch.add(self.FN_sum, FN.clone().cpu())
        return
    
    def get_results(self):
        for i in range(self.TP_sum.size(0)):
            F1 = 2.0*self.TP_sum[i] / (2*self.TP_sum[i] + self.FP_sum[i] + self.FN_sum[i])
            if torch.isnan(F1):
                F1 = 0
            self.F1_sum[i] = F1
        p = 1.0*self.TP_sum.sum() / (self.TP_sum.sum() + self.FP_sum.sum())
        r = 1.0*self.TP_sum.sum() / (self.TP_sum.sum() + self.FN_sum.sum())
        macro_F1 = self.F1_sum.mean()
        micro_F1 = 2.0*self.TP_sum.sum() / (2*self.TP_sum.sum() + self.FP_sum.sum() + self.FN_sum.sum())

        return p, r, micro_F1, macro_F1