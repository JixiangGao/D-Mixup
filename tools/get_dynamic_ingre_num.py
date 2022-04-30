from run_network import RunNetwork
from commons.config import get_args
import sys
import numpy as np
from commons.VireoFoodDatasets import *

"""
Run command like the followings.

python main.py --mode train --gpu_device 0 --saved_model_dir \
LongTail-383-mixup-all-20-100 --print_freq 100 --load_model_num -1 \
--tail_threshold 20 --head_threshold 100  --no_writer 

python main.py --mode train --gpu_device 1 --saved_model_dir \
LongTail-383-mixup-all-100-1000 --print_freq 100 --load_model_num -1 \
--tail_threshold 100 --head_threshold 1000  --no_writer 

"""

if __name__ == '__main__':
    args = get_args(sys.argv[1:])
    running = RunNetwork(args)

    if args.mode == 'train':
        running.train()
    else:
        p, r, micro_F1, macro_F1, F1_sum = running.eval() 
        dataset = FoodDataset(args.dataset_dir, tail_threshold=args.tail_threshold, 
            head_threshold=args.head_threshold, mode='train', mix_style=args.mix_style, 
            p = F1_sum, dynamic_mode=args.dynamic_mode)
        label_all = dataset.ingre

        ingre_num_list = [0] * 406
        for one_img_label in label_all:
            for label in one_img_label:
                ingre_num_list[label] += 1
        
        zero_shot_bool = dataset.zero_shot_bool
        for i in range(len(zero_shot_bool)-1, -1, -1):
            if zero_shot_bool[i] is False:
                del ingre_num_list[i]
        
        for i in ingre_num_list:
            print(i)

        print('len', len(ingre_num_list))
        