import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as trans

import os
import json
import numpy as np
from random import shuffle, randint, random
import piexif
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class FoodDataset(Dataset):
    def __init__(self, path, tail_threshold, head_threshold, mix_style='mixup-all-all', 
                transforms=None, mode='train', p=None, dynamic_mode="linear"):
        super(FoodDataset, self).__init__()

        self.transforms = transforms if transforms != None else trans.ToTensor() 
        self.tail_threshold = tail_threshold
        self.head_threshold = head_threshold
        self.mix_style = mix_style
        self.path = path
        self.mode = mode
        if 'VireoFood251' in path:
            self.imgs_dir = "VireoFood251_ori"
            self.label_dir = 'SplitAndIngreLabel-406'
        elif 'JapaneseFood100' in path:
            self.imgs_dir = "food100_224"
            self.label_dir = 'SplitAndIngreLabel'
        elif 'VireoFood172' in path:
            self.imgs_dir = 'VireoFood172_ori'
            self.label_dir = 'SplitAndIngreLabel'
        elif 'UECFood100' in path:
            self.imgs_dir = "UECFOOD100"
            self.label_dir = 'SplitAndIngreLabel'
        elif 'NUSWIDE' in path:
            self.imgs_dir = "Flickr"
            self.label_dir = "SplitAndLabel"
        elif 'OID_v6_pre' in path:
            self.path = self.path.replace('OID_v6_pre', 'OID_v6')
            self.imgs_dir = mode + '-lt-pre'
            self.label_dir = 'annotations/d-mixup-pre'
        elif 'OID_v6' in path:
            self.imgs_dir = self.mode + '-lt'
            self.label_dir = 'annotations/d-mixup'
        else:
            raise NotImplementError(path)

        self._init_data()

        if 'VireoFood172' in path:
            self.path = '../../dataset/VireoFood251/'
            self.imgs_dir = "VireoFood251_ori"
            
        self.zero_shot_bool, self.map_dict_406_383 = self._get_zero_shot_bool()
        self.ing_tail_bool, self.ing_head_bool, self.train_ingre_cnt_383 = \
            self._get_ing_train_tail_head_bool_383()

        if 'mixup-tail-tail' in self.mix_style:
            self.tail_samples_list, self.head_samples_list = \
                self._get_tail_head_samples_list(self.imgs, self.ingre)
            self.tail_samples_list_index = 0

        if p is not None and mode == 'train':
            self.set_probability(p, dynamic_mode=dynamic_mode)
            self._gen_dynamic_dataset()
        
    def _get_tail_head_samples_list(self, img_name_list, labels_list):
        '''
            Only used in mixup-tail-tail.
            Returns:
                tail_samples_list: the list of (img_name, labels) that belongs to tail.
                head_samples_list: the list of (img_name, labels) that belongs to head.
        '''
        tail_samples_list = []
        head_samples_list = []
        if self.mode == "train":
            for img_name, labels in zip(img_name_list, labels_list):
                tail_flag, head_flag = False, False
                for label in labels:
                    maped_label = self.map_dict_406_383[label]
                    if self.ing_tail_bool[maped_label]:
                        tail_flag = True
                    if self.ing_head_bool[maped_label]:
                        head_flag = True
                if tail_flag:
                    tail_samples_list.append((img_name, labels))
                if head_flag:
                    head_samples_list.append((img_name, labels))
        shuffle(tail_samples_list)
        shuffle(head_samples_list)
        return tail_samples_list, head_samples_list

    def _gen_dynamic_dataset(self):
        imgs_dynamic, ingre_dynamic = [], []

        for img, ingre in zip(self.imgs, self.ingre):
            this_sample_used_times = 0
            for i in ingre:
                used_times = 0
                i_in_383 = self.map_dict_406_383[i]

                class_num_base = self.train_ingre_cnt_383[i_in_383]
                if class_num_base < self.tail_threshold:
                    class_num = self.tail_threshold 
                else:
                    class_num = class_num_base
                class_num *= self.prob[i_in_383]
                times = class_num / class_num_base

                used_times += int(times)
                times_tmp = times - int(times)
                if times_tmp >= random():
                    used_times += 1
                    
                if used_times > this_sample_used_times:
                    this_sample_used_times = used_times

            for i in range(this_sample_used_times):
                imgs_dynamic.append(img)
                ingre_dynamic.append(ingre)
            
        self.imgs = imgs_dynamic
        self.ingre = ingre_dynamic
        return

    def _init_data(self): 
        self.train_data = self._get_data_list(os.path.join(self.label_dir, "train_label.npy"))
        self.val_data   = self._get_data_list(os.path.join(self.label_dir, "val_label.npy"))
        self.test_data  = self._get_data_list(os.path.join(self.label_dir, "test_label.npy"))
        
        if self.mode == "test":
            print("testing...")
            self.imgs, self.ingre = self.test_data
        elif self.mode == "validation":
            print("validating...")
            self.imgs, self.ingre = self.val_data
        elif self.mode == "train":
            print("---------------------TRAINING---------------------")
            self.imgs, self.ingre = self.train_data
        else:
            raise NotImplementedError(self.mode)
        return

    def _get_data_list(self, label_file):
        img_name_lst, label_lst= [], []
        f = np.load(os.path.join(self.path, label_file), allow_pickle=True)
        for line_list in f:
#             if len(line_list) <= 1 and 'test_label' not in label_file:
#                 continue
#             if len(line_list) <= 1:
#                 continue
            img_name_lst.append(line_list[0])
            label_lst.append(line_list[1:])
        return (img_name_lst, label_lst)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # load image
        img_name = self.imgs[idx]
        img = self._get_img_tensor(img_name)
        # get ingre labels
        ingre_item = self.ingre[idx]
        ingre_result = self._get_labels_tensor_383(ingre_item)

        if self.mode == 'validation' or self.mode =='test':
            return img, ingre_result
        elif self.mode == 'train':
            # get mixup pair
            return self._mix_img_label(img, ingre_result)
        else:
            raise NotImplementedError(self.mode)
    
    def _mix_img_label(self, img, ingre_result):
        if 'mixup-all-all' in self.mix_style:
            return img, img, ingre_result, ingre_result

        elif 'mixup-tail-all' in self.mix_style:
            tail_ingres_num = ingre_result[self.ing_tail_bool].sum()
            if tail_ingres_num > 0:
                rand_idx = randint(0, len(self.imgs)-1)
                img2_name, ingre2_item = self.imgs[rand_idx], self.ingre[rand_idx]
                img2 = self._get_img_tensor(img2_name)
                ingre2_result = self._get_labels_tensor_383(ingre2_item)
                return img, img2, ingre_result, ingre2_result
            else:
                return img, img, ingre_result, ingre_result
        elif 'mixup-tail-tail' in self.mix_style:
            tail_ingres_num = ingre_result[self.ing_tail_bool].sum()
            if tail_ingres_num > 0:
                idx = self.tail_samples_list_index % len(self.tail_samples_list)
                self.tail_samples_list_index += 1
                img2_name, ingre2_item = self.tail_samples_list[idx]
                img2 = self._get_img_tensor(img2_name)
                ingre2_result = self._get_labels_tensor_383(ingre2_item)
                return img, img2, ingre_result, ingre2_result
            else:
                return img, img, ingre_result, ingre_result
        else:
            raise NotImplementedError(self.mix_style)

    def _get_img_tensor(self, img_name):
        img_path = os.path.join(self.path, self.imgs_dir, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            piexif.remove(img_path)
            img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return img

    def _get_labels_tensor_383(self, ingre_item):
        ingre_result = torch.zeros(558)
        for i in ingre_item:
            ingre_result[int(i)] = 1
        ingre_result = self._remove_zero_shot(ingre_result)
        return ingre_result

    def _remove_zero_shot(self, ingre_list):
        return ingre_list[self.zero_shot_bool]

    def _get_zero_shot_bool(self):

        def _get_zero_shot_ing_idx_lst():
            """
                Get the index(0-405) that the samples' number in train, val or test is 0.
            """
            zero_shot_ing_idx_lst = []
            for mode_data in [self.train_data[1], self.val_data[1], self.test_data[1]]:
                ingre_count = [0]*558
                for one_sample_label in mode_data:
                    for ingre in one_sample_label:
                        ingre_count[ingre] += 1
                for i, idx_conut in enumerate(ingre_count):
                    if idx_conut == 0:
                        if i not in zero_shot_ing_idx_lst:
                            zero_shot_ing_idx_lst.append(i)
            return zero_shot_ing_idx_lst

        zero_shot_ing_idx_lst = _get_zero_shot_ing_idx_lst()
        zero_shot_bool = [True]*558
        for zero_shot_ing in zero_shot_ing_idx_lst:
            zero_shot_bool[zero_shot_ing] = False
        
        # get the index map relation between 406 and 383
        map_dict_406_383 = {}
        j = 0
        for i, not_zero in enumerate(zero_shot_bool):
            if not_zero:
                map_dict_406_383[i] = j
                j += 1
            else:
                map_dict_406_383[i] = -1
        
        return zero_shot_bool, map_dict_406_383

    def _get_ing_train_tail_head_bool_383(self):
        ''' 
            Get the 383d bool lists that indicate
                the tail and head ingredients. 
            Tail ingres would be set as True in ing_tail_bool.
            Head ingres would be set as True in ing_head_bool.
        '''

        train_labels = self.train_data[1]
        train_ingre_count = [0]*558
        for one_sample_label in train_labels:
            for ingre in one_sample_label:
                train_ingre_count[ingre] += 1 

        # remove the zero-shot ingres.
        j = 0
        train_ingre_cnt_383 = []
        ing_tail_bool = [False]*558
        ing_head_bool = [False]*558
        for i, ing_ori in enumerate(train_ingre_count):
            if self.zero_shot_bool[i]:
                train_ingre_cnt_383.append(ing_ori)
                if ing_ori <= self.tail_threshold:
                    ing_tail_bool[j] = True
                if ing_ori > self.head_threshold:
                    ing_head_bool[j] = True
                j += 1
        return ing_tail_bool, ing_head_bool, train_ingre_cnt_383

    def set_probability(self, p, dynamic_mode='linear'):
        prob = 1 - p
        if dynamic_mode == 'linear':
            self.prob = 1 - p
        elif dynamic_mode == 'square':
            self.prob = 1 - torch.pow(p, 2)
        # elif dynamic_mode == 'cube':
        #     self.prob = torch.pow(prob, 3)
        elif dynamic_mode == 'pow4':
            self.prob = 1 - torch.pow(p, 4)
        else:
            raise NotImplementedError('dynamic_mode error!')

if __name__  == '__main__': 
    import time
    start = time.time()

    dataset = FoodDataset('../../dataset/OID_v6/', 1, 10000, 'mixup-all-all', mode='validation',
        dynamic_mode='square', p=torch.tensor([0 for i in range(383)]))
    count = 0
    for i, j in dataset:
        count += 1
        print(count)
        pass

    # p1 = torch.tensor([0 for i in range(383)])
    # p2 = torch.zeros(383)
    # print(1-torch.pow(p1, 2))
    # print(1-torch.pow(p2, 2))
    end = time.time()
    print(len(dataset))
    print(end-start)
    # for i in range(len(dataset)):
    #     pass
