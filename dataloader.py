import cv2, glob, torch
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DRVDataset(Dataset):
    def __init__(self, data_path, data_ids):
        self.data_path = data_path
        self.data_ids = data_ids
        self._load()
        
    def _load(self):
        self.video_path = []
        self.gt_path = []
        for _id in self.data_ids:
            in_files = sorted(glob.glob(self.data_path + f'/VBM4D_rawRGB/{_id}/*.png'))
            gt_files = glob.glob(self.data_path + f'/long/{_id}/half0001_*.png')[0]
            num_f = len(in_files)-2
            for i in range(1,num_f):
                ins = [in_files[i-1], in_files[i], in_files[i+1]]
                self.video_path.append(ins)
                self.gt_path.append(gt_files)
                
        print(f'\t[Info] Load DRV Dataset completed! Total len {self.__len__()}. ')
        
    def __len__(self):
        return len(self.gt_path)
    
    def __getitem__(self, idx):
        return self.video_path[idx], self.gt_path[idx]

def img_trans():
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation((-5, 5)),
        transforms.ToTensor(),
    ])
    return trans

def get_inst(inst):
    video = []
    for frame_path in inst[0]:
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.moveaxis(frame, -1, 0)
        frame = np.float32(frame/65535.0)
        video.append(frame)
    video = np.array(video)
    
    gt_img = cv2.imread(inst[1], cv2.IMREAD_UNCHANGED)
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    gt_img = np.moveaxis(gt_img, -1, 0)
    gt_img = np.float32(gt_img/65535.0)
    
    return video, gt_img

def DRV_collate_fn(batch):
    #print(len(batch),batch)
    video_batch = []
    gt_batch = []
    for inst in batch:
        video, gt_img = get_inst(inst)
        video_batch.append(video)
        gt_batch.append(gt_img)
    return torch.FloatTensor(video_batch), torch.FloatTensor(gt_batch) #(Batch,Step,C,H,W), (Batch,C,H,W)
        
