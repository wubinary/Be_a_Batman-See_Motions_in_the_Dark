import os, sys, warnings, argparse

from dataloader import *
from train import *

def parse_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument('--data-path', default='/media/disk2/aa/SeeInDark',
                        type=str, help='path of DRV dataset')
    parser.add_argument('--num-workers', default=10,
                       type=int, help='Dataloader num of workers')
    parser.add_argument('--batch-size', default=4,
                        type=int, help='batch size')
    parser.add_argument('--epoch', default=8,
                       type=int, help='epoch to train')
    parser.add_argument('--lr', default=1e-4, 
                        type=float, help='learning rate')    
    parser.add_argument('--save-path', default='trained_model',
                        type=str, help='h5 Model file saved directory')
    parser.add_argument('--load-model-path', default='trained_model/epoch_0_loss_0.0062.pt',
                        type=str, help='load Model path')
    parser.add_argument('--gpu', default='0',
                        type=str, help='gpu number')
    args = parser.parse_args()
    if not os.path.isdir(args.save_path): os.makedirs(args.save_path)
    return args

if __name__ == '__main__':
   
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu#"0"
    warnings.filterwarnings('ignore')
    
    # Dataset ids
    train_ids = [_id.strip('\n') for _id in open(args.data_path + '/train_list.txt')]
    valid_ids = [_id.strip('\n') for _id in open(args.data_path + '/val_list.txt')]
    
    train_dataset = DRVDataset(args.data_path, train_ids)
    valid_dataset = DRVDataset(args.data_path, valid_ids)
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DRV_collate_fn,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        dataset = valid_dataset,
        batch_size=args.batch_size*4,
        shuffle=False,
        collate_fn=DRV_collate_fn,
        num_workers=args.num_workers
    )
    
    # Training
    train(args, train_dataloader, valid_dataloader)
    
        
