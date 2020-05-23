from model import * 
from loss import *

import os, glob 

def _run_epoch(dataloader, model, opt, criterion):
    
    model.train()
    
    total_loss = 0
    for idx, (video, gt_img) in enumerate(dataloader):
        b, s, c, h, w = video.shape
        video = video.cuda()
        gt_img = gt_img.cuda()
        
        opt.zero_grad()
        loss = criterion(model(video), gt_img)
        loss.backward()
        opt.step()
        
        total_loss += loss.item() * b
        print('\t [{}/{}] Traing Loss:{:.4f} '.format(idx+1, len(dataloader), loss.item()),
             end='      \r')        
        torch.cuda.empty_cache()
    
    avg_loss = total_loss/len(dataloader.dataset)
    return avg_loss
    
def _run_val(dataloader, model, criterion):
    
    model.eval()
    
    with torch.no_grad():
        total_loss = 0
        for idx, (video, gt_img) in enumerate(dataloader):
            b, s, c, h, w = video.shape
            video = video.cuda()
            gt_img = gt_img.cuda()
            
            out = model(video)
            loss = criterion(out, gt_img)
            
            total_loss += loss.item() * b
            
            print('\t [{}/{}] Valid Loss:{:.4f} '.format(idx+1, len(dataloader), loss.item()),
                 end='      \r')
            torch.cuda.empty_cache()
            
    avg_loss = total_loss/len(dataloader.dataset)
    return avg_loss

def train(args, train_dataloader, valid_dataloader):
    torch.manual_seed(87)
    torch.cuda.manual_seed_all(87)
    
    model = Model()
    if os.path.exists(args.load_model_path):
        load_model_path = glob.glob(args.model_path+f'/epoch_{args.from_epoch}*')[0]
        model.load_state_dict(torch.load(load_model_path)['state_dict'])
        print(f'\t[Info] Resume traininig from {model_path}')
    model.cuda()
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #criterion = Perceptual_Loss().cuda()
    criterion = Consistency_Loss().cuda() 

    best_loss = None
    for epoch in range(args.from_epoch+1, args.from_epoch+args.epoch+1):
        print(f' Epoch {epoch}')
        
        avg_train_loss = _run_epoch(train_dataloader, model, opt, criterion)
        print('\t [Info] Avg Traing Loss:{:.4f} '.format(avg_train_loss))
        
        avg_valid_loss = _run_val(valid_dataloader, model, criterion)
        print('\t [Info] Avg Valid Loss:{:.4f} '.format(avg_valid_loss))
        
        if best_loss is None or avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            save_path = "{}/epoch_{}_loss_{:.4f}.pt".format(
                args.model_path, epoch, avg_valid_loss)
            torch.save({'state_dict': model.state_dict()}, save_path)
            print(f'\t [Info] save weights at {save_path}')
        else:
            for param_group in opt.param_groups:
                param_group['lr'] /= 5
        print('---------------------------------------------------')
       
