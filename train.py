from model import * 
from loss import *

import os, glob, json

def mean(ls):
    return sum(ls)/(len(ls)+1e-8)

def _run_epoch(dataloader, model, opt, criterion):
    
    model.train()
    
    step_loss = [] 
    for idx, (video, gt_img) in enumerate(dataloader):
        b, s, c, h, w = video.shape
        video = video.cuda()
        gt_img = gt_img.cuda()
        
        opt.zero_grad()
        loss = criterion(model(video), gt_img)
        loss.backward()
        opt.step()
        
        step_loss.append(loss.item())
        print('\t [{}/{}] Traing Loss:{:.4f} '.format(idx+1, len(dataloader), mean(step_loss)),
             end='      \r')        
        torch.cuda.empty_cache()
    
    return step_loss 
    
def _run_val(dataloader, model, criterion):
    
    model.eval()
    
    with torch.no_grad():
        step_loss = []
        for idx, (video, gt_img) in enumerate(dataloader):
            b, s, c, h, w = video.shape
            video = video.cuda()
            gt_img = gt_img.cuda()
            
            out = model(video)
            loss = criterion(out, gt_img)
            
            step_loss.append(loss.item())
            print('\t [{}/{}] Valid Loss:{:.4f} '.format(idx+1, len(dataloader), mean(step_loss)),
                 end='      \r')
            torch.cuda.empty_cache()
            
    return step_loss

def train(args, train_dataloader, valid_dataloader):
    torch.manual_seed(87)
    torch.cuda.manual_seed_all(87)
    
    model = Model()
    #if os.path.exists(args.model_path):
    if len(glob.glob(args.model_path+f'/epoch_{args.from_epoch}*')) > 0:
        load_model_path = glob.glob(args.model_path+f'/epoch_{args.from_epoch}*')[0]
        model.load_state_dict(torch.load(load_model_path)['state_dict'])
        print(f'\t[Info] Resume traininig from {load_model_path}')
    model.cuda()
    
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    #criterion = Perceptual_Loss().cuda()
    criterion = Consistency_Loss().cuda() 

    train_step_loss, valid_step_loss, best_loss = [], [], None
    for epoch in range(args.from_epoch+1, args.from_epoch+args.epoch+1):
        print(f' Epoch {epoch}')
        print('\t [Info] lr:{:e}'.format(args.lr))
        
        loss = _run_epoch(train_dataloader, model, opt, criterion)
        train_step_loss += loss #list
        print('\t [Info] Avg Traing Loss:{:.4f} '.format(mean(loss)))
        
        loss = _run_val(valid_dataloader, model, criterion)
        valid_step_loss += loss #list
        print('\t [Info] Avg Valid Loss:{:.4f} '.format(mean(loss)))
        
        ## change lr
        avg_valid_loss = mean(loss)
        if best_loss is None or avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
        else:
            args.lr /= 2
            for param_group in opt.param_groups:
                param_group['lr'] /= args.lr 
        
        ## save path
        save_path = "{}/epoch_{}_loss_{:.4f}.pt".format(
                args.model_path, epoch, avg_valid_loss)
        torch.save({'state_dict': model.state_dict()}, save_path)
        print(f'\t [Info] save weights at {save_path}') 
       
        ## save learning curve
        with open(args.model_path+'/lr_curve.json', 'w') as f:
            json.dump({'train':train_step_loss,'valid':valid_step_loss},f,indent=4)
        
        print('---------------------------------------------------')
       
