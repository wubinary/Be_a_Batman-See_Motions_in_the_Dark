import argparse
import os
import torch
import glob
import cv2
import numpy as np
import torch.nn.functional as F

from torchvision import transforms
from model import *
#from models.model_br import Model as clstm_Model
from loss import Perceptual_Loss as perc_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir', default='/data/CV/DRV',
        type=str, help='path of DRV dataset')
    parser.add_argument(
        '--save-path', default='result',
        type=str, help='Testing result saving directory path.')
    parser.add_argument(
        '--model-path', default='trained_model/epoch_1_loss_0.0061.pt',
        type=str, help='Model loading directory path.')
    parser.add_argument(
        '--num-workers', default=10, type=int, help='Dataloader worker nums')
    parser.add_argument(
        '--gpu', default="1", type=str, help='gpu number ')
    args = parser.parse_args()
    return args

def get_ins(ins):
    video = []
    for frame_path in ins:
        frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.moveaxis(frame, -1, 0)
        frame = np.expand_dims(np.float32(frame/65535.0),axis = 0)
        video.append(frame)
    video = np.vstack(video)  # D, C, H, W
    video = torch.FloatTensor(video).unsqueeze(0)# 1, D, C, H, W
    return video


if __name__ == '__main__':

    args = parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Get Data
    data_dir = args.data_dir

    if not os.path.isdir(os.path.join(args.save_path,'video/')):
        os.makedirs(os.path.join(args.save_path,'video/'))
    if not os.path.isdir(os.path.join(args.save_path,'frames/')):
        os.makedirs(os.path.join(args.save_path,'frames/'))

    model = Model()#num_features=512, block_channel=[64, 128, 256, 512])
    model.load_state_dict(torch.load(args.model_path)['state_dict'])
    model.cuda()
    model.eval()

    height, width = 480, 688
    count = 0
    with torch.no_grad():
        count = 0
        # test dynamic videos
        for test_id in range(1,23):
            ############# Raw video ############
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video = cv2.VideoWriter('result/video/M%04d_raw.avi'%test_id,fourcc,20.0,(width,height))
            in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/M%04d/*.png'%test_id))
            for k in range(3,len(in_files)-3):
                print('running %s-th sequence %d-th raw frame...'%(test_id,k),end='    \r' if k!=len(in_files)-4 else '\n')
                
                frame_path = in_files[k]
                frame = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                frame = cv2.resize(frame, (width,height))
                frame = np.float32(frame/65535.0)
                frame = np.uint8(np.clip(frame*255,0,255))
                video.write(frame)
            video.release()

            ########### Model out video ########
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video = cv2.VideoWriter('result/video/M%04d_clstm.avi'%test_id,fourcc, 20.0, (width,height))

            in_files = sorted(glob.glob(data_dir + '/VBM4D_rawRGB/M%04d/*.png'%test_id))
            if not os.path.isdir("result/frames/M%04d"%test_id):
                os.makedirs("result/frames/M%04d"%test_id)
            for k in range(3,len(in_files)-3):

                print('running %s-th sequence %d-th model out frame...'%(test_id,k),end='   \r' if k!=len(in_files)-4 else '\n')
                in_path = [in_files[k-1], in_files[k], in_files[k+1]]
                in_frames = get_ins(in_path).cuda()

                out = model(in_frames)
                if len(out.shape) > 4:
                    _, _, c, h, w = out.shape
                    out = out.permute(0, 2, 1, 3, 4)
                    out = F.interpolate(out, size=(1,height, width), mode='trilinear',align_corners=True).squeeze().cpu().detach().numpy()
                else:
                    _, c, h, w = out.shape
                    out = F.interpolate(out, size=(height, width), mode='bilinear',align_corners=True).squeeze().cpu().detach().numpy()
                out = np.transpose(out, (1, 2, 0))
                out = np.uint8(np.clip(out*255,0,255))
                out = cv2.fastNlMeansDenoisingColored(out,None,10,10,7,21)
                cv2.imwrite("result/frames/M%04d/%04d.png"%(test_id,k+1),out)
                video.write(out)
            video.release()

