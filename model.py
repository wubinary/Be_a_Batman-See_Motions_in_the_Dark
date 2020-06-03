import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable 

################################################################
######################  Convolution LSTM  ######################

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        #assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = Variable(torch.zeros(1, hidden, shape[0], shape[1]).cuda())
            self.Wcf = Variable(torch.zeros(1, hidden, shape[0], shape[1]).cuda())
            self.Wco = Variable(torch.zeros(1, hidden, shape[0], shape[1]).cuda())
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda()),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1]).cuda()))


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is a list of succeeding lstm layers.
    def __init__(self, input_channels, hidden_channels, kernel_size, step=1, effective_step=[1]):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        internal_state = []
        outputs = []
        for step in range(self.step):
            x = input[:,step,:,:,:]
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, height, width = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=(height, width))
                    internal_state.append((h, c))

                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step in self.effective_step:
                outputs.append(x.unsqueeze(1))
                
            torch.cuda.empty_cache()
            
        outputs = torch.cat(outputs, dim=1)
        return x[0], outputs # (x:final step result, outputs:all step result)
    

################################################################
#######################  3D-Convolution  #######################

class CNN_3D(nn.Module):
    def __init__(self, in_channels, hidd_channels=32, num_classes=2):
        super(CNN_3D, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=hidd_channels, 
                         kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm3d(hidd_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            )
        
        self.block2 = nn.Sequential(
                nn.Conv3d(in_channels=hidd_channels, out_channels=hidd_channels*2, 
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(hidd_channels*2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            )
        
        hidd_channels *= 2
        self.block3 = nn.Sequential(
                nn.Conv3d(in_channels=hidd_channels, out_channels=hidd_channels*2, 
                         kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(hidd_channels*2),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            )
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(hidd_channels*2*(1**3), num_classes)
        
    def forward(self, x):
        x = x.permute(0,2,1,3,4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avg_pool(x)
        x = x.contiguous().view(x.shape[0],-1) #(batch, vec_size)
        x = self.fc(x)
        return nn.Sigmoid()(x)
    
    
################################################################
########################  Auto-Encoder  ########################

def cubes_2_maps(cubes):
    b, d, c, h, w = cubes.shape
    return cubes.contiguous().view(b*d, c, h, w), b, d

def maps_2_cubes(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d, x_c, x_h, x_w)
    return x # B, D, C, H, W

def maps_2_maps(x, b, d):
    x_b, x_c, x_h, x_w = x.shape
    x = x.contiguous().view(b, d * x_c, x_h, x_w)
    return x

class UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(UpProjection, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=True) #原 bilinear
        x_conv1 = self.relu(self.bn1(self.conv1(x.contiguous())))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))

        out = self.relu(bran1 + bran2)
        return out

class Encoder(nn.Module):
    def __init__(self, original_model):
        super(Encoder, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)
        return x_block1, x_block2, x_block3, x_block4

class Decoder(nn.Module):
    def __init__(self, num_features = 512):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features //2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)
        self.up1 = UpProjection(
                    num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up2 = UpProjection(
                    num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up3 = UpProjection(
                    num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
        self.up4 = UpProjection(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

    def forward(self, x_block1, x_block4):
        x_d0 = F.relu(self.bn(self.conv(x_block4)))
        x_d1 = self.up1(x_d0, [x_block1.size(2)//4, x_block1.size(3)//4])
        x_d2 = self.up2(x_d1, [x_block1.size(2)//2, x_block1.size(3)//2])
        x_d3 = self.up3(x_d2, [x_block1.size(2)  , x_block1.size(3)  ])
        x_d4 = self.up4(x_d3, [x_block1.size(2)*2, x_block1.size(3)*2])
        return x_d4

class MFF(nn.Module):
    def __init__(self, block_channel, num_features=32):
        super(MFF, self).__init__()
        self.up1 = UpProjection(
                   num_input_features=block_channel[0], num_output_features=16)
        self.up2 = UpProjection(
                    num_input_features=block_channel[1], num_output_features=16)
        #self.up3 = UpProjection( 
        #            num_input_features=block_channel[2], num_output_features=8)
        #self.up4 = UpProjection(
        #           num_input_features=block_channel[3], num_output_features=8)
        self.conv = nn.Conv2d(
                    num_features, num_features, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        x_m1 = self.up1(x_block1, size)
        x_m2 = self.up2(x_block2, size)
        #x_m3 = self.up3(x_block3, size)
        #x_m4 = self.up4(x_block4, size)

        #x = self.bn(self.conv(torch.cat((x_m1, x_m2, x_m3, x_m4), axis=1)))
        x = self.bn(self.conv(torch.cat((x_m1, x_m2,), axis=1)))
        x = F.relu(x)
        return x

class Model(nn.Module):
    def __init__(self, num_features=512):
        super(Model, self).__init__()
        self.encoder = Encoder(models.resnet18(pretrained=True))
        for p in self.encoder.parameters():
            p.requires_grad = False 
        self.decoder = Decoder(num_features)
        self.MFF = MFF(block_channel=[64,128,256,512], num_features=32)
        self.clstm = ConvLSTM(input_channels=48, hidden_channels=[32,3], 
                kernel_size=3, step=3, effective_step=[0,1,2])
            
    def forward(self, x):
        x, b, d = cubes_2_maps(x)
        x_block1, x_block2, x_block3, x_block4 = self.encoder(x)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_block1.size(2)*2,x_block1.size(3)*2])
        x_decoder = self.decoder(x_block1, x_block4)
        x_decoder = torch.cat((x_decoder,x_mff), axis=1)
        x_decoder = maps_2_cubes(x_decoder, b, d)
        _, out = self.clstm(x_decoder)
        out = (out+1.)/2.

        return out


