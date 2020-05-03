# Be a Batman-See motions in the dark-An CLSTM Implementation Version

![](https://i.imgur.com/eeBx9aK.png)

&nbsp;&nbsp;&nbsp;&nbsp; An CLSTM pytorch implementation of Seeing motions in the dark.
&nbsp;&nbsp;&nbsp;&nbsp; We Reproduce the idea of see motion in the dark using CLSTM model.

[#Report](https://github.com/wubinary/Be_a_Batman-See_Motion_in_the_Dark/blob/master/documents/report.pdf)

## How to run
```
# training
make train_titan ##(VRAM>21G)
make train_1080 ##(VRAM>6G)

# after training
modify makefile
line 10 : (model path)

# export vido result
make export_video
```

## Our result
* [video results](https://github.com/wubinary/Be_a_Batman-See_Motion_in_the_Dark/tree/master/result/video)
* [frame results](https://github.com/wubinary/Be_a_Batman-See_Motion_in_the_Dark/tree/master/result/frames)

### Origin paper
1. [Learning to see In the Dark](https://arxiv.org/pdf/1805.01934.pdf) [CVPR 2018]
2. [See Motion in the Dark](https://cqf.io/papers/Seeing_Motion_In_The_Dark_ICCV2019.pdf) [ICCV 2019]

---
---
# __Be a Batman (Report)__
## Overview
- [Introduction](#Introduction)
- [Related work](#Related-work)
- [Our Methods](#Our-Methods)
    - [Model](#Model)
    - [Spatial feature extraction network](#Spatial-feature-extraction-network)
    - [CLSTM](#CLSTM)
    - [Our model](#Our-model)
- [Experiments](#Experiments)
    - [Raw dark video dataset](#Raw-dark-video-dataset)
    - [Training](#Training)
    - [Result](#Result)
    - [Demo](#Demo)
- [Conclusion](#Conclusion)
- [Reference](#Reference)

## Introduction
&nbsp;&nbsp;&nbsp;&nbsp; There is some extremely low-light circumstance in our daily life, like candle dinner, outdoors under the moonlight and so on. In this regime, the traditional camera processing pipeline breaks down. Although researchers have proposed techniques for denoising, deblurring, and enhancement of low-light images. These Techniques generally assume that images are captured in somewhat dim environments with moderate levels of noise. In recent work [2], Chen et al. addressed this problem and proposed a siamese network, which gives to impressive results. However, they only consider one frame at a time during inference. Intuitively, taking the temporal correlations of consecutive frames into consideration is helpful. Therefore, we propose two methods, using CLSTM and 3D CNN, to take advantage of this useful information and obtain promising result compared with traditional pipelines.

## Related work
&nbsp;&nbsp;&nbsp;&nbsp; Chen et al.[1] first proposed a new image processing pipeline that addresses the challenges of extreme low-light photography via a data-driven approach and train deep neural networks to learn the image processing pipeline for low-light raw data, including color transformations, demosaicing, noise reduction, and image enhancement. The pipeline is trained end-to-end to avoid the noise amplification and error accumulation that characterize traditional camera processing pipelines in this regime. In our work we want to get clear videos from processing the extremely low-light videos, but in [1], it only consider spatial artifacts but not temporal artifacts.

&nbsp;&nbsp;&nbsp;&nbsp; And in [2], Chen et al. proposed a siamese network that preserves color while significantly suppressing spatial and temporal artifacts. The model was trained on static videos only but was shown to generalize to dynamic video.

## Our Methods
### Model
&nbsp;&nbsp;&nbsp;&nbsp; As the input frames are continuous in the temporal dimension, taking the temporal correlations of these frames into consideration is intuitive and presumably helpful. In terms of achieving this goal, both the 3D CNN and the CLSTM are competent. Therefore, our model has two versions, which employ the 3D CNN and the CLSTM respectively.
### Spatial feature extraction network
&nbsp;&nbsp;&nbsp;&nbsp; Spatial feature extraction is the key to the performance and processing speed. In our work, we use a structure akin to [3], where the network contains an encoder, a decoderand a multi-scale feature fusion module (MFF). We show the details of our spatial feature extraction network in Fig. 1(left). The encoder can be any 2D CNN model. Due to resource limitation, we apply a shallow ResNet-18 model as the encoder. The decoder employs four up-projection modules to improve the spatial resolution and decreases the number of channels of the feature maps. The MFF module is designed to integrate features of different scales.
![](https://i.imgur.com/k4CwWf1.png)
### CLSTM 
&nbsp;&nbsp;&nbsp;&nbsp; The structure of CLSTM is shown in Fig. 1(right). Specifically, the proposed CLSTM cell can be expressed as:
![](https://i.imgur.com/opxuudK.png)
&nbsp;&nbsp;&nbsp;&nbsp; where * is the convolutional operator. Wf, Wi, WC, Wo and bf, bi, bC, bo denote thekernels and bias terms at the corresponding convolution layers. After extracting the spatialfeatures of video frames, we concatenate ft−1 with the feature map of current frame ft to formulate a feature map with 2c channels. Next, we feed the concatenated feature map to CLSTM to update the information stored in memory cell. Finally, we concatenate the information in the updated memory cell Ct and the feature map of output gate, then feed them to next layer of CLSTM , continued until last output layer than obtain our final results.
### Our model
***Encoder + MFF + CLSTM***
![](https://i.imgur.com/h20wI9X.png)
&nbsp;&nbsp;&nbsp;&nbsp; The first version of our model is shown in Fig. 2. We use a spatial feature extraction network to obtain feature maps for each frame and then feed them into CLSTM to capture long and short term temporal dependencies. Note that the decoder is discarded in this version because of hardware resource limitation.

## Experiments
### Raw dark video dataset
&nbsp;&nbsp;&nbsp;&nbsp; We use the dataset of [2] which is collected by using a Sony RX100 VI camera, that can capture raw image sequences at approximately 16~18 frames per second in continuous shooting mode, and the buffer can keep around 110 frames in total. This is equivalent to 5.5 seconds video with 20 fps. The resolution of the image is 3672 X 5496. The
dataset include indoor and outdoor scenes.

&nbsp;&nbsp;&nbsp;&nbsp; Because it is difficult to get the ground truth of extremely low-light dynamic videos, Chen et al. [2] collected both static videos with corresponding long-exposure images as their ground truth and dynamic videos without ground truth which is used only for
perceptual experiments. Most scenes in the dataset are in the 0.5 to 5 lux range . And the dataset is proved having bias noise compared with the prediction by synthetic model which is applied to the ground truth by Chen et al.[2].

&nbsp;&nbsp;&nbsp;&nbsp; This dataset has 202 static videos for training and quantitative evaluation. Randomly divide them into approximately 64% for training, 12% for validation, and 24% for testing. Videos for the same scene are distributed within one of the sets but not across these sets. And Some scenes are in different lighting conditions. Examples are shown in Figure 4.
![](https://i.imgur.com/c6344RQ.png)

### Training
&nbsp;&nbsp;&nbsp;&nbsp; Our method is implemented using Pytorch. We train our model on an Nvidia GTX 1080Ti GPU with 11 GB of memory or on an Nvidia TITAN RTX GPU with 24 GB of memory. We use the L1/L2 loss and the AdamW optimizer, setting the batch size to 2. The initial learning rate is 10 . We keep training the network 10^−4 until validation loss doesn’t improve for 3 epochs.

### Result
![](https://i.imgur.com/4GuEYo0.jpg)
&nbsp;&nbsp;&nbsp;&nbsp; Figures 5. shows that the lighter circumstances are the more precise result we can get by our model. Otherwise, the results will contain more artifacts.
### Demo
Youtube video demo ([https://www.youtube.com/watch?v=jp6JZnTpg9k](https://www.youtube.com/watch?v=jp6JZnTpg9k))

## Conclusion
&nbsp;&nbsp;&nbsp;&nbsp; In this work, we first consider the relation between consecutive reference frames
using CLSTM and 3D-CNN. Quantitative and qualitative analysis demonstrate that our
method achieves promising results compared with other traditional pipelines but there is
still room for improvement. Besides, while 3D CNN maintains better temporal cons- istency,
it often leads to blurry frames. On the other hand, CLSTM gives to better results with
respect to frame quality but fails to keep temporal consistency.

## Reference
[1] C. Chen, Q. Chen, J. Xu, and V. Koltun. Learning to see in the dark. In The IEEE
Conference on Computer Vision and Pattern Recognition (CVPR), 2018.
[2] C. Chen, Q. Chen, Minh N. Do, and V. Koltun. Seeing motion in the dark. In The IEEE
International Conference on Computer Vision (ICCV), 2019.
[3] H. Zhang, C. Shen, Y. Li, Y. Cao, Y. Liu, and Y. Yan. Exploiting temporal consistency for
real-time video depth estimation. In The IEEE Conference on Computer Vision and
Pattern Recognition (CVPR), 2019
