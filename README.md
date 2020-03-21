# DCTGAN-master
Code for this paper [GAN-Based Image Deblurring Using the DCT discriminator]

Takahiro Kudo, Hiroki Tomosada, Takanori Fujisawa, Masaaki Ikehara

IEEE Access 

## Overview

We propose the single image deblurring method that preserves texture and suppresses ringing artifacts in the restored 
image, named “DCTGAN.” In the proposed method, we adopt the architecture of GAN because of retaining
the details in the restored image. Besides, DCT discriminator is introduced to the proposed method. It
compares only the high frequency components of the image by discrete cosine transform (DCT) for the fake
image obtained by the generator and the real image of the correct answer. Hereby, DCTGAN may reduce
block noise or ringing artifacts while keeping the deblurring performance. Both
numerical and subjective results in the experiments show that DCTGAN is processed while retaining the
details of the restored images, and it also suppress ringing artifacts and excessive patterns.

![](./doc_images/kohler_visual.png)
![](./doc_images/restore_visual.png)
![](./doc_images/gopro_table.png)
![](./doc_images/lai_table.png)
<!---![](./doc_images/dvd_table.png)-->
<!---![](./doc_images/kohler_table.png)-->

## DeblurGAN-v2 Architecture

![](./doc_images/pipeline.jpg)

<!---Our architecture consists of an FPN backbone from which we take five final feature maps of different scales as the 
output. Those features are later up-sampled to the same 1/4 input size and concatenated into one tensor which contains 
the semantic information on different levels. We additionally add two upsampling and convolutional layers at the end of 
the network to restore the original image size  and reduce artifacts. We also introduce a direct skip connection from 
the input to the output, so that the learning focuses on the residue. The input images are normalized to \[-1, 1\].
 e also use a **tanh** activation layer to keep the output in the same range.-->

<!---The new FPN-embeded architecture is agnostic to the choice of feature extractor backbones. With this plug-and-play 
property, we are entitled with the flexibility to navigate through the spectrum of accuracy and efficiency. 
By default, we choose ImageNet-pretrained backbones to convey more semantic-related features.--> 

## Train and Test Datasets

The datasets for training can be downloaded via the links below:
- [GoPro](https://drive.google.com/file/d/1KStHiZn5TNm2mo3OLZLjnRvd0vVFCI0W/view)


## Training

## Testing

## Pre-trained models

## Results of the paper experiment
