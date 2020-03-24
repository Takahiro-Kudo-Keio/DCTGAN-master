# DCTGAN-master
Code for "GAN-Based Image Deblurring Using DCT discriminator"

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

## Train and Test Datasets
The GoPro Dataset and Real image dataset can be downloaded via the links below:
[GoPro Train] : 
[GoPro Test] : 

Also, this dataset can be downloaded via
[GoPro Dataset]



## Experimental Results
Experimental Results using pre-trained models below in the paper can be downloaded as
[GoPro] :
[Real] :

## The pre-trained models
The experiment models in the paper can be downloaded via the link below:
https://drive.google.com/file/d/1EkLJWUjSmbDFuSF5U5jW3hqmLcIehL8j/view?usp=sharing

