# DCTGAN-master
Code for "GAN-Based Image Deblurring Using DCT discriminator"

Takahiro Kudo, Hiroki Tomosada, Takanori Fujisawa, Masaaki Ikehara

Reviewing for IEEE Access 

## Overview

We propose the single image deblurring method that preserves texture and suppresses ringing artifacts in the restored 
image, named “DCTGAN.” In the proposed method, we adopt the architecture of GAN because of retaining
the details in the restored image. Besides, DCT discriminator is introduced to the proposed method. It
compares only the high frequency components of the image by discrete cosine transform (DCT) for the fake
image obtained by the generator and the real image of the correct answer. Hereby, DCTGAN may reduce
block noise or ringing artifacts while keeping the deblurring performance. Both
numerical and subjective results in the experiments show that DCTGAN is processed while retaining the
details of the restored images, and it also suppress ringing artifacts and excessive patterns.

## Test Datasets
The GoPro Dataset and Real image dataset can be downloaded via the links below:

[GoPro Test(Blurred)](https://drive.google.com/file/d/1rzAaZCrD5TTqtKAeskhdxuyo4CIhlR9J/view?usp=sharing)

[GoPro Test(Sharp)](https://drive.google.com/file/d/1rzAaZCrD5TTqtKAeskhdxuyo4CIhlR9J/view?usp=sharing)

[Real Test(Blurred)](https://drive.google.com/file/d/1dc9ToG-rRarge3z4j_OYAth8Q7QSKdep/view?usp=sharing)

Also, GoPro Dataset originated by "Nah et al." can be downloaded below :

[GoPro Dataset](https://github.com/SeungjunNah/DeepDeblur_release)

Moreover, originated Real image dataset can be downloaded below :

[Real image dataset](http://vllab.ucmerced.edu/wlai24/cvpr16_deblur_study/)

## Experimental Results
Experimental Results using pre-trained models in the paper can be downloaded as

[GoPro](https://drive.google.com/file/d/1XZfmWCvhaO95KjN6CTLEr1FcA1Y5SiZ8/view?usp=sharing)

[Real](https://drive.google.com/file/d/10e_XqajnQeiFlNk9o98uI8960Wzjl8EC/view?usp=sharing)

## Code for comparison by PSNR and SSIM
The code used the comparison of PSNR and SSIM in the numerical experiment is below : 

[Code for numerical experiment](https://drive.google.com/file/d/1TlV2UjN0JmwvhoqNe36CVZsexnwfT9Lh/view?usp=sharing)

## The pre-trained models
The experiment models in the paper can be downloaded via the link below:
<https://drive.google.com/file/d/1EkLJWUjSmbDFuSF5U5jW3hqmLcIehL8j/view?usp=sharing "Models">

For further information, please contact: {kudo, tomosada, ikehara}@tkhm.elec.keio.ac.jp

