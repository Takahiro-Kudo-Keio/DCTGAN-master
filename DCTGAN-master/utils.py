from torch.autograd import Variable
import numpy as np

# YCbCr transformation
def rgb_to_ycbcr(img_input):
    output = Variable(img_input.data.new(*img_input.size()))
    output[:, 0, :, :] = img_input[:, 0, :, :] * (65.481 / 255) + img_input[:, 1, :, :] * (128.553 / 255) + img_input[:, 2, :, :] * (24.966 / 255) + (16 / 255)

    return output

# Making the mask
def circle(r):
    output = np.ones((256, 256))
    for i in range(256):
        for j in range(256):
            R = i ** 2 + j ** 2
            if R < r ** 2:
                output[i,j] = 0 
    return output