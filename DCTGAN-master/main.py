import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from network import Generator, Discriminator_1, Discriminator_2, FeatureExtractor
from dataset_train import TrainDataset
from dataset_test import TestDataset

from dct import dct_2d
from utils import rgb_to_ycbcr, circle
import time
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id",type=int,default=0,help="The number of the GPU")

parser.add_argument("--mode",type=str,default="test",help="train or test")
parser.add_argument("--train_mode",type=str,default="GAN",help="pretrain or GAN")

parser.add_argument("--load_G",type=str,default="./weights/",help="The pretrained 'Generator' network for additional training")
parser.add_argument("--load_D1",type=str,default="./weights/",help="The pretrained 'Discriminator' network for additional training")
parser.add_argument("--load_D2",type=str,default="./weights/",help="The pretrained 'DCT discriminator' network for additional training")

parser.add_argument("--train_dataset",type=str,default="",help='The pass of the train dataset')
parser.add_argument("--test_dataset",type=str,default="",help='The pass of the test dataset')

parser.add_argument("--network_name",type=str,default="DCTGAN",help="The name used at saving the models and producing the results")
parser.add_argument("--test_model_name",type=str,default="",help="The generator network for the test")

parser.add_argument("--param_dct",default=15,type=float,help="The threshold 'th' for making the binary maps")
parser.add_argument("--param_mask",default=96,type=int,help="The threshold 'r' for making the binary maps")
parser.add_argument("--param_alpha",default=12,type=float,help="L1Loss parameter 'alpha' of the discriminator on the 'GAN' traing")
parser.add_argument("--param_beta",default=0.1,type=float,help="Feature loss parameter 'beta' on the 'GAN' traing")
parser.add_argument("--param_gamma",default=1,type=float,help="The discriminator loss parameter 'gamma' on the 'GAN' traing")
parser.add_argument("--param_delta",default=4,type=float,help="The DCT discriminator loss parameter 'delta' on the 'GAN' traing")

parser.add_argument("--batch_size",default=8,type=int,help="The training batch size")
parser.add_argument("--epoch",default=1001,type=int,help='Training epochs')
parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")

opt = parser.parse_args()

# GPU or CPU
if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark=True
    torch.cuda.set_device(opt.gpu_id)
else:
    device = 'cpu'

    
def train():
    # Loading the dataset
    dataloader = DataLoader(
        TrainDataset(opt.train_dataset),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    
    # Difinition of networks
    generator = Generator()
    discriminator_1 = Discriminator_1()
    discriminator_2 = Discriminator_2()
    feature_extractor = FeatureExtractor()
    
    # Loading the weights
    if opt.train_mode == "GAN":
        generator.load_state_dict(torch.load(opt.load_G))
        if opt.load_D1:
            discriminator_1.load_state_dict(torch.load(opt.load_D1))
        if opt.load_D2
            discriminator_2.load_state_dict(torch.load(opt.load_D2))
    
    generator.to(device)
    discriminator_1.to(device)
    discriminator_2.to(device)
    feature_extractor.to(device)

    feature_extractor.eval()
    generator.train()
    discriminator_1.train()
    discriminator_2.train()

    # Training options
    criterion_GAN = torch.nn.MSELoss()
    criterion_L1 = torch.nn.L1Loss()
    criterion_content = torch.nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=[0.8, 0.999])
    optimizer_D_1 = torch.optim.Adam(discriminator_1.parameters(), lr=1e-4, betas=[0.8, 0.999])
    optimizer_D_2 = torch.optim.Adam(discriminator_2.parameters(), lr=1e-4, betas=[0.8, 0.999])

    valid = Variable(torch.ones((opt.batch_size, 1)), requires_grad=False).to(device)
    fake = Variable(torch.zeros((opt.batch_size, 1)), requires_grad=False).to(device)
    
    for epoch in range(opt.epoch):
        for i, imgs in enumerate(dataloader):
            if i == len(dataloader) - 1:
                continue
            train_data = imgs['Blurred']
            train_label = imgs['Sharp']
            
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            if opt.train_mode == 'pretrain':    
                ### Training the generator
                optimizer_G.zero_grad()

                g_train_output = generator(train_data)

                # Loss calculation
                loss_pre = criterion_GAN(train_label, g_train_output)
                loss_pre.backward()
                optimizer_G.step()

                print("[Epoch %d/%d] [Batch %d/%d] [G loss: %f]"
                % (epoch + 1, opt.epoch, i + 1, len(dataloader), loss_pre.item()))


            elif opt.train_mode == 'GAN':
                g_train_output = generator(train_data)
            
                ###### Training the discriminator
                optimizer_D_1.zero_grad()

                # output of the discriminator
                dis_1_label_out = Variable(discriminator_1(train_label), requires_grad=False) 
                dis_1_fake_out = Variable(discriminator_1(g_train_output.detach()), requires_grad=True) 

                # Loss calculation
                loss_D1_real = criterion_content(dis_1_label_out, valid)
                loss_D1_fake = criterion_content(dis_1_fake_out, fake)

                loss_D1 = torch.log((loss_D1_real + loss_D1_fake) / 2)

                loss_D1.backward()
                optimizer_D_1.step()

                ###### Training the DCT discriminator
                optimizer_D_2.zero_grad()

                # Preparing the mask with radius r
                masking = np.zeros([opt.batch_size, 1, 256, 256])

                masking_0 = circle(opt.param_mask)
                masking_1 = np.expand_dims(np.expand_dims(masking_0, 0), 0)

                for i in range(opt.batch_size):
                    masking[i, :, :, :] = masking_1

                masking = torch.Tensor(masking).to(device)

                # Changing RGB to YCbCr
                train_label_Y = rgb_to_ycbcr(train_label)
                g_train_output_Y = rgb_to_ycbcr(g_train_output.detach())
                
                # DCT transformation
                train_label_dct_0 = torch.abs(dct_2d(train_label_Y[:, 0, :, :]).unsqueeze(1))
                g_train_output_dct_0 = torch.abs(dct_2d(g_train_output_Y[:, 0, :, :]).unsqueeze(1))
                
                # Threshold processing
                train_label_dct_1 = train_label_dct_0 > opt.param_dct
                g_train_output_dct_1 = g_train_output_dct_0 > opt.param_dct
                
                train_label_dct_2 = train_label_dct_1.type(torch.cuda.FloatTensor)
                g_train_output_dct_2 = g_train_output_dct_1.type(torch.cuda.FloatTensor)

                # Remain the high frequency component
                train_label_dct = train_label_dct_2 * masking
                g_train_output_dct = g_train_output_dct_2 * masking
                
                dis_2_label_out = Variable(discriminator_2(train_label_dct), requires_grad=False)
                dis_2_fake_out = Variable(discriminator_2(g_train_output_dct), requires_grad=True) 
                
                # Loss Calculation
                loss_D2_real = criterion_content(dis_2_label_out, valid)
                loss_D2_fake = criterion_content(dis_2_fake_out, fake)
                
                loss_D2 = torch.log((loss_D2_real + loss_D2_fake) / 2)

                loss_D2.backward()
                optimizer_D_2.step()

                ###### Training the generator
                optimizer_G.zero_grad()

                # Adversarial loss of the generator
                loss_adv = criterion_L1(train_label, g_train_output) * opt.param_alpha

                # Content loss of the generator
                gen_features = feature_extractor(g_train_output)
                real_features = feature_extractor(train_label)
                loss_G = criterion_GAN(gen_features, real_features.detach()) * opt.param_beta

                # Adversarial loss of the discriminator
                dis_1_fake_out_mix = Variable(discriminator_1(g_train_output), requires_grad=True) 
                loss_D1_fake_mix = criterion_content(dis_1_fake_out_mix, valid)

                loss_D1_mix = torch.log(loss_D1_fake_mix) * opt.param_gamma

                # Adversarial loss of the DCT discriminator
                g_train_output_Y_0 = rgb_to_ycbcr(g_train_output)
                g_train_output_dct_00 = torch.abs(dct_2d(g_train_output_Y_0[:, 0, :, :]).unsqueeze(1))
                g_train_output_dct_11 = g_train_output_dct_00 > opt.param_dct
                g_train_output_dct_22 = g_train_output_dct_11.type(torch.cuda.FloatTensor)
                g_train_output_dct_33 = g_train_output_dct_22 * masking
                dis_2_fake_out_mix_0 = Variable(discriminator_2(g_train_output_dct_33), requires_grad=True) 
                loss_D2_fake_mix = criterion_content(dis_2_fake_out_mix_0, valid)
                
                loss_D2_mix = torch.log(loss_D2_fake_mix) * opt.param_delta

                # Total loss
                loss_mix = loss_adv + loss_G - loss_D1_mix - loss_D2_mix

                loss_mix.backward()
                optimizer_G.step()

                print("[Epoch %d/%d] [Batch %d/%d] [Total loss: %f] [G adv loss: %f] [G loss: %f] [D1 loss: %f] [D2 loss: %f]"
                % (epoch + 1, opt.epoch, i + 1, len(dataloader), loss_mix.item(), loss_adv.item(), loss_G.item(), loss_D1_mix.item(), loss_D2_mix.item()))

            else:
                NotImplementedError(opt.train_mode)
                

        # Save the models
        if (epoch) % 100 == 0:
            if opt.train_mode == 'pretrain':
                torch.save(generator.cpu().state_dict(), "saved_models/pre/"+opt.network_name+"_G_"+str(epoch)+".pth")
                generator = generator.to(device)
            if opt.train_mode == 'GAN':
                torch.save(generator.cpu().state_dict(), "saved_models/GAN/"+opt.network_name+"/"+opt.network_name+"_G_"+str(epoch) + ".pth")
                torch.save(discriminator_1.cpu().state_dict(), "saved_models/GAN/"+opt.network_name+"/"+opt.network_name+"_D1_"+str(epoch)+".pth")
                torch.save(discriminator_2.cpu().state_dict(), "saved_models/GAN/"+opt.network_name+"/"+opt.network_name+"_D2_"+str(epoch)+".pth")
                generator = generator.to(device)
                discriminator_1 = discriminator_1.to(device)
                discriminator_2 = discriminator_2.to(device)

def test():
    # Roading the network
    Network = Generator()
    param = torch.load(opt.test_model_name)
    Network.load_state_dict(param)
    Network = Network.to(device)
    Network.eval()

    # Roading the dataset
    testloader = DataLoader(
    TestDataset(opt.test_dataset),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu
    )

    with torch.no_grad():
        for i, imgs in enumerate(testloader):
            imgs = imgs.to(device)
            outputs = Network(imgs)
            print(outputs[0].cpu().numpy().transpose(1, 2, 0).shape)
            # Save the images
            cv2.imwrite("results/" + opt.network_name + "/" + str(i) + ".png", cv2.cvtColor(outputs[0].cpu().numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB) * 255)

    

if __name__ == "__main__":
    if opt.mode == "train":
        train()
    elif opt.mode == "test":
        t1 = time.time()
        test()
        t2 = time.time()
        # The calculation of the average processing time of GoPro Dataset
        elapsed_time = (t2 - t1) / 1111
        print(elapsed_time)

    else:
        NotImplementedError(opt.mode)