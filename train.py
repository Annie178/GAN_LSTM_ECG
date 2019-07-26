import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn as nn

# For testing using the MNIST dataset
from torchvision import transforms, datasets
from torch.autograd.variable import Variable
from torch.utils.data import Dataset, DataLoader
sns.set(rc={'figure.figsize':(11, 4)})

import datetime 
from datetime import date
today = date.today()

import random
import json as js
import pickle
import os

from model import Discriminator, Generator, MinibatchDiscrimination, noise
from data import ECGData, PD_to_Tensor, GetECGData
from mmd import pdist, MMDStatistic,  pairwisedistances


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Loading in training data
source_filename = './mitbih_train.csv'
ECG_data = GetECGData(source_file = source_filename,class_id = 0)

sample_size = 119
data_loader = torch.utils.data.DataLoader(ECG_data, batch_size=sample_size, shuffle=True)
# Num batches
num_batches = len(data_loader)

#Get Test Data
test_filename =  './mitbih_test.csv'

data_test = GetECGData(source_file = test_filename,class_id = 0)
data_loader_test = torch.utils.data.DataLoader(sine_data_test[:18088], batch_size=sample_size, shuffle=True)

#Defining parameters
seq_length = sine_data[0].size()[0] #Number of features

hidden_nodes_g = 50
hidden_nodes_d = 50
minibatch_layer = 0
minibatch_normal_init_ = True
layers = 2
D_rounds = 1
G_rounds = 3

num_epoch = 36
learning_rate = 0.0002
tanh_layer = False

#Looping over possible number of minibatch outputs
minibatch_out = [0,3,5,8,10]
for minibatch_layer in minibatch_out:
  path = "./Run_"+str(minibatch_layer)
  os.mkdir(path)

  dict = {'data' : source_filename, 
        'sample_size' : sample_size, 
        'seq_length' : seq_length,
        'num_layers': layers, 
        'hidden_dims_generator': hidden_nodes_g, 
        'hidden_dims_discriminator':hidden_nodes_d,
        'D_rounds': D_rounds,
        'G_rounds': G_rounds,
        'num_epoch':num_epoch,
        'learning_rate' : learning_rate,
        'tanh_layer': tanh_layer,
        'minibatch_layer': minibatch_layer,
        'minibatch_normal_init_':minibatch_normal_init_}

  json = js.dumps(dict)
  f = open(path+"/settings.json","w")
  f.write(json)
  f.close()
  
  #Defining the generator and discriminator
  generator = Generator(seq_length,sample_size,hidden_dim = hidden_nodes_g, num_layers = layers, tanh_output = tanh_layer).cuda()
  discriminator = Discriminator(seq_length, sample_size,minibatch_normal_init = minibatch_normal_init_, hidden_dim = hidden_nodes_d, num_layers = layers, minibatch = minibatch_layer).cuda()
  d_optimizer = torch.optim.Adam(discriminator.parameters(),lr = learning_rate)
  g_optimizer = torch.optim.Adam(generator.parameters(),lr = learning_rate)
  #Loss function 
  loss = torch.nn.BCELoss()

  generator.train()
  discriminator.train()

  G_losses = []
  D_losses = []
  mmd_list = []
  series_list = np.zeros((1,seq_length))

  for n in tqdm(range(num_epoch)):
     # for k in range(1):

      for n_batch, sample_data in enumerate(data_loader):

        for d in range(D_rounds):
      ### TRAIN DISCRIMINATOR ON FAKE DATA
          discriminator.zero_grad()

          h_d = discriminator.init_hidden()
          h_g = generator.init_hidden()

          #Generating the noise and label data
          noise_sample = Variable(noise(len(sample_data),seq_length))

          #Use this line if generator outputs hidden states: dis_fake_data, (h_g_n,c_g_n) = generator.forward(noise_sample,h_g)
          dis_fake_data = generator.forward(noise_sample,h_g).detach()

          y_pred_fake, (h_d_n,c_d_n) = discriminator(dis_fake_data,h_d)

          loss_fake = loss(y_pred_fake,torch.zeros([len(sample_data),1]).cuda())
          loss_fake.backward()    

          #Train discriminator on real data   
          h_d = discriminator.init_hidden()
          real_data = Variable(sample_data.float()).cuda()    
          y_pred_real,(h_d_n,c_d_n)= discriminator.forward(real_data,h_d)

          loss_real = loss(y_pred_real,torch.ones([len(sample_data),1]).cuda())
          loss_real.backward()

          d_optimizer.step() #Updating the weights based on the predictions for both real and fake calculations.


        for g in range(G_rounds):
        #Train Generator  
          generator.zero_grad()
          h_g = generator.init_hidden()
          h_d = discriminator.init_hidden()
          noise_sample = Variable(noise(len(sample_data), seq_length))


          #Use this line if generator outputs hidden states: gen_fake_data, (h_g_n,c_g_n) = generator.forward(noise_sample,h_g)
          gen_fake_data = generator.forward(noise_sample,h_g)
          y_pred_gen, (h_d_n,c_d_n)= discriminator(gen_fake_data,h_d)

          error_gen = loss(y_pred_gen,torch.ones([len(sample_data),1]).cuda())
          error_gen.backward()
          g_optimizer.step()





       

        if n_batch == num_batches - 1:
          G_losses.append(error_gen.item())
          D_losses.append((loss_real+loss_fake).item())

          

          torch.save(generator.state_dict(), path+'/generator_state_'+str(n)+'.pt')
          torch.save(discriminator.state_dict(),path+ '/discriminator_state_'+str(n)+'.pt')

          
          # Check how the generator is doing by saving G's output on fixed_noise
        
          with torch.no_grad():
              h_g = generator.init_hidden()
              fake = generator(noise(len(sample_data), seq_length),h_g).detach().cpu()
              generated_sample = torch.zeros(1,seq_length).cuda()
              
              
              for test_batch in range(0,int(len(data_test)/sample_size)):
                noise_sample_test = noise(sample_size, seq_length)
                h_g = generator.init_hidden()
                generated_data = generator.forward(noise_sample_test,h_g).detach().squeeze()
                generated_sample = torch.cat((generated_sample,generated_data),dim = 0)


              # Getting the MMD Statistic for each Training Epoch
              generated_sample = generated_sample[1:][:]
              sigma = [pairwisedistances(data_test[:].type(torch.DoubleTensor),generated_sample.type(torch.DoubleTensor).squeeze())] 
              mmd = MMDStatistic(len(data_test[:]),generated_sample.size(0))
              mmd_eval = mmd(data_test[:].type(torch.DoubleTensor),generated_sample.type(torch.DoubleTensor).squeeze(),sigma, ret_matrix=False)
              mmd_list.append(mmd_eval.item())



              series_list = np.append(series_list,fake[0].numpy().reshape((1,seq_length)),axis=0)
 
  
  
  
  #Dumping the errors and mmd evaluations for each training epoch.
  with open(path+'/generator_losses.txt', 'wb') as fp:
      pickle.dump(G_losses, fp)
  with open(path+'/discriminator_losses.txt', 'wb') as fp:
      pickle.dump(D_losses, fp)   
  with open(path+'/mmd_list.txt', 'wb') as fp:
      pickle.dump(mmd_list, fp)
  
  #Plotting the error graph
  plt.plot(G_losses,'-r',label='Generator Error')
  plt.plot(D_losses, '-b', label = 'Discriminator Error')
  plt.title('GAN Errors in Training')
  plt.legend()
  plt.savefig(path+'/GAN_errors.png')
  plt.close()
  
  
  #Plot a figure for each training epoch with the MMD value in the title
  i = 0
  while i < num_epoch:
    if i%3==0:
      fig, ax = plt.subplots(3,1,constrained_layout=True)
      fig.suptitle("Generated fake data")
    for j in range(0,3):
      ax[j].plot(series_list[i][:])
      ax[j].set_title('Epoch '+str(i)+ ', MMD: %.4f' % (mmd_list[i]))
      i = i+1
    plt.savefig(path+'/Training_Epoch_Samples_MMD_'+str(i)+'.png')
    plt.close(fig)
    

  #Checking the diversity of the samples:
  generator.eval()
  h_g = generator.init_hidden()
  test_noise_sample = noise(sample_size, seq_length)
  gen_data= generator.forward(test_noise_sample,h_g).detach()


  plt.title("Generated ECG Waves")
  plt.plot(gen_data[random.randint(0,sample_size-1)].tolist(),'-b')
  plt.plot(gen_data[random.randint(0,sample_size-1)].tolist(),'-r')
  plt.plot(gen_data[random.randint(0,sample_size-1)].tolist(),'-g')
  plt.plot(gen_data[random.randint(0,sample_size-1)].tolist(),'-', color = 'orange')
  plt.savefig(path+'/Generated_Data_Sample1.png')
  plt.close()
    
