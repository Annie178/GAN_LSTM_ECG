import torch
import torch.nn as nn
from torch.autograd.variable import Variable

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# The aim of this module is to 

class MinibatchDiscrimination(torch.nn.Module):
   def __init__(self,input_features,output_features,minibatch_normal_init, hidden_features=16):
      super(MinibatchDiscrimination,self).__init__()
      
      self.input_features = input_features
      self.output_features = output_features
      self.hidden_features = hidden_features
      self.T = torch.nn.Parameter(torch.randn(self.input_features,self.output_features, self.hidden_features))
      if minibatch_normal_init == True:
        nn.init.normal(self.T, 0,1)
      
   def forward(self,x):
      M = torch.mm(x,self.T.view(self.input_features,-1))
      M = M.view(-1, self.output_features, self.hidden_features).unsqueeze(0)
      M_t = M.permute(1, 0, 2, 3)
      # Broadcasting reduces the matrix subtraction to the form desired in the paper
      out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
      
      return torch.cat([x, out], 1)
    
class Discriminator(torch.nn.Module):
  def __init__(self,seq_length,batch_size,minibatch_normal_init,minibatch = 0,n_features = 1, hidden_dim = 50, num_layers = 2 ):
      super(Discriminator,self).__init__()
      self.n_features = n_features
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.seq_length = seq_length
      self.batch_size = batch_size
      self.minibatch = minibatch
      
      self.layer1 = torch.nn.LSTM(input_size = self.n_features, hidden_size = self.hidden_dim, num_layers = self.num_layers
                                  ,batch_first = True#,dropout = 0.2
                                 )
      self.layer2 = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim,1),torch.nn.LeakyReLU(0.2),torch.nn.Dropout(0.2)) 
      #Adding a minibatch discriminator layer to add a cripple affect to the discriminator so that it needs to generate sequences that are different from each other.
      if self.minibatch > 0:
        self.mb1= MinibatchDiscrimination(self.seq_length,self.minibatch,minibatch_normal_init)
        self.out = torch.nn.Sequential(torch.nn.Linear(self.seq_length+self.minibatch,1),torch.nn.Sigmoid()) # to make sure the output is between 0 and 1
      else:
        self.out = torch.nn.Sequential(torch.nn.Linear(self.seq_length,1),torch.nn.Sigmoid()) # to make sure the output is between 0 and 1
 
  def init_hidden(self):
      # This line creates a new type as the first parameter tensor
      weight = next(self.parameters()).data
      #This line creates a new tensor of the same type as weight initailised to zero
      hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().cuda(), weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().cuda())
      return hidden
    
  def get_mb_params(self):
      try:
        params = self.mb1.get_parameters()
        return params
      except: 
        if self.minibatch == 0:
            print("Minibatch Discrimination was not configured")
        else:
          print("Issue with get_parameter function")
        return torch.ones((1,1))
      
  
  def forward(self,x,hidden):
      
      x,hidden = self.layer1(x.view(self.batch_size,self.seq_length,1),hidden)
      
      x = self.layer2(x.view(self.batch_size,self.seq_length,self.hidden_dim))
      if self.minibatch > 0:
        x = self.mb1(x.squeeze())
      x = self.out(x.squeeze())
      
      return x,hidden

class Generator(torch.nn.Module):
  def __init__(self,seq_length,batch_size,n_features = 1, hidden_dim = 50, num_layers = 2, tanh_output = False):
      super(Generator,self).__init__()
      self.n_features = n_features
      self.hidden_dim = hidden_dim
      self.num_layers = num_layers
      self.seq_length = seq_length
      self.batch_size = batch_size
      self.tanh_output = tanh_output
      

      
      self.layer1 = torch.nn.LSTM(input_size = self.n_features, hidden_size = self.hidden_dim, 
                                  num_layers = self.num_layers,batch_first = True#,dropout = 0.2,
                                 )
      if self.tanh_output == True:
        self.out = torch.nn.Sequential(torch.nn.Linear(self.hidden_dim,1),torch.nn.Tanh()) # to make sure the output is between 0 and 1 - removed ,torch.nn.Sigmoid()
      else:
        self.out = torch.nn.Linear(self.hidden_dim,1) 
      
  def init_hidden(self):
      weight = next(self.parameters()).data
      hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().cuda(), weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_().cuda())
      return hidden
  
  def forward(self,x,hidden):
      
      x,hidden = self.layer1(x.view(self.batch_size,self.seq_length,1),hidden)
      
      x = self.out(x)
      
      return x #,hidden 

def noise(batch_size, features):
  noise_vec = torch.randn(batch_size, features).cuda()
  return noise_vec
