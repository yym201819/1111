from torch.utils.data import DataLoader,Dataset
import numpy as np
import scipy.io as sio

def load_data(filename, split):
    if len(filename)==2:
        d1 = sio.loadmat(filename[0])['p_map']  # sio.loadmat用来加载mat文件
        d2 = sio.loadmat(filename[1])['d_map']
        data = np.concatenate((d1[:,:,:,np.newaxis], d2[:,:,:,np.newaxis]), axis=3)
        data = np.array(data, dtype=np.float32)
    train = data[0:split[0],:,:,:]
    validate = data[split[0]:split[0]+split[1],:,:,:]
    test = data[split[0]+split[1]:split[0]+split[1]+split[2],:,:,:]
    return data, train, validate, test
# split = [17520, 4416, 4368]
# data, train_data, val_data, test_data = load_npy_data(
#         filename=["./data/citybike/p_map.npy","./data/citybike/d_map.npy"], split = split)
split = [43824, 8760, 8760]
data,train_data,val_data,test_data = load_data(
    filename=['./data/taxi/p_map.mat','./data/taxi/d_map.mat'],split=split
)

print(data.shape,train_data.shape,val_data.shape,test_data.shape)

class MinMaxNorm(object):
    def __init__(self):
        pass

    def fit(self,data):
        self.min = np.amin(data)
        self.max = np.amax(data)
        print("min:",self.min,"max:",self.max)

    def transform(self,data):
        norm_data = 1.*(data -self.min)/(self.max-self.min)
        return norm_data

    def fit_transform(self,data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self,data):
        inverse_norm_data = 1.*data*(self.max-self.min)+self.min
        return inverse_norm_data

    def real_loss(self,loss):
        # loss is rmse
        return loss*(self.max-self.min)

normlization = MinMaxNorm()
normlization.fit(data=train_data)
train_data = normlization.transform(train_data)
val_data = normlization.transform(val_data)
test_data = normlization.transform(test_data)

class MyDataset(Dataset):

    def __init__(self,Data,input_steps=10,output_steps=1):
        self.Data = Data
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.all_seqlen = self.input_steps + self.output_steps
        self.train_x = []
        self.train_y = []


        for i in range(len(self.Data)-self.input_steps):
            self.train_x.append(self.Data[i:i+self.input_steps])
            self.train_y.append(self.Data[i+self.input_steps:i+self.all_seqlen])

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self,index):
        return self.train_x[index],self.train_y[index]

# print(train_data.shape)    #(43824, 64, 64, 2)
train_data = np.transpose(train_data,(0,3,2,1))
print(train_data.shape)
train_data = MyDataset(Data=train_data,input_steps=10,output_steps=1)
print(len(train_data))
train_dataloader = DataLoader(dataset=train_data,batch_size=16,shuffle=True,drop_last=True)


val_data = np.transpose(val_data,(0,3,2,1))
print(val_data.shape)
val_data = MyDataset(Data=val_data,input_steps=10,output_steps=1)
val_dataloader = DataLoader(dataset=val_data,batch_size=16,shuffle=False,drop_last=True)
print(len(val_dataloader))
test_data = np.transpose(test_data,(0,3,2,1))
test_data = MyDataset(Data=test_data,input_steps=10,output_steps=1)
test_dataloader = DataLoader(dataset=test_data,batch_size=16,shuffle=False,drop_last=True)


import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

batch_size = 16


def attention_2(hidden_list, last_output, batch_size):
    output_flatten = torch.reshape(last_output, (batch_size, -1))  # torch.Size([16, 16384])
    # print(output_flatten.shape)
    h0_h9_flatten = torch.reshape(hidden_list, (batch_size, 9, -1))  # torch.Size([16, 9, 16384])
    # print(h0_h9_flatten.shape)
    output_flatten = output_flatten.unsqueeze(dim=2)
    # print(output_flatten.shape)  # 16,16384,1

    att_weight = torch.bmm(h0_h9_flatten, output_flatten)
    # print(att_weight.shape)   # 16,9,1

    softmax_weight = torch.nn.functional.softmax(att_weight.squeeze(), dim=1)
    # print(softmax_weight.shape)  # 16,9

    context = torch.bmm(h0_h9_flatten.transpose(1, 2), softmax_weight.unsqueeze(dim=2))
    # print(context.shape)  # torch.Size([16, 16384, 1])

    context = context.squeeze()
    # print(context.shape)   # torch.Size([16, 16384])

    real_output = torch.reshape(context, (batch_size, 64, 16, -1))
    # print(real_output.shape)  # torch.Size([16, 64, 16, 16])

    return real_output, softmax_weight


import torch
from torch import nn
from CONVLSTM import ConvLSTM
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()  # 输入（16，10，2,64，64）--->(160,2,64,64)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels=2,out_channels=8,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(8),     # (160,8,32,32)
            nn.ReLU(),
            nn.Conv2d(8,16,3,2,1),    # (160,16,16,16)
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
                                 # --->(16,10,16,16,16)
        self.convlstm = ConvLSTM(input_dim=16,hidden_dim=64,kernel_size=(3,3),num_layers=2,
                                 batch_first=True,bias=True,return_all_layers=True)
        # self.convlstmcell1 = ConvLSTMCell(input_dim=16,hidden_dim=64,kernel_size=(3,3),bias=True)
        # self.convlstmcell2 = ConvLSTMCell(input_dim=64,hidden_dim=64,kernel_size=(3,3),bias=True)


        self.deconv = nn.Sequential(nn.ConvTranspose2d(in_channels=64,out_channels=8,
            kernel_size=3,stride=2,padding=1,output_padding=1),
                                    # nn.BatchNorm2d(8),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(8,2,3,2,1,1),
                                    # nn.BatchNorm2d(2),
                                    nn.ReLU())
    def forward(self, x):
        b = []
# output.cuda().data.cpu().numpy()
        for i in range(len(x)):
            a = x[i,:]
            # print(a.shape)
            a1 = self.conv_relu(a)
            a1 = a1.cuda().data.cpu().numpy()
            # a1 = a1.detach().numpy()
            # print(a1.shape)
            b.append(a1)
        x = np.array(b)
        x = torch.from_numpy(x)
        # print(x.shape)
        #  shape(16,10,16,16,16)
        layer_output,last_state = self.convlstm(x)


        hidden_list = layer_output[1][:,:-1,:,:]
        # print(hidden_list.shape)  # 16,9,64,16,16
        last_output = layer_output[1][:,-1,:,:,:]
        # print(last_output.shape)  # 16,64,16,16
        output,_ = attention_2(hidden_list=hidden_list,last_output=last_output,batch_size=batch_size)
        # print(output.shape)



        W1 = torch.zeros((1))
        W1 = torch.nn.init.constant_(W1,0.5)
        W2 = torch.zeros((1))
        W2 = torch.nn.init.constant_(W2,0.5)
        W1 = W1.cuda()
        W2 = W2.cuda()
        final_output = W1*output+W2*last_output


        output = self.deconv(final_output)   # 16,2,64,64
        output = output.unsqueeze(dim=1)
        return output


def save():
    model = Model()
    # print(model)
    # for p in model.parameters():
    #     torch.nn.init.xavier_normal(p,0,1)
    def weights_init(m):
        for p in model.parameters():
            classname=p.__class__.__name__
            if classname.find('Conv') != -1:
                torch.nn.init.xavier_normal_(p.weight.data,gain=0.5)
                torch.nn.init.xavier_normal_(p.bias.data,gain=0.5)

    model.apply(weights_init)
    # model.load_state_dict(torch.load('another3-epoch'+str(200)+'model_param.pkl'))

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0002)
    loss_func = nn.MSELoss(reduction='sum')
    model = model.cuda()


    for epoch in range(300):
        train_loss = 0
        # model.train()
        for i,(batch_x,batch_y) in enumerate(train_dataloader):
            # print(batch_x.shape,batch_y.shape)
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            out = model(batch_x)
            # print(out.shape,batch_y.shape)
            loss = loss_func(out,batch_y)
            # print('loss',loss.item())
            # loss = np.sqrt(np.array(loss.item()))
            # real_loss = normlization.real_loss(np.sqrt(np.array(loss.item())))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if i%100==0:
            #     rmse = np.sqrt(np.array(loss.item()))
            #     real_rmse = normlization.real_loss(rmse)
            #     print("loss:",loss.item(),"rmse:",rmse,"real_rmse:",real_rmse)
            # print('batch loss',normlization.real_loss(np.sqrt(np.array(loss.item()))))
            # train_loss+=real_loss
            # train_loss+=normlization.real_loss(np.sqrt(np.array(loss.item())))
            # print(batch_y.shape)
            # batch_loss = np.sqrt(loss.item()/np.prod(batch_y.shape))
            # print("batch loss:",batch_loss,"real",normlization.real_loss(batch_loss))
            train_loss+=loss.item()
        # print(train_loss)
        M = np.prod(batch_y.shape)*len(train_dataloader)
        rmse=np.sqrt(train_loss/np.prod([16.,1.,64.,64.,2.,2738.]))
        # print('epoch:',epoch,"train_loss:",train_loss,"rmse",rmse,"real_rmse",normlization.real_loss(rmse))

        # model.eval()
        val_loss = 0
        for i,(x,y) in enumerate(val_dataloader):
            x = x.cuda()
            y = y.cuda()
            # print(y.shape)
            output = model(x)
            loss = loss_func(output,y)
            val_loss+=loss.item()
        test_rmse = np.sqrt(val_loss/np.prod([16., 1., 64., 64., 2., 546.]))
        print('epoch:', (epoch+1),"train_real_rmse",normlization.real_loss(rmse),"test_real_rmse", normlization.real_loss(val_rmse))

        torch.save(model.state_dict(), 'another5-epoch'+str(epoch+1)+'model_param.pkl')


# save()

import os

def restore_param():
    model = Model()

    for epoch in range(298,299,1):
        model.load_state_dict(torch.load('another3-epoch'+str(epoch+1)+'model_param.pkl'))

        loss_func = nn.MSELoss(reduction='sum')

        model = model.cuda()

        # train_loss = 0
        # for i, (x, y) in enumerate(train_dataloader):
        #     x = x.cuda()
        #     y = y.cuda()
        #     out = model(x)
        #     loss = loss_func(out, y)
        #     train_loss += loss.item()
        # rmse = np.sqrt(train_loss / np.prod([16., 1., 64., 64., 2., 2738.]))
        #
        # # model.eval()
        # val_loss = 0
        # for i, (x, y) in enumerate(val_dataloader):
        #     x = x.cuda()
        #     y = y.cuda()
        #     # print(y.shape)
        #     output = model(x)
        #     loss = loss_func(output, y)
        #     val_loss += loss.item()
        # val_rmse = np.sqrt(val_loss / np.prod([16., 1., 64., 64., 2., 546.]))

        test_loss=0
        prediction = []
        target = []
        for i,(test_batch_x,test_batch_y) in enumerate(test_dataloader):
            test_batch_x = test_batch_x.cuda()
            test_batch_y = test_batch_y.cuda()
            output = model(test_batch_x)
            loss = loss_func(output,test_batch_y)
            test_loss+=loss.item()

            prediction.append(output.cpu().detach().numpy())
            target.append(test_batch_y.cpu().detach().numpy())
        test_rmse = np.sqrt(test_loss/(np.prod(test_batch_y.shape)*len(test_dataloader)))

        if not os.path.exists('testtdaconvlstmbn'):
            os.mkdir('testtdaconvlstmbn')
        np.save('testtdaconvlstmbn/prediction',prediction)
        np.save('testtdaconvlstmbn/target',target)
        print('epoch',(epoch+1), 'test_rmse',test_rmse,normlization.real_loss(test_rmse))

restore_param()
