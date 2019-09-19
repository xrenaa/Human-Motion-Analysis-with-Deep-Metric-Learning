import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from torch import Tensor
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from human_motion_analysis_with_gru import MMD_NCA_Net

num_epochs = 50000
learning_rate = 0.0001

use_cuda = torch.cuda.is_available()
    
class MMD_NCA_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def kernel_function(self, x1, x2):
        k1 = torch.exp(-torch.pow((x1 - x2), 2) / 2)
        k2 = torch.exp(-torch.pow((x1 - x2), 2) / 8)
        k4 = torch.exp(-torch.pow((x1 - x2), 2) / 32)
        k8 = torch.exp(-torch.pow((x1 - x2), 2) / 128)
        k16 = torch.exp(-torch.pow((x1 - x2), 2) / 512)
        k_sum = k1+k2+k4+k8+k16
        return k_sum
    
    def MMD(self, x, x_IID, y, y_IID):
        # x, x_IID's dimension: m*25,  y, y_IID's dimension: n*25
        m = x.size()[0]
        n = y.size()[0]
        x = x.view(m, 1, -1)
        # print(x.shape)
        x_square = x.repeat(1, m, 1)
        # print(x_square)
        # x_IID = x_IID.view(1, -1, m)
        x_IID = x_IID.view(-1, m, 1)
        # print(x_IID.shape)
        x_IID_square = x_IID.repeat(m, 1, 1)
        # print(x_IID_square.shape)
        value_1 = torch.sum(self.kernel_function(x_square, x_IID_square)) / (m**2)
        y = y.view(1, n, -1)
        y_square = y.repeat(n, 1, 1)
        value_2 = torch.sum(self.kernel_function(x_square, y_square)) / (m*n)
        y_IID = y_IID.view(n, 1, -1)
        y_IID_square = y_IID.repeat(1, n, 1)
        value_3 = torch.sum(self.kernel_function(y_IID_square, y_square)) / (n**2)
        return value_1 - 2*value_2 + value_3
    
    def forward(self, x):
        # print(x[0].shape)
        x = x.view(7, 25)
        #print(x[0], x[1])
        # numerator = torch.exp(-self.MMD(x[0], x[1], x[2], x[3]))
        numerator = torch.exp(-self.MMD(x[0], x[0], x[1], x[1]))
        # numerator = torch.exp(self.MMD(x[0], x[1], x[2], x[3]))
        # print(self.MMD(x[0], x[1], x[2], x[3]))
        # calculate the denominator in MMD NCA loss, only use 3 negative catogories
#         value_1 = torch.exp(-self.MMD(x[0], x[1], x[5], x[6]))
#         value_2 = torch.exp(-self.MMD(x[0], x[2], x[7], x[8]))
#         value_3 = torch.exp(-self.MMD(x[0], x[1], x[9], x[10]))
        value_1 = torch.exp(-self.MMD(x[0], x[0], x[2], x[2]))
        value_2 = torch.exp(-self.MMD(x[0], x[0], x[3], x[3]))
        value_3 = torch.exp(-self.MMD(x[0], x[0], x[4], x[4]))
        value_4 = torch.exp(-self.MMD(x[0], x[0], x[5], x[5]))
        value_5 = torch.exp(-self.MMD(x[0], x[0], x[6], x[6]))
#         value_1 = torch.exp(self.MMD(x[0], x[1], x[5], x[6]))
#         value_2 = torch.exp(self.MMD(x[0], x[2], x[7], x[8]))
#         value_3 = torch.exp(self.MMD(x[0], x[1], x[9], x[10]))
        # print(value_1, value_2, value_3)
        denominator = value_1 + value_2 + value_3 + value_4 + value_5
        # print(numerator, denominator)
        loss = torch.exp(- numerator / denominator)
#        return numerator / denominator
        return loss

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj) 
    
def save_to_json(dic,target_dir):
    dumped = json.dumps(dic, cls=NumpyEncoder)  
    file = open(target_dir, 'w')  
    json.dump(dumped, file)
    file.close()
    
def read_from_json(target_dir):
    f = open(target_dir,'r')
    data = json.load(f)
    data = json.loads(data)
    f.close()
    return data

class MMD_NCA_Dataset(Dataset):
    def __init__(self, json_name, num_MMD_NCA_Groups):
        # datafile
        self.df = read_from_json(json_name)
        for key in self.df:
            self.df[key] = np.asarray(self.df[key])
        self.num_MMD_NCA_Groups = num_MMD_NCA_Groups
        self.training_MMD_NCA_Groups = self.generate_MMD_NCA_Dataset(self.df, self.num_MMD_NCA_Groups)
    
    @staticmethod
    def generate_MMD_NCA_Dataset(df, num_MMD_NCA_Groups):
        
        MMD_NCA_Groups = []
        classes     = []
        for key in df:
            classes.append(key)
        # face_classes = make_dictionary_for_face_class(df)
        
        for _ in range(num_MMD_NCA_Groups):
            pos_class = np.random.choice(classes)
            neg_class_1 = np.random.choice(classes)
            neg_class_2 = np.random.choice(classes)
            neg_class_3 = np.random.choice(classes)
            neg_class_4 = np.random.choice(classes)
            neg_class_5 = np.random.choice(classes)
            while len(df[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while (neg_class_1 == pos_class or len(df[neg_class_1]) < 2):
                neg_class_1 = np.random.choice(classes)
            while ((neg_class_2 == pos_class) or (neg_class_2 == neg_class_1) or (len(df[neg_class_2]) < 2)):
                neg_class_2 = np.random.choice(classes)
            while ((neg_class_3 == pos_class) or (neg_class_3 == neg_class_1) or (neg_class_3 == neg_class_2) or (len(df[neg_class_3]) < 2)):
                neg_class_3 = np.random.choice(classes)
            while ((neg_class_4 == pos_class) or (neg_class_4 == neg_class_1) or (neg_class_4 == neg_class_2) or (neg_class_4 == neg_class_3) or (len(df[neg_class_4]) < 2)):
                neg_class_4 = np.random.choice(classes)
            while ((neg_class_5 == pos_class) or (neg_class_5 == neg_class_1) or (neg_class_5 == neg_class_2) or (neg_class_5 == neg_class_3) or (neg_class_5 == neg_class_4) or (len(df[neg_class_5]) < 2)):
                neg_class_5 = np.random.choice(classes)
                
            # select two positive samples
#             pos_sample_1 = df[pos_class][np.random.choice(df[pos_class].shape[0])]
#             pos_sample_2 = df[pos_class][np.random.choice(df[pos_class].shape[0])]
#             while (pos_sample_2 == pos_sample_1).all():
#                 pos_sample_2 = df[pos_class][np.random.choice(df[pos_class].shape[0])]
            arr = np.arange(df[pos_class].shape[0])
            np.random.shuffle(arr)
            for i in range(25):
                if i == 0:
                    MMD_NCA_Group = df[pos_class][arr[i]]
                else:
                    MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[pos_class][arr[i]]), axis = 0)
            
            # select two anchor positive samples
#             pos_anchor_sample_1 = df[pos_class][np.random.choice(df[pos_class].shape[0])]
#             while ((pos_anchor_sample_1 == pos_sample_1).all() or (pos_anchor_sample_1 == pos_sample_2).all()):
#                 pos_anchor_sample_1 = df[pos_class][np.random.choice(df[pos_class].shape[0])]
#             pos_anchor_sample_2 = df[pos_class][np.random.choice(df[pos_class].shape[0])]
#             while ((pos_anchor_sample_2 == pos_sample_1).all() or (pos_anchor_sample_2 == pos_sample_2).all() or (pos_anchor_sample_2 == pos_anchor_sample_1).all()):
#                 pos_anchor_sample_2 = df[pos_class][np.random.choice(df[pos_class].shape[0])]
            arr = np.arange(df[pos_class].shape[0])
            np.random.shuffle(arr)
            for i in range(25):
                MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[pos_class][arr[i]]), axis = 0)
                
            # select two negative 1 samples
#             neg_1_sample_1 = df[neg_class_1][np.random.choice(df[neg_class_1].shape[0])]
#             neg_1_sample_2 = df[neg_class_1][np.random.choice(df[neg_class_1].shape[0])]
#             while (neg_1_sample_2 == neg_1_sample_1).all():
#                 neg_1_sample_2 = df[neg_class_1][np.random.choice(df[neg_class_1].shape[0])]
            arr = np.arange(df[neg_class_1].shape[0])
            np.random.shuffle(arr)
            for i in range(25):
                MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[neg_class_1][arr[i]]), axis = 0)
    
            # select two negative 2 samples
#             neg_2_sample_1 = df[neg_class_2][np.random.choice(df[neg_class_2].shape[0])]
#             neg_2_sample_2 = df[neg_class_2][np.random.choice(df[neg_class_2].shape[0])]
#             while (neg_2_sample_2 == neg_2_sample_1).all():
#                 neg_2_sample_2 = df[neg_class_2][np.random.choice(df[neg_class_2].shape[0])]
            arr = np.arange(df[neg_class_2].shape[0])
            np.random.shuffle(arr)
            for i in range(25):
                MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[neg_class_2][arr[i]]), axis = 0)
    
            # select two negative 3 samples
#             neg_3_sample_1 = df[neg_class_3][np.random.choice(df[neg_class_3].shape[0])]
#             neg_3_sample_2 = df[neg_class_3][np.random.choice(df[neg_class_3].shape[0])]
#             while (neg_3_sample_2 == neg_3_sample_1).all():
#                 neg_3_sample_2 = df[neg_class_3][np.random.choice(df[neg_class_3].shape[0])]
            arr = np.arange(df[neg_class_3].shape[0])
            np.random.shuffle(arr)
            for i in range(25):
                MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[neg_class_3][arr[i]]), axis = 0)
            
            arr = np.arange(df[neg_class_4].shape[0])
            np.random.shuffle(arr)
            for i in range(25):
                MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[neg_class_4][arr[i]]), axis = 0)
                
            arr = np.arange(df[neg_class_5].shape[0])
            np.random.shuffle(arr)
            for i in range(25):
                MMD_NCA_Group = np.concatenate((MMD_NCA_Group, df[neg_class_5][arr[i]]), axis = 0)
            
#             MMD_NCA_Group = np.concatenate((pos_sample_1, pos_sample_2, pos_anchor_sample_1, pos_anchor_sample_2, \
#                                           neg_1_sample_1, neg_1_sample_2, neg_2_sample_1, neg_2_sample_2, neg_3_sample_1, neg_3_sample_2), axis=0)
            MMD_NCA_Groups.append(MMD_NCA_Group)
            
        return MMD_NCA_Groups
        
    def __getitem__(self, index):
        # key stands for dictionary key, index stands for index for one group of MMD_NCA_Dataset
        return self.training_MMD_NCA_Groups[index]
        
    def __len__(self):
        return len(self.training_MMD_NCA_Groups)

def train(model, train_loader, myloss, optimizer, epoch):
    model.train()
    for batch_idx, train_data in enumerate(train_loader):
        train_data = Variable(train_data).type(torch.cuda.DoubleTensor).squeeze().view(175,50,34).permute(1,0,2)
        optimizer.zero_grad()
        output = model(train_data)
        # loss = myloss(output, train_data)
        loss = myloss(output)
        loss.backward()
        optimizer.step()
        if batch_idx%100 == 0:
             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, batch_idx*len(train_data), len(train_loader.dataset),
                 100.*batch_idx/len(train_loader), 10000.*loss.data.cpu().numpy()))
        return loss

def save_models(epoch):
    torch.save(model.state_dict(), "./log/model_new_{}.pth".format(epoch))

model = MMD_NCA_Net().cuda().double()
criterion = MMD_NCA_loss()
#generate training data
train_data = MMD_NCA_Dataset('./dataset/GIT_zizi.json', 30000)
train_loader = DataLoader(train_data, batch_size = 1, shuffle = True)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

loss_total = 0.
for epoch in range(num_epochs):
    loss = train(model, train_loader, criterion, optimizer, epoch)
    loss_total += loss.data.cpu().numpy()
    if (epoch+1)%2000 == 0:
        print('loss mean after {} epochs: {}'.format((epoch+1), loss_total / 2000))
        loss_total = 0.
    if (epoch+1)%5000 == 0:
        save_models(epoch)

# save_models(num_epochs)

