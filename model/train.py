import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision.models as pretrained_models
from torch.optim.lr_scheduler import StepLR
import numpy as np
# import matplotlib.pyplot as plt
import random
import sys
import pandas as pd
sys.path.append('../data')
from preprocess import Patch, Patchify
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
writer = SummaryWriter('../plots')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


loss_logger = setup_logger('loss_logger', '../logs/los_logfile.log')
weights_logger = setup_logger('weights_logger', '../logs/weights_logfile.log')
metric_logger = setup_logger('metric_logger', '../logs/metric_logfile.log')

def pretrained_agg(X, model):
    output = []
    for x_in in X:
        mod_output = torch.unsqueeze(model(x_in), 0)
        output.append(mod_output)
    return torch.cat(output)
    

class Attention(nn.Module):
    def __init__(self,hidden_dim,v_dim):
        super(Attention, self).__init__()
        self.V = nn.Linear(hidden_dim,v_dim,bias=False)
        torch.nn.init.xavier_normal_(self.V.weight)
        self.w = nn.Linear(v_dim,1,bias=False)
        torch.nn.init.xavier_normal_(self.w.weight)

    def transform(self, embedding):
        embedding = torch.tanh(self.V(embedding))
        embedding = self.w(embedding)
        return torch.exp(embedding)

    def _forward(self, embeddings, device='cpu'):
        weights = torch.zeros(len(embeddings)).to(device)
        norm_factor = 0
        attn_embedding = torch.zeros_like(embeddings[0]).to(device)
        for i,embedding in enumerate(embeddings):
            embedding = self.transform(embedding)
            norm_factor+=embedding
            weights[i] = embedding
        normalized_weights =  weights.view(-1)/norm_factor.item()
        for weight,embedding in zip(normalized_weights,embeddings):
            attn_embedding = torch.add(attn_embedding, weight*embedding)

        return attn_embedding, normalized_weights

    def forward(self, X, device = "cpu"):
        attn_embeddings = [] 
        weights = []
        for x_in in X:
            output, wts = self._forward(x_in, device)
            output = torch.unsqueeze(output, 0)
            wts = torch.unsqueeze(wts, 0)
            attn_embeddings.append(output)
            weights.append(wts)
        
        return torch.cat(attn_embeddings,0), torch.cat(weights,0)

class Classifier(nn.Module):
    def __init__(self,hidden_dim,num_classes):
        super(Classifier, self).__init__()
        self.classify_one = nn.Linear(hidden_dim,500)
        torch.nn.init.xavier_normal_(self.classify_one.weight)
        self.classify_two = nn.Linear(500, num_classes)
        torch.nn.init.xavier_normal_(self.classify_two.weight)
        self.relu = nn.LeakyReLU()
    def forward(self,attn_embedding):
        output = self.classify_one(attn_embedding)
        output = self.relu(output)
        output = self.classify_two(output)
        output = self.relu(output)
        return output


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)


def nonzero_grad(model):
    total = 0
    nonzero = 0
    for params in model.parameters():
        grad = torch.nonzero(params.grad.view(-1))
        params = params.view(-1)
        nonzero += grad.shape[0]
        total += params.shape[0]
    return nonzero/total


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    device = "cpu"

    PATCH_DIM = 620
    BAG_SZIE = (3100/PATCH_DIM)**2
    HIDDEN_DIM = 1000
    V_DIM = 500


    NUM_CLASSES = 4
    LOAD_SAVED = False
    PRE_TRAINED = True


    PATH = '/home/chamelyon/Documents/Gautam/harvard_original_processed/all/'
    df = pd.read_csv('../data/unique.csv')
    names = list(df['name'])
    labels= list(df['label'])

    #------------------Set up DataLoaders---------------#
    transform_scheme = transforms.Compose([
        transforms.ToTensor(),
        Patchify(PATCH_DIM)
    ])

    PatchDataset = Patch(PATH, PATCH_DIM, names, labels, transform = transform_scheme)
    PatchLoader = DataLoader(PatchDataset, batch_size=2, shuffle=True, num_workers=8)

    


    #---------------------Setup ResNet-------------------#
    patch_model = pretrained_models.resnet18(pretrained = True)
    patch_model.fc = nn.Linear(512,HIDDEN_DIM)
    # patch_model = nn.DataParallel(patch_model)
    # patch_model = patch_model


    #--------------------Set Up Attention---------------#
    attention_layer = Attention(HIDDEN_DIM, V_DIM)
    attention_layer = attention_layer.train()
  

    #--------------------Set up classifier---------------#
    classifier = Classifier(HIDDEN_DIM,NUM_CLASSES)
    classifier = classifier.train()






    if LOAD_SAVED:
        print("Loading saved model")
        patch_model.load_state_dict(torch.load('../trained_models/patch_model.pth'))
        attention_layer.load_state_dict(torch.load('../trained_models/attention_layer.pth'))
        classifier.load_state_dict(torch.load('../trained_models/classifier.pth'))

    loss_function = nn.CrossEntropyLoss()
    patch_optim = optim.Adam(patch_model.parameters(), lr=0.00001)
    # patch_scheduler = StepLR(patch_optim, step_size=1, gamma=0.1)
    attn_optim = optim.Adam(attention_layer.parameters(), lr = 0.00001)
    # attn_scheduler = StepLR(attn_optim, step_size=1, gamma=0.1)
    class_optim = optim.Adam(classifier.parameters(), lr = 0.00001)
    # class_scheduler = StepLR(class_optim, step_size=1, gamma=0.1)

    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    weight_log = []
    loss_log = []
    avg_loss_log = []
    ids_log = []
    bags_log = []
    labels_log = []

    torch.cuda.empty_cache()

#------------------------------------------------- Training Loop -------------------------------------------------#
    loader = patch_obj.train_loader(names, labels)
    cum_loss = torch.zeros(1).to(device)
    epoch_loss_log = []
    for j in range(NUM_EPOCHS):
        epoch_loss = torch.zeros(1).to(device)
        patch_model.train()
        attention_layer.train()
        classifier.train()
        loader = patch_obj.train_loader(names, labels)
        for i, (whole_img, label) in enumerate(PatchLoader):
            output = pretrained_agg(whole_img, patch_model)
            output, weights = attention_layer(output)
            output = classifier(output).view(-1,NUM_CLASSES)
            loss = loss_function(output,label)
            cum_loss+=loss
            epoch_loss += loss
            patches.to('cpu')
            torch.cuda.empty_cache()

            if i%BATCH_SIZE == 0:
                cum_loss.backward()
                loss_log.append(cum_loss.item())
                patch_optim.step()
                attn_optim.step()
                class_optim.step()
                weight_log.append(weights)
                labels_log.append(label.item())
                print("--------Epoch: {0}--------".format(j))
                print("--------{0}/{1}--------".format(i+1, len(labels)))
                print("Max Weight:",torch.max(weights))
                print("Max Weight Index:",torch.argmax(weights))
                print("Label:",label.item())
                loss_log = []
                cum_loss = torch.zeros(1).to(device)

        loss_logger.info("Epoch: {0}, Loss: {1}".format(j, epoch_loss.item()))
        writer.add_scalar('Epoch Loss', epoch_loss.item(), j)


    MODEL_PATH = '../trained_models'
    torch.save(patch_model.state_dict(), os.path.join(MODEL_PATH,str(j)+'_'+'patch_model.pth'))
    torch.save(attention_layer.state_dict(), os.path.join(MODEL_PATH,str(j)+'_'+'attention_layer.pth'))
    torch.save(classifier.state_dict(), os.path.join(MODEL_PATH,str(j)+'_'+'classifier.pth'))