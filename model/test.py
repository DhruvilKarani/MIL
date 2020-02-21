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
from preprocess import Patch
import os
import logging
from torch.utils.tensorboard import SummaryWriter
sys.path.append('../trained_models')
from train import Attention, Classifier



if __name__ == '__main__':
    logging.basicConfig(filename='../logs/weights_logfile.log',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.INFO)


    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"


    assert device == "cuda", "CUDA not available"

    PATCH_DIM = 620
    BAG_SZIE = (3100/PATCH_DIM)**2
    HIDDEN_DIM = 1000
    V_DIM = 500
    NUM_CLASSES = 4
    LOAD_SAVED = False
    PRE_TRAINED = True


    PATH = '../trained_models'
    PATCH_MODEL = '49_patch_model.pth'
    ATTN_MODEL = '49_attention_layer.pth'
    CLASS_MODEL = '49_classifier.pth'


    patch_model = pretrained_models.resnet18(pretrained = True)
    patch_model.fc = nn.Linear(512,HIDDEN_DIM)
    patch_model = nn.DataParallel(patch_model)
    patch_model = patch_model.to(device)


    attn_model = Attention(HIDDEN_DIM, V_DIM)
    attn_model = attn_model.to(device).train()


    class_model = Classifier(HIDDEN_DIM,NUM_CLASSES)  
    class_model = class_model.to(device).train()


    patch_model.load_state_dict(torch.load(os.path.join(PATH, PATCH_MODEL)))
    attn_model.load_state_dict(torch.load(os.path.join(PATH, ATTN_MODEL)))
    class_model.load_state_dict(torch.load(os.path.join(PATH, CLASS_MODEL)))
    print("Model loaded.")

    PATH = '/home/chamelyon/Documents/Gautam/harvard_original_processed/all/'
    df = pd.read_csv('../data/unique.csv')
    names = list(df['name'])
    labels= list(df['label'])
    patch_obj = Patch(PATH, PATCH_DIM)
    loader = patch_obj.train_loader(names, labels)
    patches, label = next(loader)


    loader = patch_obj.train_loader(names, labels)
    y_true = []
    y_pred = []
    for i,(patches,label) in enumerate(loader): 
        patches = patches.to(device)
        label = torch.LongTensor([label]).to(device)
        patch_output = patch_model(patches)
        attention_output, weights = attn_model(patch_output)
        output = class_model(attention_output).view(-1,NUM_CLASSES)
        pred = torch.argmax(output)
        y_pred.append(pred.item())
        y_true.append(label.item())
        info = "NAME: {0}, MAX_WEIGHT: {1}, LOCATION: {2}".format(names[i], torch.max(weights).item(), torch.argmax(weights).item())
        logging.info(info)

    pred_dict = {'y_true':y_true,
                'y_pred':y_pred}

    df = pd.DataFrame(pred_dict)
    df.to_csv('../logs/predictions.csv')
