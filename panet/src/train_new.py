from select import select
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose
from models.fewshotv3 import FewShotSegV3
from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
from datetime import datetime
#from torchinfo import summary
import random

USE_GPU = False

dtype = torch.float32 # we will be using float throughout this tutorial

#if USE_GPU and torch.cuda.is_available():
if USE_GPU:
    device = torch.device('mps')
    #set the device to gpu0
    #torch.cuda.set_device(device=0)
else:
    device = torch.device('cpu')

seed=1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
select_set=0
#setup the experiment
setup={}
input_size = (417, 417)
setup["seed"]=seed
setup["input dims"]=input_size
setup["steps"]=30000
setup["ways"]=1
setup["shots"]=1
setup["num_queries"]=1
setup["batch_size"]=1
setup["learning_rate"]=1e-3
setup["distfunc"]="euclidean"
setup["backbone"]="vgg"
setup["dataset"]="VOC"
setup["select_set"]=select_set
batch_size=setup['batch_size']

steps=setup['steps']
lambda_PAR=1
dataset_name=setup["dataset"]


#Get the dataset functions from here
if dataset_name == 'VOC':
    gen_dataset = voc_fewshot
    data_dir='../../data/Pascal/VOCdevkit/VOC2012/'
    data_split='trainaug'
elif dataset_name == 'COCO':
    gen_dataset = coco_fewshot
    data_dir='../../data/COCO/'
    data_split='train'
# else: 
#     dataset=cityscape_fewshot

labels = CLASS_LABELS[dataset_name][select_set]
labels_val = CLASS_LABELS[dataset_name]["all"]-labels
transforms = Compose([Resize(size=input_size),
                        RandomMirror()])
dataset = gen_dataset(
    base_dir=data_dir,
    split=data_split,
    transforms=transforms,
    to_tensor=ToTensorNormalize(),
    labels=labels,
    max_iters=setup['batch_size'] *steps,
    n_ways=setup['ways'],
    n_shots=setup['shots'],
    n_queries=setup['num_queries']
)

train_loader = DataLoader(
    dataset,
    batch_size=setup['batch_size'],
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    drop_last=True
)


dataset_val = gen_dataset(
    base_dir=data_dir,
    split=data_split,
    transforms=transforms,
    to_tensor=ToTensorNormalize(),
    labels=labels_val,
    max_iters=setup['batch_size'] *steps,
    n_ways=setup['ways'],
    n_shots=setup['shots'],
    n_queries=setup['num_queries']
)

val_loader = DataLoader(
    dataset,
    batch_size=setup['batch_size'],
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    drop_last=True
)


print_every=100
save_every=1000
def train_model(model, optimizer, scheduler ,epochs=1):

    train_loss={}
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    train_loss["loss_query"]=[]
    train_loss["loss_PAR"]=[]
    for e in range(epochs):
        lossq=0
        losspar=0
        for t,samples in enumerate(train_loader):
            
           #get a list of support images ( Sx W x H x W)
            support_img=[[shot for shot in way]
                          for way in samples['support_images']]
            fg_mask=[[shot[f'fg_mask'].float() for shot in way]
                           for way in samples['support_mask']]
            bg_mask=[[shot[f'bg_mask'].float() for shot in way]
                           for way in samples['support_mask']]
            query_img= [query
                        for query in samples['query_images']]

            query_gt = torch.cat([queryGT.long()
                        for queryGT in samples['query_labels']], dim=0)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            #get the predicted query and the Prototype alignment loss
            pred_query,lossPAR= model(support_img,fg_mask,bg_mask,query_img)

            lossQ= criterion(pred_query,query_gt)

            loss = lossQ+ lossPAR * lambda_PAR
            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            scheduler.step()

            lossq+=lossQ.detach().cpu().numpy()
            losspar+=lossPAR

            if (t+1) % print_every == 0:
                print('Epoch %d, Iteration %d, Seg loss = %.8f' % (e, t+1, (lossq/(t+1))))
                print('Epoch %d, Iteration %d, PAR loss = %.8f' % (e, t+1, (losspar/(t+1))))
                train_loss["loss_query"].append(lossq/(t+1))
                train_loss["loss_PAR"].append((losspar.detach().cpu().numpy()/(t+1)))
            if (t+1) % save_every == 0:
                torch.save(model.state_dict(),model_path)    



    return (train_loss)

#cfg True means align is on
milestones= [steps//3,steps//2,steps]

#if backbone is set to dv3 then select dv3 model else use vgg model
if(setup['backbone']=="dv3"):
    model = FewShotSegV3(cfg={'align': True},distfunc=setup["distfunc"])
else:
    model = FewShotSeg(cfg={'align': True},distfunc=setup["distfunc"])
optimizer = torch.optim.SGD(model.parameters(),lr=setup["learning_rate"],momentum=0.9,weight_decay=0.00005)
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
criterion = nn.CrossEntropyLoss(ignore_index=255)


if __name__ == '__main__':
    info=str(model)
    now = datetime.now()
    timestamp= now.strftime("%d%m%H%M%S")
    filename="./test_models/info/cfg_"+timestamp+".json"
    model_path="./test_models/model/fs_"+timestamp+".pth"

    setup["desctiption"]="Experiment1: fewshot with vgg encoder with euclidean dist"
    #setup["model_summary"]=info
    setup["optimizer"]=str(optimizer)
    setup["epochs"]=1
    setup["model path"]=model_path

    json_directory = json.dumps(setup,indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_directory)

    #update file with loss array
    train_loss=train_model(model, optimizer, scheduler ,epochs=1)
    torch.save(model.state_dict(),model_path)
    setup["losses"]=train_loss
    json_directory = json.dumps(setup,indent=4)
    with open(filename, "w") as outfile:
        outfile.write(json_directory)
