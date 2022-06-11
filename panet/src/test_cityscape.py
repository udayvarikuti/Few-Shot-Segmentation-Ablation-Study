from soupsieve import select
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose
from models.fewshotv3 import FewShotSegV3
from models.fewshot import FewShotSeg
from models.fewshotv3Inet import FewShotSegV3Inet
from models.fewshotrnet import FewShotSegResnet
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
import torch.optim as optim
import matplotlib.pyplot as plt
from util.metric import Metric
import cv2
import numpy as np
import random
from tqdm import tqdm
from dataloaders.cityscape import Cityscape
from torchvision.transforms import Compose
from torchvision import transforms

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    #set the device to gpu0
    torch.cuda.set_device(device=0)
else:
    device = torch.device('cpu')

import json 

seed=1234
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

id="1006054524"
dir= "./test_models/info/cfg_"+id+".json"
model_path="./test_models/model/fs_"+id+".pth"
openfile=open(dir, "r")
setup= json.load(openfile)

print(setup)

input_size = setup["input dims"]
batch_size=setup["batch_size"]
distfunc=setup["distfunc"]
backbone=setup["backbone"]
ways=setup["ways"]
shots=setup["shots"]
nqueries=setup["num_queries"]
dataset_name=setup["dataset"]
steps=1000
n_runs=1
lambda_PAR=1
criterion = nn.CrossEntropyLoss(ignore_index=255)
select_set=0
#Get the dataset functions from here
if dataset_name == 'Cityscape':
        cityscapesPath = '../../data/Cityscape'
        dataset_size = 1000

        flip_transform = transforms.RandomHorizontalFlip(p=0.25)
        img_transforms = Compose([transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                            transforms.Resize(size=input_size),])
        # img_transforms = Compose([transforms.ToTensor(),
        #                       transforms.Resize(size=input_size),])
                            #flip_transform])
        mask_transforms = Compose([transforms.ToTensor(),
                            transforms.Resize(size=input_size),])
                            #flip_transform])

        dataset = Cityscape(cityscapesPath, dataset_size, labels=CLASS_LABELS[dataset_name][select_set], split='val',img_transforms=img_transforms, mask_transforms=mask_transforms,
                        n_ways=1, n_shots=1, n_queries=1)
        labels = CLASS_LABELS[dataset_name][select_set]
        labels_val = CLASS_LABELS[dataset_name]["all"]-labels
        
else:
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
    transforms = Compose([Resize(size=input_size)])
    dataset = gen_dataset(
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

metric = Metric(max_label=19, n_runs=n_runs)

def test_model(model):
    
    metric_dict={}

    model.eval()
    with torch.no_grad():
        for run in range(n_runs):
            measure={}
            set_seed(seed+run)
 

            test_loader = DataLoader(
                dataset,
                batch_size=setup['batch_size'],
                shuffle=True,
                num_workers=1,
                pin_memory=True,
                drop_last=True
            )
        

            lossq=0
            losspar=0
            train_loss={}
            train_loss["loss_query"]=[]
            train_loss["loss_PAR"]=[]
            for t,samples in enumerate(test_loader):
                print_every=100

                label_ids = list(samples['class_ids'])

                model = model.to(device=device)  # move the model parameters to CPU/GPU

                #get a list of support images ( Sx W x H x W)
                support_img=[[shot.cuda() for shot in way]
                            for way in samples['support_images']]
                fg_mask=[[shot[f'fg_mask'].float().cuda() for shot in way]
                            for way in samples['support_mask']]
                bg_mask=[[shot[f'bg_mask'].float().cuda() for shot in way]
                            for way in samples['support_mask']]
                query_img= [query.cuda()
                            for query in samples['query_images']]

                query_gt = torch.cat([queryGT.long().cuda()
                            for queryGT in samples['query_labels']], dim=0)

                #get the predicted query and the Prototype alignment loss
                pred_query,lossPAR= model(support_img,fg_mask,bg_mask,query_img)

                query_labels = torch.cat(
                [query_label.cuda()for query_label in samples['query_labels']], dim=0)
                
                lossQ= criterion(pred_query,query_gt)

                loss = lossQ+ lossPAR * lambda_PAR
               
                lossq+=lossQ.detach().cpu().numpy()
                losspar+=lossPAR

                if (t+1) % print_every == 0:
                    print('Epoch %d, Iteration %d, Seg loss = %.8f' % (run, t+1, (lossq/(t+1))))
                    print('Epoch %d, Iteration %d, PAR loss = %.8f' % (run, t+1, (losspar/(t+1))))
                    train_loss["loss_query"].append(lossq/(t+1))
                    train_loss["loss_PAR"].append(losspar/(t+1))
                
                metric.record(np.array(pred_query.argmax(dim=1)[0].cpu()),
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)   

            measure["loss"]=train_loss
            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)
            measure["class IoU"]=classIoU
            measure["mean IoU"]=meanIoU
            measure["class IoU binary"]=classIoU_binary
            measure["mean IoU binary"]=meanIoU_binary
            metric_dict[run]=measure
        
    return metric_dict

#cfg True means align is on
learning_rate=1e-3
milestones= [steps//3,steps//2,steps]
if(setup['backbone']=="dv3"):
    print("dv3")
    model = FewShotSegV3(cfg={'align': True},distfunc=setup["distfunc"])
elif(setup['backbone']=="dv3INet"):
    model = FewShotSegV3Inet(cfg={'align': True},distfunc=setup["distfunc"])
elif(setup['backbone']=="resnet50"):
    model =FewShotSegResnet(cfg={'align': True},distfunc=setup["distfunc"])
else:
    print("vgg")
    model = FewShotSeg(cfg={'align': True},distfunc=setup["distfunc"])
model.load_state_dict(torch.load(model_path))


metrics=test_model(model)

print(metrics)