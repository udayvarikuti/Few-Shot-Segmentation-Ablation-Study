import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import Compose
from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
import numpy as np

USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
    #set the device to gpu0
    torch.cuda.set_device(device=0)
else:
    device = torch.device('cpu')


input_size = (417, 417)
batch_size=1
steps=30000
lambda_PAR=1

#Get the dataset functions from here
dataset_name="VOC"
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

labels = CLASS_LABELS[dataset_name][0]
transforms = Compose([Resize(size=input_size),
                        RandomMirror()])
dataset = gen_dataset(
    base_dir=data_dir,
    split=data_split,
    transforms=transforms,
    to_tensor=ToTensorNormalize(),
    labels=labels,
    max_iters=batch_size*steps,
    n_ways=1,
    n_shots=2,
    n_queries=1
)

train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    drop_last=True
)

model_path= "./misc/fewshotvgg_1w2s.pth"

print_every=100
save_every=1000
def train_model(model, optimizer, scheduler ,epochs=1):

    train_loss={}
    #lossfn=nn.CosineEmbeddingLoss(margin=0,reduction="mean")
    lossfn=nn.MSELoss()
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    train_loss["loss_query"]=0
    train_loss["loss_PAR"]=0
    for e in range(epochs):
        lossq=0
        losspar=0
        for t,samples in enumerate(train_loader):
            
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
            if (t+1) % save_every == 0:
                torch.save(model.state_dict(),model_path)    

        train_loss["loss_query"]=lossq/len(train_loader)
        train_loss["loss_PAR"]=losspar/len(train_loader)

    return (train_loss)

#cfg True means align is on
learning_rate=1e-3
milestones= [steps//3,steps//2,steps]
model = FewShotSeg(cfg={'align': True})
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum=0.9,weight_decay=0.00005)
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
criterion = nn.CrossEntropyLoss(ignore_index=255)

train_loss=train_model(model, optimizer, scheduler ,epochs=1)