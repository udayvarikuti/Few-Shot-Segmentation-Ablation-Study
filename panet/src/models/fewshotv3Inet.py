"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .dv3INet import Encoder
import lpips

class FewShotSegV3Inet(nn.Module):
    def __init__(self, in_channels=3, pretrained_path=None, cfg=None,  distfunc="cosine"):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        self.distfunc=distfunc
        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)),]))

    def forward(self, img_sup, fore_mask, back_mask, query_imgs):
        n_ways = len(img_sup)
        n_shots = len(img_sup[0])
        n_queries = len(query_imgs)
        batch_size = img_sup[0][0].shape[0]
        img_size = img_sup[0][0].shape[-2:]

        #concatenate all images along dim 0
        sup_concat = []
        for cls in img_sup:
            sup_concat.append(torch.cat(cls, dim=0))
        query_concat = [torch.cat(query_imgs, dim=0)]
        imgs_concat = torch.cat(sup_concat + query_concat, dim=0)
        img_features = self.encoder(imgs_concat)
        feature_size = img_features.shape[-2:]

        sup_features = img_features[:n_ways * n_shots * batch_size].view(n_ways, n_shots, batch_size, -1, *feature_size) 
        query_features = img_features[n_ways * n_shots * batch_size:].view(n_queries, batch_size, -1, *feature_size)

        fore_mask_stack = []
        for mask in fore_mask:
            fore_mask_stack.append(torch.stack(mask, dim=0))
        back_mask_stack = []
        for mask in back_mask:
            back_mask_stack.append(torch.stack(mask, dim=0))
        fore_mask = torch.stack(fore_mask_stack, dim=0) 
        back_mask = torch.stack(back_mask_stack, dim=0)

        align_loss = 0
        outputs = []
        for batch in range(batch_size):
            fore_sup_features = []
            back_sup_features = []
            for way in range(n_ways):
                features_fore_way = []
                features_back_way = []
                for shot in range(n_shots):
                    sup_image = sup_features[way, shot, [batch]]
                    fore_mask_img = fore_mask[way, shot, [batch]]
                    back_mask_img = back_mask[way, shot, [batch]]
                    features_fore_way.append(self.compute_features(sup_image, fore_mask_img))
                    features_back_way.append(self.compute_features(sup_image, back_mask_img))
                fore_sup_features.append(features_fore_way)
                back_sup_features.append(features_back_way)

            fore_prototypes, back_prototype = self.compute_prototype(fore_sup_features, back_sup_features)
            prototypes = [back_prototype,] + fore_prototypes
            dist = []
            for prototype in prototypes:
                dist.append(self.compute_dist(query_features[:, batch], prototype))

            prediction = torch.stack(dist, dim=1)
            outputs.append(F.interpolate(prediction, size=img_size, mode='bilinear'))
            if self.config['align'] and self.training:
                align_loss_batch = self.alignLoss(query_features[:, batch], prediction, sup_features[:, :, batch],
                                                fore_mask[:, :, batch], back_mask[:, :, batch])
                align_loss += align_loss_batch
        output = torch.stack(outputs, dim=1)
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size

    def compute_dist(self, fts, prototype, scaler=20):
        if(self.distfunc=="cosine"):
            dist = F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scaler
            #print(self.distfunc)
        elif(self.distfunc=="euclidean"):
            fts=torch.permute(fts,(0,2,3,1))
            scaler=5
            dist= F.pairwise_distance(fts,prototype[:,None,None,:],p=2)*scaler
            #print(self.distfunc)
        elif(self.distfunc=="manhattan"):
            fts=torch.permute(fts,(0,2,3,1))
            scaler=10
            dist= F.pairwise_distance(fts,prototype[:,None,None,:],p=1)*scaler
            print(self.distfunc)
        elif(self.distfunc=="lpips"):
            loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
            dist= loss_fn_alex(fts, prototype[..., None, None])
            print(self.distfunc)
        return dist

    def compute_features(self, features, mask):
        features = F.interpolate(features, mask.shape[-2:], mode='bilinear')
        masked_features = torch.sum(features * mask[None, ...], dim=(2, 3)) / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_features

    def compute_prototype(self, fore_features, back_features):
        n_ways, n_shots = len(fore_features), len(fore_features[0])
        fore_proto = [sum(way) / n_shots for way in fore_features]
        back_proto = sum([sum(way) / n_shots for way in back_features]) / n_ways
        return fore_proto, back_proto

    def alignLoss(self, query_features, pred, support_features, fore_mask, back_mask):
        n_ways, n_shots = len(fore_mask), len(fore_mask[0])
        pred_mask = pred.argmax(dim=1, keepdim=True)
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]
        pred_mask = torch.stack(binary_masks, dim=1).float()
        qry_prototypes = torch.sum(query_features.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        qry_prototypes = qry_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)

        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            prototypes = [qry_prototypes[[0]], qry_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = support_features[way, [shot]]
                supp_dist = [self.compute_dist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=fore_mask.shape[-2:],
                                          mode='bilinear')
                supp_label = torch.full_like(fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[fore_mask[way, shot] == 1] = 1
                supp_label[back_mask[way, shot] == 1] = 0
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss
