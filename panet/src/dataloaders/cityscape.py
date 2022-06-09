import cityscapesscripts.preparation.createTrainIdInstanceImgs as ins
import os, glob, sys
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels import name2label
from collections import defaultdict
import random
from PIL import Image
from PIL import ImageDraw
import torchvision
from torch.utils.data import Dataset
import numpy as np

class Cityscape(Dataset):
    def __init__(self, annotation_path, dataset_size, img_transforms=None, mask_transforms=None, n_ways=1,
                 n_shots=1, n_queries=1,apply_flip=False):
        super().__init__()
        self.annotation_path = annotation_path
        self.dataset_size = dataset_size
        self.label_dict = self.get_label_dict()
        self.class_files = self.get_labels_classwise(self)
        self.n_ways = n_ways
        self.n_shots = n_shots
        self.n_queries = n_queries
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.dataset = self.generate_dataset()
        
    def get_label_dict(self):
        labels =  ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 
          'person','rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
        return dict(zip(range(len(labels)), labels))

    def get_labels_classwise(self, files):
        searchFine = os.path.join(self.annotation_path ,"gtFine", "*" , "*" ,"*_gt*_polygons.json")
        filesFine = glob.glob(searchFine)
        class_files = defaultdict(set)
        for f in filesFine:
            annotation = Annotation()
            annotation.fromJsonFile(f)
            for obj in annotation.objects:
                if obj.label in list(self.label_dict.values()):
                    class_files[obj.label].add(f)
        return class_files
    
    def generate_dataset(self):
        dataset = []
        for _ in range(self.dataset_size):
            sample = {}
            self.support_classes = random.choices(list(self.label_dict.values()), k=self.n_ways)
            self.query_classes = random.choices(self.support_classes, k=self.n_queries)
            
            support_labels, query_labels = [], []
            for sup_class in self.support_classes:
                support_labels.append(random.choices(list(self.class_files[sup_class]), k=self.n_shots))
            for que_class in self.query_classes:
                query_labels.extend(random.choices(list(self.class_files[que_class]), k=1))
            dataset.append(support_labels+query_labels)
            
        return dataset
    
    def createLabelImg(self, f, class_label):
        annotation = Annotation()
        annotation.fromJsonFile(f)
        size = (annotation.imgWidth , annotation.imgHeight)
        background = name2label['unlabeled'].id
        fg_mask = Image.new("1", size, background)
        drawer = ImageDraw.Draw(fg_mask)
        for obj in annotation.objects:
            label = obj.label
            if ( not label in name2label ) and label.endswith('group'):
                label = label[:-len('group')]
            if not label in name2label:
                print( "Label '{}' not known.".format(label) )
            elif obj.label == class_label:
                # If the object is deleted, skip it
                if obj.deleted or name2label[label].id < 0:
                    continue
                polygon = obj.polygon
                val = name2label[label].id
                drawer.polygon(polygon, fill=1)
        bg_mask = Image.new("1", size, 1)
        bg_mask = np.asarray(bg_mask)
        bg_mask[np.array(fg_mask)==1] = 0
        return fg_mask, Image.fromarray(bg_mask)

    def __len__(self):
        return self.dataset_size
    
    def get_filename_from_annotation(self, ann_filename):
        image_path = os.path.join(self.annotation_path ,"leftImg8bit")
        ext_path = '/'.join(ann_filename.split('/')[-3:]).replace('gtFine_polygons', 'leftImg8bit').replace('.json'
                                                                                                            , '.png')
        return image_path + '/' + ext_path
    
    def __getitem__(self, idx):
        sample = {}
        shot = {}
        support_labels = self.dataset[idx][:-self.n_queries]
        query_labels = self.dataset[idx][-self.n_queries:]
        
        sample['support_images'] = [[self.img_transforms(Image.open(self.get_filename_from_annotation(path)))  
                                    for path in label_path] for label_path in support_labels]
        sample['query_images'] = [self.img_transforms(Image.open(self.get_filename_from_annotation(label_path)))
                                   for label_path in query_labels]
        sample['support_mask'] = []
        for way in support_labels:
            masks = []
            for (f, class_label) in zip(way,self.support_classes):
                shot['fg_mask'] = self.mask_transforms(self.createLabelImg(f, class_label)[0])
                shot['bg_mask'] = self.mask_transforms(self.createLabelImg(f, class_label)[1])
                masks.append(shot)
            sample['support_mask'].append(masks)
                
#         sample['support_mask'] = [[shot['fg_mask'] = self.mask_transforms(self.createLabelImg(f, class_label)[0]),
#                             shot['bg_mask'] = self.mask_transforms(self.createLabelImg(f, class_label)[1]) 
#                             for (f, class_label) in zip(way,self.support_classes)] for way in support_labels]
        #shot['bg_mask'] = [[self.mask_transforms(self.createLabelImg(f, class_label)[1]) for (f, class_label) in 
                              #zip(way,self.support_classes)] for way in support_labels]
        #sample['support_mask'] = shot
        sample['query_labels'] = [self.mask_transforms(self.createLabelImg(f, class_label)[0]) for (f, class_label) in 
                                   zip(query_labels,self.query_classes)]
        
#         #Apply Transformations
#         if self.transforms is not None:
#             sample = self.transforms(sample)
#         # Transform to tensor
#         if self.to_tensor is not None:
#             sample = self.to_tensor(sample)
                             
        return sample

### USAGE
# from torchvision.transforms import Compose
# from torchvision import transforms

# #RandomMirror, Resize, ToTensorNormalize
# input_size = (417, 417)
# cityscapesPath = '../../data/Cityscape'
# dataset_size = 300

# flip_transform = transforms.RandomHorizontalFlip(p=0.25)
# img_transforms = Compose([transforms.ToTensor(),
#                       transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#                       transforms.Resize(size=input_size),
#                       flip_transform])
# mask_transforms = Compose([transforms.ToTensor(),
#                       transforms.Resize(size=input_size),
#                       flip_transform])

# dataset = Cityscape(cityscapesPath, dataset_size, img_transforms=img_transforms, mask_transforms=mask_transforms,
#                    n_ways=5, n_shots=2, n_queries=2)