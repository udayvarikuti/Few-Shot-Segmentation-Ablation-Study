import cityscapesscripts.preparation.createTrainIdInstanceImgs as ins
import os, glob
from cityscapesscripts.helpers.annotation import Annotation
from cityscapesscripts.helpers.labels import name2label
from collections import defaultdict
import random
from PIL import Image
from PIL import ImageDraw
from torch.utils.data import Dataset
import numpy as np

class Cityscape(Dataset):
    def __init__(self, base_path, dataset_size, labels=None, split='train', img_transforms=None, mask_transforms=None, n_ways=1,
                 n_shots=1, n_queries=1,apply_flip=False):
        super().__init__()
        self.base_path = base_path
        self.image_path = os.path.join(self.base_path, 'leftImg8bit', split)
        self.annotation_path = os.path.join(self.base_path, 'gtFine', split)
        self.labels = list(labels)
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
        dataset_labels =  ['bicycle', 'sidewalk', 'traffic sign', 'rider','truck','road', 'building', 'wall', 'fence', 'pole', 'traffic light', 'vegetation', 'terrain', 'sky', 
          'person', 'car', 'truck', 'bus', 'train', 'motorcycle']
        return dict(zip(range(len(dataset_labels)), dataset_labels))

    def get_labels_classwise(self, files):
        searchFine = os.path.join(self.annotation_path, "*" ,"*_gt*_polygons.json")
        filesFine = glob.glob(searchFine)
        class_files = defaultdict(set)
        if self.labels is not None:
            self.selected_labels = {key: self.label_dict[key] for key in self.labels}
        else:
            self.selected_labels = self.label_dict
        for f in filesFine:
            annotation = Annotation()
            annotation.fromJsonFile(f)
            for obj in annotation.objects:
                if obj.label in list(self.selected_labels.values()):
                    class_files[obj.label].add(f)
        return class_files
        
    def generate_dataset(self):
        dataset = []
        for _ in range(self.dataset_size):
            self.support_classes = random.choices(list(self.selected_labels.values()), k=self.n_ways)
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
                print("Label '{}' not known.".format(label))
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
        ext_path = '/'.join(ann_filename.split('/')[-2:]).replace('gtFine_polygons', 'leftImg8bit').replace('.json','.png')
        return self.image_path + '/' + ext_path
    
    def __getitem__(self, idx):
        sample = {}
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
                shot = {}
                shot['fg_mask'] = self.mask_transforms(self.createLabelImg(f, class_label)[0]).squeeze(0)
                shot['bg_mask'] = self.mask_transforms(self.createLabelImg(f, class_label)[1]).squeeze(0)
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
                             
        return sample
        
# USAGE
# CLASS_LABELS = {
#     'VOC': {
#         'all': set(range(1, 21)),
#         0: set(range(1, 21)) - set(range(1, 6)),
#         1: set(range(1, 21)) - set(range(6, 11)),
#         2: set(range(1, 21)) - set(range(11, 16)),
#         3: set(range(1, 21)) - set(range(16, 21)),
#     },
#     'COCO': {
#         'all': set(range(1, 81)),
#         0: set(range(1, 81)) - set(range(1, 21)),
#         1: set(range(1, 81)) - set(range(21, 41)),
#         2: set(range(1, 81)) - set(range(41, 61)),
#         3: set(range(1, 81)) - set(range(61, 81)),
#     },
#     'Cityscape': {
#         'all': set(range(1, 20)),
#         0: set(range(6, 20)),
#         1: set(range(1, 6)),
#     }
# }
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

# dataset = Cityscape(cityscapesPath, dataset_size, labels=CLASS_LABELS['Cityscape'][0], split='train',img_transforms=img_transforms, mask_transforms=mask_transforms,
#                    n_ways=2, n_shots=1, n_queries=1)