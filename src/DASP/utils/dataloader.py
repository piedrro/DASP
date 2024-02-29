from torch.utils import data
import torch
import numpy as np
from skimage import exposure
import albumentations as A
import torch.nn.functional as F
from torchvision.transforms import Normalize
import cv2
import matplotlib.pyplot as plt
from utils.visualise import normalize99, rescale01






class load_dataset(data.Dataset):

    def __init__(self,
                 images: list = [],
                 masks: list = [],
                 labels: list = [],
                 treatment_labels: list = [],
                 num_classes: int = 0,
                 augment: bool = False,
                 mode = "segmentor",
                 maskrcnn: bool = False,
                 image_size: tuple = (),
                 ):

        self.augment = augment
        self.mode = mode
        self.images = images
        self.masks = masks
        self.labels = labels
        self.num_classes = num_classes
        self.maskrcnn = maskrcnn
        self.image_size = image_size
                    

    def __len__(self):
        return len(self.images)

    def resize_image(self, img, mask = []):
        
        from albumentations.augmentations.crops.transforms import RandomCrop
        
        img = np.moveaxis(img,0,-1)
        
        if type(mask) != list:
        
            transform = A.Compose([RandomCrop(height=self.image_size[0], width=self.image_size[1])])
            transformed = transform(image=img, mask = mask)
    
            img = transformed["image"]
            mask = transformed["mask"]
            
            img = np.moveaxis(img,-1,0)
            
            return img, mask

        else:
            
            transform = A.Compose([RandomCrop(height=self.image_size[0], width=self.image_size[1])])
            transformed = transform(image=img)
    
            img = transformed["image"]

            img = np.moveaxis(img,-1,0)
            
            return img
            

    def augment_segmentation_images(self, img, mask):

        from albumentations.augmentations.blur.transforms import Blur
        from albumentations.augmentations.transforms import RGBShift, GaussNoise, PixelDropout, ChannelShuffle
        from albumentations import RandomBrightnessContrast, RandomRotate90, Flip, Affine
        
        """applies albumentations augmentation to image and mask, including resizing the images/mask to the crop_size"""


        shift_channels = A.Compose([Affine(translate_px=[-3,3])])

        for i, chan in enumerate(img):
            if i != 0:
                chan = shift_channels(image=chan)['image']
                img[i] = chan

        img = np.moveaxis(img,0,-1)
        
        # geometric transforms
        transform = A.Compose([
            Flip(),
            Affine(scale=(0.8,1.2),shear=(-10,10),rotate=(-360,360)),
        ])

        transformed = transform(image=img, mask = mask)
        
        img = transformed["image"]
        mask = transformed["mask"]

        #pixel transforms
        transform = A.Compose([
            GaussNoise(var_limit=0.0005, per_channel=True, always_apply=False),
            Blur(blur_limit=5, always_apply=False, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.5, always_apply=False),
            PixelDropout(dropout_prob=0.05, per_channel=True, p=0.5),
        ])

        img = transform(image=img)['image']

        img = np.moveaxis(img,-1,0)

        return img, mask

    def augment_classifier_images(self, img):
       
            from albumentations.augmentations.blur.transforms import Blur
            from albumentations.augmentations.transforms import RGBShift, GaussNoise, PixelDropout, ChannelShuffle
            from albumentations import RandomBrightnessContrast, RandomRotate90, Flip, Affine
            
            """applies albumentations augmentation to image and mask, including resizing the images/mask to the crop_size"""

            shift_channels = A.Compose([Affine(translate_px=[-5,5])])
            
            mask = img[0].copy()
             
            for i, chan in enumerate(img):
                if i != 0:
                    chan = shift_channels(image=chan)['image']
                    chan[mask==0] = 0
                    img[i] = chan
            
            img = np.moveaxis(img,0,-1)
            
            # geometric transforms
            transform = A.Compose([
            RandomRotate90(),
            Flip(),
            Affine(
                scale=(0.5,2),
                shear=(-30,30),
                rotate=(-360,360),
                translate_px=[-10,10]
                ),
            ])
            
            img = transform(image=img)['image']
            mask = img.copy()
             
            #pixel transforms
            transform = A.Compose([
            GaussNoise(var_limit=0.05, per_channel=True, always_apply=False),
            Blur(blur_limit=10, always_apply=False, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.5, always_apply=False),
            PixelDropout(dropout_prob=0.05, per_channel=True, p=0.5),
            ])
             
            img = transform(image=img)['image']
            img[mask==0] = 0
             
            img = np.moveaxis(img,-1,0)
             
            return img


    def extract_maskrcnn_data(self, mask):
    
        bboxes = []
        
        mask_ids = np.unique(mask)
        
        mask_stack = []
        
        for mask_id in mask_ids:
            
            if mask_id != 0:
                
                sub_mask = np.zeros(mask.shape, dtype=np.uint8)
                sub_mask[mask == mask_id] = 1
                
                cnt, _ = cv2.findContours(sub_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                
                x,y,w,h = cv2.boundingRect(cnt[0])
                bbox = [x, y, x+w, y+h]
                
                bboxes.append(bbox)
                mask_stack.append(sub_mask)
                       
        bboxes = np.array(bboxes) 
        
        if len(mask_stack) > 0:
            mask_stack = np.stack(mask_stack)
        else:
            mask_stack = np.zeros(mask.shape, dtype=np.uint8)
        
        return bboxes, mask_stack


    def postprocess(self, x, y, m = ""):
        """re-formats the image/masks/labels for training"""

        # Typecasting
        x = torch.from_numpy(x.copy()).float()
        y = F.one_hot(torch.tensor(y), num_classes=self.num_classes).float()
        
        if type(m) != str:
            m = torch.from_numpy(m.copy()).float()
            return x, y, m
        else:
            return x, y
            

    def __getitem__(self, index: int):
        
        if self.mode == "segmentor":

            image, mask, label = self.images[index], self.masks[index], self.labels[index]
            
            image = image.astype(np.float32)
                    
            if self.augment:
                image, mask = self.augment_segmentation_images(image, mask)
            
            if self.image_size != ():
                image, mask = self.resize_image(image, mask)
            
            if self.maskrcnn == True:
                boxes, masks = self.extract_maskrcnn_data(mask)
                
                boxes = torch.from_numpy(boxes).float()
                image = torch.from_numpy(image.copy()).float()
                masks = torch.from_numpy(masks.copy()).float().type(torch.int8)
                labels = torch.tensor([label]*len(boxes), dtype=torch.int64)
        
                # labels = torch.ones((len(boxes),), dtype=torch.int64)
                
                targets = [{"boxes":boxes, "masks":masks, "labels":labels}]
    
                return image, targets
                
            else:
                image,label,mask = self.postprocess(image,label,mask)
                
                return image, mask, label
        
        if self.mode == "classifier":
            
            image, label = self.images[index], self.labels[index]
            
            image = image.astype(np.float32)
            
            if self.augment:
                image = self.augment_classifier_images(image)
                # image = normalize99(image)
                # image = rescale01(image)
                
            
            if self.image_size != ():
                image, mask = self.resize_image(image, mask)
    
            image,label = self.postprocess(image,label)
            
            return image, label
    
        if self.mode == "pipeline":
            
            image = self.images[index]
            
            image = image.astype(np.float32)
            
            if self.augment:
                image = self.augment_classifier_images(image)
                # image = normalize99(image)
                # image = rescale01(image)
            
            if self.image_size != ():
                image = self.resize_image(image)
                
            image = torch.from_numpy(image.copy()).float()
             
            return image
             
             
             
             