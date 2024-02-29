import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from multiprocessing import Pool
import tqdm
import pandas as pd
import pathlib
import shutil
from PIL import Image


def yolo_to_bboxes(yolo_data, img_width, img_height):
    
    bboxes = []
    labels = []
    
    for data in yolo_data:
        label = data[0]
        x1 = float(img_width) * (2.0 * float(data[1]) - float(data[3])) / 2.0
        y1 = float(img_height) * (2.0 * float(data[2]) - float(data[4])) / 2.0
        x2 = float(img_width) * (2.0 * float(data[1]) + float(data[3])) / 2.0
        y2 = float(img_height) * (2.0 * float(data[2]) + float(data[4])) / 2.0
        
        box = [int(x1), int(y1), int(x2), int(y2)]
        bboxes.append(box)
        labels.append(label)
 
    return bboxes, labels
        

def bboxes_to_yolo(bboxes, labels, img_width, img_height):
    
    yolo_data = []
    
    for (box,label) in zip(bboxes,labels):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1

        # normalize the bounding box coordinates and dimensions
        center_x /= img_width
        center_y /= img_height
        width /= img_width
        height /= img_height

        yolo_data.append([label, center_x, center_y, width, height])

    return yolo_data

def read_yolo_file(label_path):
    
    with open(label_path) as f:
        data = f.read()
        
    yolo_data = [[float(num) for num in line.split(" ")] for line in data.split("\n") if line != ""]
    
    return yolo_data
    
def write_yolo_bboxes_to_file(yolo_bboxes, filename):
    
    with open(filename, 'w') as f:
        for bbox in yolo_bboxes:
            f.write(' '.join(map(str, bbox)) + '\n')
    
def get_yolo_dataset():
    
    yolo_dir = r"/home/turnerp/PycharmProjects/AK-DASP/pothole_dataset_v8/train/images/"

    image_paths = glob(yolo_dir + "*.jpg")[:10]
    
    label_paths = [path.replace("/images/","/labels/").replace(".jpg",".txt") for path in image_paths]

    yolo_dataset = []
    
    for i, (image_path, label_path) in enumerate(zip(image_paths,label_paths)):
        
        if os.path.exists(label_path) and os.path.exists(image_path):
            
            img = Image.open(image_path)
            img = np.array(img)
            
            img_width = img.shape[1]
            img_height = img.shape[0]
            
            yolo_data = read_yolo_file(label_path)
            
            bboxes, labels = yolo_to_bboxes(yolo_data, img_width, img_height)
            
            yolo_data = bboxes_to_yolo(bboxes, labels, img_width, img_height)
            bboxes, labels = yolo_to_bboxes(yolo_data, img_width, img_height)

            for bbox in bboxes:
                
                x1,y1,x2,y2 = bbox
                
                dat_dict = dict(label_path = label_path,
                                image_path = image_path,
                                label_name = os.path.basename(label_path),
                                image_name = os.path.basename(image_path),
                                x1 = x1,
                                y1 = y1,
                                x2 = x2,
                                y2 = y2)
                        
                yolo_dataset.append(dat_dict)
            
    yolo_dataset = pd.DataFrame.from_dict(yolo_dataset)
    
    return yolo_dataset

def extract_yolo_data(mask, label):

    img_width = mask.shape[1]
    img_height = mask.shape[0]

    bboxes = []
    
    mask_ids = np.unique(mask)
    
    for mask_id in mask_ids:
        
        if mask_id != 0:
            
            sub_mask = np.zeros(mask.shape, dtype=np.uint8)
            sub_mask[mask == mask_id] = 1
            
            cnt, _ = cv2.findContours(sub_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            x,y,w,h = cv2.boundingRect(cnt[0])
            bbox = [x, y, x+w, y+h]
            
            bboxes.append(bbox)
          
    labels = [label]*len(bboxes)    
    
    yolo_data = bboxes_to_yolo(bboxes, labels, img_width, img_height)

    return yolo_data

def yolo_to_mask(yolo_data, img_width, img_height):

    mask = np.zeros((img_height, img_width), dtype=np.uint16)
    
    labels = []
    
    for i,yolo in enumerate(yolo_data):
        
        label = yolo.pop(0)
        labels.append(int(label))
        
        coords = [yolo[i:i+2] for i in range(0, len(yolo[1:]), 2)]
        
        cnt = []
        
        for [x,y] in coords:
            
            cnt.append([int(x*img_width),int(y*img_height)])
            
        cnt = np.array(cnt).reshape(-1,1,2)
            
        cv2.drawContours(mask, [cnt], -1, i+1, -1)
        
    return mask, labels

def mask_to_yolo(mask, labels, mode = "bboxes"):
    
    img_width = mask.shape[1]
    img_height = mask.shape[0]

    mask_ids = sorted(np.unique(mask))
    mask_ids.pop(0)
    
    if type(labels) == int:
        labels = [labels]*len(mask_ids)

    assert len(labels) == len(mask_ids)

    if mode == "bboxes":
        
        bboxes = []
        
        for mask_id in mask_ids:
            
            sub_mask = np.zeros(mask.shape, dtype=np.uint8)
            sub_mask[mask == mask_id] = 1
            
            cnt, _ = cv2.findContours(sub_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            x,y,w,h = cv2.boundingRect(cnt[0])
            bbox = [x, y, x+w, y+h]
            
            bboxes.append(bbox)
              
        yolo_data = bboxes_to_yolo(bboxes, labels, img_width, img_height)
        
    else:

        yolo_data = []
        
        for i, mask_id in enumerate(mask_ids):
            
            yolo = [labels[i]]
            
            sub_mask = np.zeros(mask.shape, dtype=np.uint8)
            sub_mask[mask==mask_id] = 1
            
            cnt, _ = cv2.findContours(sub_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
            for [[x,y]] in cnt[0]:
                
                x /= img_width
                y /= img_height
                
                yolo.append(x)
                yolo.append(y)
            
            yolo_data.append(yolo)
            
    return yolo_data


def export_yolo_data(yolo_data):

    image_name = os.path.basename(yolo_data["file_names"]).replace(".tif",f".png")
    label_name = image_name.replace(".png",".txt")
    
    save_dir = yolo_data["save_dir"]
    image_directory = yolo_data["image_directory"]
    label_directory = yolo_data["label_directory"]
    
    image_path = os.path.abspath(os.path.join(image_directory, image_name))
    label_path = os.path.abspath(os.path.join(label_directory, label_name))
    
    image = yolo_data["images"]
    label = yolo_data["labels"]
    mask = yolo_data["masks"]

    image = np.transpose(image, (1, 2, 0))
    plt.imsave(image_path, image, format='png')
    
    yolo_data = mask_to_yolo(mask, label, mode="segmentations")
    write_yolo_bboxes_to_file(yolo_data, label_path)
    
    
def export_yolo_dataset(dataset, yolo_dir = "yolo_dataset", export_mode = "segmentations"):

    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)

    yml_dict = {}
    
    for dataset_name, dataset_values in dataset.items():
        
        save_dir = os.path.join(yolo_dir, dataset_name)
        image_directory = os.path.join(save_dir, "images")
        label_directory = os.path.join(save_dir, "labels")
        
        label_list = dataset_values["label_list"]

        if not os.path.exists(image_directory):
            os.makedirs(image_directory)
            
        if not os.path.exists(label_directory):
            os.makedirs(label_directory)
        
        label_names = dataset_values['label_list']
        
        yml_dict[dataset_name] = os.path.abspath(save_dir)
        yml_dict["names"] = label_names
        yml_dict["nc"] = len(label_names)

        export_data = []
    
        for i in range(len(dataset_values["images"])):
            
            dat = {}
            
            for key, value in dataset_values.items():
                
                if key != "label_list":
                
                    dat[key] = value[i]
                    
                else:
                
                    dat["label_list"] = value
                    
            dat["export_mode" ] = export_mode
            dat["save_dir"] = save_dir
            dat["image_directory"] = image_directory
            dat["label_directory"] = label_directory
                    
            export_data.append(dat)
            
        with Pool(10) as pool:
            
            tqdm.tqdm(pool.imap(export_yolo_data, export_data), total=len(export_data))
            
            pool.close()
            pool.join()
                     
    yaml_content = ""
            
    if "train" in yml_dict.keys():
        yaml_content += f"train: {yml_dict['train']}\n"
    if "val" in yml_dict.keys():
        yaml_content += f"val: {yml_dict['val']}\n"
    if "test" in yml_dict.keys():
        yaml_content += f"test: {yml_dict['test']}\n"
    yaml_content += f"nc: {yml_dict['nc']}\n"
    yaml_content += f"names: {yml_dict['names']}\n"
        
    yml_path = os.path.join(yolo_dir, 'data.yaml')

    with pathlib.Path(yml_path).open('w') as f:
        f.write(yaml_content)
    
    return yml_path