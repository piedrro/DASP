
import pandas as pd
import numpy as np
from glob2 import glob
import os
from datetime import datetime
import tifffile
import json
import pathlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models
# from trainer import Trainer
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import pandas as pd
import itertools
from itertools import compress, product
from skimage import exposure
import cv2
import json
from imgaug import augmenters as iaa
# import albumentations as A
import tifffile
import tqdm
from multiprocessing import Pool
import traceback
from functools import partial
import random
import torch.nn.functional as F
import pathlib
from skimage.registration import phase_cross_correlation
import scipy
from utils.visualise import normalize99,rescale01
from utils.stats import get_stats
import warnings



def align_images(images):

    try:

        shift, error, diffphase = phase_cross_correlation(images[0], images[1], upsample_factor=100)
        images[1] = scipy.ndimage.shift(images[1], shift).astype(np.uint16)

    except:
        pass

    return images

def resize_image(image_size, h, w, cell_image_crop, colicoords = False, resize=False):
    
    cell_image_crop = np.moveaxis(cell_image_crop, 0, -1)

    if h < image_size[0] and w < image_size[1]:
        
        seq = iaa.CenterPadToFixedSize(height=image_size[0], width=image_size[1])
 
    else:
        
        if resize == True:
        
            if h > w:
                seq = iaa.Sequential([iaa.Resize({"height": image_size[0], "width": "keep-aspect-ratio"}),
                                      iaa.CenterPadToFixedSize(height=image_size[0],width=image_size[1])])
            else:
                seq = iaa.Sequential([iaa.Resize({"height": "keep-aspect-ratio", "width": image_size[1]}),
                                      iaa.CenterPadToFixedSize(height=image_size[0],width=image_size[1])])
                
        else:
            seq = iaa.Sequential([iaa.CenterPadToFixedSize(height=image_size[0],width=image_size[1]),
                                  iaa.CenterCropToFixedSize(height=image_size[0],width=image_size[1])])

    seq_det = seq.to_deterministic()
    cell_image_crop = seq_det.augment_images([cell_image_crop])[0]
        
        
    cell_image_crop = np.moveaxis(cell_image_crop, -1, 0)
    
    if colicoords:
        if h > w:
            cell_image_crop = np.rot90(cell_image_crop,axes=(1,2))


    return cell_image_crop

def get_metadata(dataset_directory):
    
    parts = [*pathlib.Path(dataset_directory).parts, "*", "All_images", "**", "*.tif"]
    glob_search = str(pathlib.Path('').joinpath(*parts))

    files = glob(glob_search)

    file_metadata = []

    for file in files:
        
        file_name = os.path.basename(file)
        
        dataset_folder = pathlib.Path(dataset_directory).parts[-1]
        
        dataset_index = pathlib.Path(file).parts.index(dataset_folder)
        
        dataset = pathlib.Path(file).parts[dataset_index+1]
        
        treatment_concentration = file_name.split("_")[3]
        
        if "[" in treatment_concentration:
        
            treatment_concentration = treatment_concentration.strip("[]")
        
            if "0" in treatment_concentration and len(treatment_concentration) > 1:
                treatment_concentration = treatment_concentration[:1] + '.' + treatment_concentration[1:]
                
            treatment_concentration = float(treatment_concentration)
            
        meta = dict(
            path = file,
            file_name = file_name,
            dataset = dataset,
            condition = "",
            experiment_date = datetime.strptime(file_name.split("_")[0], "%y%m%d").date(),
            repeat = file_name.split("_")[1],
            experiment_id = "",
            species_id = file_name.split("_")[2],
            treatment_concentration = treatment_concentration,
            project_code = file_name.split("_")[4],
            processing_stage = file_name.split("_")[5],
            aquisition_id = file_name.split("_")[6],
            treatment = file_name.split("_")[7],
            position_id = file_name.split("_")[8].replace(".tif",""),
            )
        
        file_metadata.append(meta)
        
    file_metadata = pd.DataFrame(file_metadata)

    for dataset in file_metadata.dataset.unique():
        
        if dataset == "MG1655":
            
            MG1655_datasets = []
            
            dataset_metadata = file_metadata[file_metadata["dataset"] == dataset]
            
            treatment_list = dataset_metadata.treatment.unique().tolist()
        
            if "WT+ETOH" in treatment_list:
                treatment_list.append(treatment_list.pop(treatment_list.index('WT+ETOH')))
               
            for treatment in treatment_list:
                
                if treatment != "WT+ETOH":
                
                    treatment_metadata = dataset_metadata[dataset_metadata["treatment"] == treatment].sort_values(["experiment_date"])
                    untreatment_metadata = dataset_metadata[dataset_metadata["treatment"] == "WT+ETOH"].sort_values(["experiment_date"])
                    
                    untreated_experiment_dates = treatment_metadata["experiment_date"].unique().tolist()
                    treated_experiment_dates = treatment_metadata["experiment_date"].unique().tolist()
                    
                    for experiment_date in treated_experiment_dates:
                        
                        assert len(list(set(treated_experiment_dates) & set(untreated_experiment_dates))) == len(treated_experiment_dates)
                        
                        experiment_id = treated_experiment_dates.index(experiment_date) + 1
                        
                        treated_metadata = file_metadata[
                            (file_metadata["dataset"] == dataset) &
                            (file_metadata["treatment"] == treatment) &
                            (file_metadata["experiment_date"] == experiment_date)].copy()
                        
                        untreated_metadata = file_metadata[
                            (file_metadata["dataset"] == dataset) &
                            (file_metadata["treatment"] == "WT+ETOH") &
                            (file_metadata["experiment_date"] == experiment_date)].copy()
    
                        condition_dataset = pd.concat((treated_metadata,untreated_metadata))
                        condition_dataset["experiment_id"] = experiment_id
                        
                        condition_dataset["condition"] = str(["WT+ETOH",treatment])
                        
                        # print(treatment,experiment_date,experiment_id,len(treated_metadata),len(untreated_metadata), len(treatment_dataset))
                        
                        MG1655_datasets.append(condition_dataset)
                    
            MG1655_datasets = pd.concat(MG1655_datasets)
            file_metadata = pd.concat((file_metadata[file_metadata["dataset"]!="MG1655"],MG1655_datasets)).reset_index(drop=True)

        else:
            
            dataset_metadata = file_metadata[file_metadata["dataset"] == dataset]
            
            metadata_groups = dataset_metadata[["species_id", "treatment_concentration"]].drop_duplicates()
            
            for i in range(len(metadata_groups)):
                
                metadata_group = metadata_groups.iloc[i].to_dict()
                
                metadata_subset = dataset_metadata.copy()
                
                for key,value in metadata_group.items():
                    
                    metadata_subset = metadata_subset[metadata_subset[key] == value]
                    
                experiment_dates = sorted(metadata_subset["experiment_date"].unique().tolist())
                
                for experiment_date in experiment_dates:
                    
                    experiment_id = experiment_dates.index(experiment_date) + 1
                
                    file_metadata.loc[(file_metadata["dataset"] == dataset) &
                                      (file_metadata["species_id"] == metadata_group["species_id"]) &
                                      (file_metadata["treatment_concentration"] == metadata_group["treatment_concentration"]) &
                                      (file_metadata["experiment_date"] == experiment_date), "experiment_id"] = experiment_id
                    
                    file_metadata.loc[(file_metadata["dataset"] == dataset) &
                                      (file_metadata["species_id"] == metadata_group["species_id"]) &
                                      (file_metadata["treatment_concentration"] == metadata_group["treatment_concentration"]) &
                                      (file_metadata["experiment_date"] == experiment_date), "condition"] = "['WT+ETOH','CIP+ETOH']"
                    
    return file_metadata
        

def cache_images(file_metadata, mode, dataset, class_labels, class_list = [],  image_import_limit = 'None',
                 mask_background = False, normalise = True, align = True, 
                 stats=True, cell_image_size = (64,64),
                 image_channels = ["Nile Red", "DAPI"], channel_last=False):
    
    data = file_metadata[file_metadata["dataset"] == dataset]
    
    if class_list != []:
        data = data[data[class_labels].isin(class_list)]
        
    data = data.copy().sample(frac=1, random_state=42).reset_index(drop=True)
    
    label_list = sorted(data[class_labels].unique().tolist())
    
    if "WT+ETOH" in label_list:
        label_list.insert(0, label_list.pop(label_list.index("WT+ETOH")))
    
    data = [data.iloc[i] for i in range(len(data))]
    
    if image_import_limit != 'None':
        data = data[:int(image_import_limit)]
    
    print(f"Loading {len(data)} images/masks from dataset '{dataset}' with labels {label_list}...\n")
    
    if mode == "images":
    
        with Pool(10) as pool:
            
            results = tqdm.tqdm(pool.imap(partial(
                load_images,
                class_labels = class_labels,
                label_list = label_list, 
                mask_background = mask_background,
                normalise = normalise,
                align = align,
                channel_last = channel_last),
                data),
                total = len(data),
                disable=True)
            
            pool.close()
            pool.join()
        
        images, masks, file_names, labels, label_names, condition, experiment_ids, species_ids, treatment_concentrations = zip(*results)
        
        cached_data = dict(condition = list(condition),
                           experiment_ids = list(experiment_ids),
                           images = list(images),
                           masks = list(masks),
                           labels = list(labels),
                           label_names = list(label_names),
                           file_names = list(file_names),
                           label_list = list(label_list),
                           species_ids = list(species_ids),
                           treatment_concentrations = list(treatment_concentrations))
    
    else:
        
        with Pool(10) as pool:
            
            results = tqdm.tqdm(pool.imap(partial(
                load_cell_images,
                class_labels = class_labels,
                label_list = label_list, 
                mask_background = mask_background,
                normalise = normalise,
                align = align,
                stats = stats,
                image_size=cell_image_size,
                channel_last=channel_last,
                ),
                data),
                total = len(data),
                disable=True)
            
            pool.close()
            pool.join()
            
        results = list(results)
        
        random.seed(42) 
        random.shuffle(results)
        
        images, masks, labels, label_names, file_names, condition, experiment_ids, species_ids, treatment_concentrations, statistics = zip(*results)
        
        images = [item for sublist in images for item in sublist]
        masks = [item for sublist in masks for item in sublist]
        labels = [item for sublist in labels for item in sublist]
        label_names = [item for sublist in label_names for item in sublist]
        file_names = [item for sublist in file_names for item in sublist]
        experiment_ids = [item for sublist in experiment_ids for item in sublist]
        species_ids = [item for sublist in species_ids for item in sublist]
        treatment_concentrations = [item for sublist in treatment_concentrations for item in sublist]
        statistics = [item for sublist in statistics for item in sublist]
        condition = [item for sublist in condition for item in sublist]
        
        cached_data = dict(condition = list(condition),
                           experiment_ids = list(experiment_ids),
                            images = list(images),
                            masks = list(masks),
                            labels = list(labels),
                            label_names = list(label_names),
                            file_names = list(file_names),
                            statistics = list(statistics),
                            label_list = list(label_list),
                            species_ids = list(species_ids),
                            treatment_concentrations = list(treatment_concentrations))
        
    return cached_data



def extract_bboxes(mask):
    
    bboxes = []
    
    mask_ids = np.unique(mask)
    
    for mask_id in mask_ids:
        
        if mask_id != 0:
            
            # Use np.where to find the y, x indexes of the pixels that belong to the mask
            y_indexes, x_indexes = np.where(mask==mask_id)
            
            # If the mask is empty, return an empty bounding box
            if y_indexes.shape[0] == 0 or x_indexes.shape[0] == 0:
                return np.array([0, 0, 0, 0])
            
            # Find the min and max y, x indexes to form the bounding box
            y1, y2 = y_indexes.min(), y_indexes.max()
            x1, x2 = x_indexes.min(), x_indexes.max()
            
            bbox = np.array([y1, x1, y2, x2])
            
            bboxes.append(bbox)
            
    if bboxes == []:
        bboxes = [np.array([0,0,0,0])]
            
    return bboxes




def get_pipeline_data(images, masks, labels = [], mask_background=True,
                            image_size = (64,64), colicoords=False, 
                            resize=True):
    
    if len(labels) != len(images):
        labels = [None]*len(images)
    
    pipeline_data = []
    
    for image, mask, label in zip(images, masks, labels):
        
        cell_images, cell_masks, cell_contours = convert_image_to_cells(
            image, mask,
            mask_background=mask_background, 
            image_size=image_size,
            colicoords=colicoords, 
            resize=resize
            )
    
        image_data = dict(image=image,
                          mask=mask,
                          cell_images = cell_images,
                          cell_masks = cell_masks,
                          cell_labels = [label]*len(cell_images),
                          cell_contours = cell_contours)
        
        pipeline_data.append(image_data)
        
    return pipeline_data




def convert_image_to_cells(image, mask, mask_background=True, mask_mode = "predicted",
                            image_size = (64,64), colicoords=False, resize=True,  normalise=True):
    
    mask_ids = np.unique(mask)
    
    cell_images = []
    cell_masks = []
    cell_contours = []
        
    if len(mask_ids) > 1:

        for mask_id in mask_ids:
            
            try:

                if mask_ids[mask_id] != 0:
    
                    cell_mask = np.zeros(mask.shape, dtype=np.uint8)
    
                    cell_mask[mask==mask_id] = 255
    
                    contours, hierarchy = cv2.findContours(cell_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                    
                    cnt = contours[0]
                    M = cv2.moments(cnt)	
                    
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
    
                    x,y,w,h = cv2.boundingRect(cnt)
                    y1,y2,x1,x2 = y,(y+h),x,(x+w)
                    
                    edge = False
                    
                    if y1 - 1 < 0:
                        edge = True
                    if y2 + 1 > cell_mask.shape[0]:
                        edge = True
                    if x1 - 1 < 0:
                        edge = True
                    if x2 + 1 > cell_mask.shape[1]:
                        edge = True
                    
                    if edge != True:

                        cell_mask_crop = cell_mask[y1:y2,x1:x2]
                        cell_image_crop = image[:,y1:y2,x1:x2].copy()
                        
                        if mask_background:
                            cell_image_crop[:,cell_mask_crop==0] = 0
                        
                        cell_image_crop = resize_image(image_size, h, w, cell_image_crop, colicoords, resize)
                        cell_mask_crop = resize_image(image_size, h, w, cell_mask_crop, colicoords, resize)
                        
                        # if normalise:
                        #     cell_image_crop = normalize99(cell_image_crop)
                        
                        if (np.max(cell_image_crop) - np.min(cell_image_crop)) > 0:
    
                            cell_image_crop = rescale01(cell_image_crop)
                            cell_image_crop = cell_image_crop.astype(np.float32)
                        
                            cell_images.append(cell_image_crop)
                            cell_masks.append(cell_mask_crop)
                            cell_contours.append(cnt)

            except:
                pass
                        
    return cell_images, cell_masks, cell_contours
    
    
    
    






    
def convert_to_cell_dataset(image_dataset, mask_background=True, mask_mode = "predicted",
                            image_size = (64,64), colicoords=False, resize=True):
    
    cell_dataset = {}
    
    label_list = image_dataset["label_list"]
    label_list = [str(dat) for dat in label_list]
    
    for key in image_dataset.keys():
        if key != "pred_masks":
            cell_dataset[key] = []
    
    for i in range(len(image_dataset["images"])):
        
        if mask_mode == "predicted":
            mask = image_dataset["pred_masks"][i]
        else:
            mask = image_dataset["masks"][i]
            
        image = image_dataset["images"][i]
        
        mask_ids = np.unique(mask)
        
        if len(mask_ids) > 1:
    
            for mask_id in mask_ids:
                
                try:
    
                    if mask_ids[mask_id] != 0:
        
                        cell_mask = np.zeros(mask.shape, dtype=np.uint8)
        
                        cell_mask[mask==mask_id] = 1
        
                        contours, hierarchy = cv2.findContours(cell_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        
                        cnt = contours[0]
        
                        x,y,w,h = cv2.boundingRect(cnt)
                        y1,y2,x1,x2 = y,(y+h),x,(x+w)
                        
                        edge = False
                        
                        if y1 - 1 < 0:
                            edge = True
                        if y2 + 1 > cell_mask.shape[0]:
                            edge = True
                        if x1 - 1 < 0:
                            edge = True
                        if x2 + 1 > cell_mask.shape[1]:
                            edge = True
                        
                        if edge != True:
                            
                            cell_mask_crop = cell_mask[y1:y2,x1:x2]
            
                            cell_image_crop = image[:,y1:y2,x1:x2].copy()
                            
                            if mask_background:
                                cell_image_crop[:,cell_mask_crop==0] = 0
        
                            cell_image_crop = resize_image(image_size, h, w, cell_image_crop, colicoords, resize)
                            cell_mask_crop = resize_image(image_size, h, w, cell_mask_crop, colicoords, resize)
                            
                            cell_dataset["images"].append(cell_image_crop)
                            cell_dataset["masks"].append(cell_mask_crop)
                        
                            for key in image_dataset.keys():
                                if key not in ["images","masks"]:
                                    cell_dataset[key].append(image_dataset[key][i])
                                
                except:
                    pass

    cell_dataset["label_list"] = label_list

    return cell_dataset






def load_cell_images(dat, class_labels, label_list, stats = True, mask_background = False,
                     normalise = True, resize=False, align=True,
                     image_import_limit="None", image_channels = ["Nile Red", "DAPI"],
                     image_size = (64,64), colicoords=False, channel_last = False):
    
    try:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            file_name = dat["file_name"]
            
            label_names = dat[class_labels]
            label = label_list.index(label_names)
            
            experiment_id = dat["experiment_id"]
            condition = dat["condition"]
            species_id = dat["species_id"]
            treatment_concentration = dat["treatment_concentration"]
            
            image_path = pathlib.Path(dat["path"])
            

            dataset_index = pathlib.Path(image_path).parts.index(dat["dataset"])
            
            mask_path = (*image_path.parts[:dataset_index + 1], "All_segmentations", *image_path.parts[dataset_index + 2:])
            mask_path = pathlib.Path('').joinpath(*mask_path)
            
            image = tifffile.imread(image_path)
            
            if align == True:
                image = align_images(image)
    
            if os.path.exists(mask_path):
                mask = tifffile.imread(mask_path)
            else:
                mask = np.zeros(image[0].shape)
                
            cell_images = []
            cell_masks = []
            cell_labels = []
            cell_label_names = []
            cell_condition = []
            cell_experiment = []
            cell_file_names = []
            cell_mask_id = []
            cell_statistics = []
            cell_species_id = []
            cell_treatment_concentration = []
            
            mask_ids = np.unique(mask)
            
            if len(mask_ids) > 1:
        
                if image_import_limit == 'None' or image_import_limit > len(mask_ids):
                    image_import_limit = len(mask_ids)
                else:
                    image_import_limit = int(image_import_limit)
        
                for i in range(image_import_limit):
                    
                    try:
        
                        if mask_ids[i] != 0:
            
                            cell_mask = np.zeros(mask.shape, dtype=np.uint8)
            
                            cell_mask[mask==mask_ids[i]] = 1
            
                            contours, hierarchy = cv2.findContours(cell_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            
                            cnt = contours[0]
            
                            x,y,w,h = cv2.boundingRect(cnt)
                            y1,y2,x1,x2 = y,(y+h),x,(x+w)
                            
                            edge = False
                            
                            if y1 - 1 < 0:
                                edge = True
                            if y2 + 1 > cell_mask.shape[0]:
                                edge = True
                            if x1 - 1 < 0:
                                edge = True
                            if x2 + 1 > cell_mask.shape[1]:
                                edge = True
                            
                            if edge != True:
                        
                                cell_mask_crop = cell_mask[y1:y2,x1:x2]
                
                                cell_image_crop = image[:,y1:y2,x1:x2].copy()
                                
                                if stats == True:
                                    cell_stats = get_stats(image_channels, image, mask, cell_mask, cell_image_crop, cell_mask_crop, cnt)
                                else:
                                    cell_stats = []
                
                                if mask_background:
                                    cell_image_crop[:,cell_mask_crop==0] = 0
            
                                cell_image_crop = resize_image(image_size, h, w, cell_image_crop, colicoords, resize)
                                cell_mask_crop = resize_image(image_size, h, w, cell_mask_crop, colicoords, resize)
            
                                if normalise:
                                    cell_image_crop = normalize99(cell_image_crop)
            
                                if (np.max(cell_image_crop) - np.min(cell_image_crop)) > 0:
                                    
                                    if channel_last:
                                        cell_image_crop = np.swapaxes(cell_image_crop, 0, -1)
                                        
            
                                    cell_image_crop = rescale01(cell_image_crop)
                                    cell_image_crop = cell_image_crop.astype(np.float32)
            
                                    cell_label_names.append(label_names)
                                    cell_images.append(cell_image_crop)
                                    cell_masks.append(cell_mask_crop)
                                    cell_labels.append(label)
                                    cell_species_id.append(species_id)
                                    cell_treatment_concentration.append(treatment_concentration)
            
                                    cell_file_names.append(file_name)
                                    cell_experiment.append(experiment_id)
                                    cell_mask_id.append(mask_ids[i])
                                    cell_statistics.append(cell_stats)
                                    cell_condition.append(condition)
                                    
                    except:
                        print(traceback.format_exc())
                        pass
    
    except:
        print(traceback.format_exc())
        cell_images, cell_masks, cell_labels, cell_label_names, cell_file_names, cell_condition, cell_experiment, cell_species_id, cell_treatment_concentration, cell_statistics = [],[],[],[],[],[],[],[],[],[]

    return cell_images, cell_masks, cell_labels, cell_label_names, cell_file_names, cell_condition, cell_experiment, cell_species_id, cell_treatment_concentration, cell_statistics


def load_images(dat, class_labels, label_list, mask_background = False,
                normalise = True, resize=False, align=True, channel_last=False):

    try:
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            file_name = dat["file_name"]
            
            label_names = dat[class_labels]
            label = label_list.index(label_names)
            
            condition = dat["condition"]
            experiment_id = dat["experiment_id"]
            species_id = dat["species_id"]
            treatment_concentration = dat["treatment_concentration"]
            
            image_path = pathlib.Path(dat["path"])
            
            dataset_index = pathlib.Path(image_path).parts.index(dat["dataset"])
            
            mask_path = (*image_path.parts[:dataset_index + 1], "All_segmentations", *image_path.parts[dataset_index + 2:])
            mask_path = pathlib.Path('').joinpath(*mask_path)
            
            image = tifffile.imread(image_path)
            
            if align == True:
                image = align_images(image)
    
            if os.path.exists(mask_path):
                mask = tifffile.imread(mask_path)
            else:
                mask = np.zeros(image[0].shape)
                
            if mask_background:
                image[:,mask==0] = 0

            if normalise:
                image = normalize99(image)
                
            image = rescale01(image)
            
            if channel_last:
                image = np.swapaxes(image, 0, -1)
                
            image = image.astype(np.float32)
            mask = mask.astype(np.int32)

    except:
        pass
    
    return image, mask, file_name, label, label_names, condition, experiment_id, species_id, treatment_concentration


def balance_dataset(dataset):

    data_sort = dataset["labels"]
    unique, counts = np.unique(data_sort, return_counts=True)

    max_count = np.min(counts)

    balanced_dataset = {}

    for unique_key in unique:

        unique_indices = np.argwhere(np.array(data_sort) == unique_key)[:,0].tolist()[:max_count]

        for key, value in dataset.items():

            if key not in balanced_dataset.keys():
                    balanced_dataset[key] = []

            balanced_dataset[key].extend([value[index] for index in unique_indices])

    return balanced_dataset


def shuffle_train_data(train_data):
      
    dict_names = list(train_data.keys())     
    dict_values = list(zip(*[value for key,value in train_data.items()]))
    
    random.shuffle(dict_values)
    
    dict_values = list(zip(*dict_values))
    
    train_data = {key:list(dict_values[index]) for index,key in enumerate(train_data.keys())}
    
    return train_data
                    

def limit_train_data(train_data, num_files):
    
    for key,value in train_data.items():
        
        train_data[key] = value[:num_files]
        
    return train_data


def get_training_data(cached_data, mode = "", test_experiment = 1,
                      dataset_ratios = [0.7, 0.2, 0.1], label_limit = "None",
                      shuffle = True,  balance = False):

    experiment_ids = np.array(cached_data["experiment_ids"])
    label_list = cached_data.pop("label_list")
    
    train_indcies_list = []
    val_indices_list = []
    test_indcies_list = []
    
    train_data = {}
    val_data = {}
    test_data = {}

    if mode == "experiment_id":
    
        data_sort = pd.DataFrame(cached_data).drop(labels = ["images","masks"], axis=1)
        data_sort = data_sort.groupby(["labels"])
        
        for i in range(len(data_sort)):
        
            data = data_sort.get_group(list(data_sort.groups)[i])
            
            train_indices = data[data["experiment_ids"] != test_experiment].index.values.tolist()
            test_indices = data[data["experiment_ids"] == test_experiment].index.values.tolist()
            
            train_size = dataset_ratios[0] / sum(dataset_ratios[:2])
            
            train_indices, val_indices = train_test_split(train_indices,
                                                          train_size=train_size,
                                                          random_state=42,
                                                          shuffle=True)
            
            if label_limit != "None":
                train_indices = train_indices[:label_limit]
                val_indices = val_indices[:label_limit]
                test_indices = test_indices[:label_limit]   
            
            train_indcies_list.extend(train_indices)
            val_indices_list.extend(val_indices)
            test_indcies_list.extend(test_indices)
            
    else:
        

        data_sort = pd.DataFrame(cached_data).drop(labels = ["images","masks"], axis=1)
        data_sort = data_sort.groupby(["labels"])
        
        for i in range(len(data_sort)):
            
            data = data_sort.get_group(list(data_sort.groups)[i])
            
            dataset_indices = np.arange(len(data))
            
            
            train_indices, test_indices = train_test_split(dataset_indices,
                                                           train_size = 1 -dataset_ratios[-1],
                                                           random_state=42,
                                                           shuffle=True)
            
            train_size = dataset_ratios[0] / sum(dataset_ratios[:2])
        

            train_indices, val_indices = train_test_split(train_indices,
                                                          train_size=train_size,
                                                          random_state=42,
                                                          shuffle=True)
            
            if label_limit != "None":
                train_indices = train_indices[:label_limit]
                val_indices = val_indices[:label_limit]
                test_indices = test_indices[:label_limit]    
            
            
            train_indcies_list.extend(train_indices)
            val_indices_list.extend(val_indices)
            test_indcies_list.extend(test_indices)

        
        
    print(f"train size [{len(train_indcies_list)}], validation size [{len(val_indices_list)}], test size [{len(test_indcies_list)}]")
    
    
    for key,value in cached_data.items():

        if key in ["images","masks"]:
           
            train_dat = list(np.array(value)[train_indcies_list])
            val_dat = list(np.array(value)[val_indices_list])
            test_dat = list(np.array(value)[test_indcies_list])
            
        else:
           
            train_dat = np.take(np.array(value), train_indcies_list).tolist()
            val_dat = np.take(np.array(value), val_indices_list).tolist()
            test_dat = np.take(np.array(value), test_indcies_list).tolist()
         
        if label_limit != "None":
            
            train_dat = train_dat[:label_limit]
            val_dat = val_dat[:label_limit]
            test_dat = test_dat[:label_limit]    
              
        if key in train_data.keys():
      
          train_data[key].extend(train_dat)
          val_data[key].extend(val_dat)
          test_data[key].extend(test_dat)
  
        else:
          
          train_data[key] = train_dat
          val_data[key] = val_dat
          test_data[key] = test_dat

    if shuffle == True:
        train_data = shuffle_train_data(train_data)    
        val_data = shuffle_train_data(val_data)
        test_data = shuffle_train_data(test_data)

    if balance == True:
        train_data = balance_dataset(train_data)
        val_data = balance_dataset(val_data)
        test_data = balance_dataset(test_data)

    train_data["label_list"] = label_list
    val_data["label_list"] = label_list
    test_data["label_list"] = label_list

    return train_data, val_data, test_data  










