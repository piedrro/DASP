import traceback
import seaborn as sns
import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from skimage import exposure
from datetime import datetime
import os
import pathlib
import itertools
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
# from visalize import generate_plots
import matplotlib.pyplot as plt
import shap
import copy
import warnings
warnings.filterwarnings("ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.")
import optuna
import io
from torch.utils import data
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import cv2
import tifffile
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

# from visualise import generate_plots, process_image, get_image_predictions, normalize99,rescale01
import pandas as pd
from utils.dataloader import load_dataset
from utils.visualise import normalize99,rescale01, process_image, plot_publication_confusion_matrix, plot_pipeline_detections, plot_pipeline_histogram
from utils.file_io import get_pipeline_data

import timm
import copy
import time
import warnings

class model_wrapper:

    def __init__(self,
                 images: list = [],
                 labels: list = [],
                 segmentor_path: str = "",
                 classifier_path: str = "",
                 augment: bool = True,
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 verbose:bool = True,
                 gpu_int:int = 0,
                 tensorflow_model:bool = False,
                 ):
        
        
        self.images = images
        self.lables = labels
        self.classifier_path = classifier_path
        self.segmentor_path = segmentor_path
        self.augment = augment
        self.timestamp = timestamp
        self.verbose = verbose
        self.gpu_int = gpu_int
        self.tensorflow_model = tensorflow_model
        
        self.initialise_segmentor()
        self.initialise_classifier()
        
        torch.cuda.set_device(self.gpu_int)
        
    def initialise_segmentor(self):
        
        if os.path.exists(self.segmentor_path) == False:
            
            print(f"Segmentor model does not exist")
            
        else:
            
            model_name = os.path.basename(self.segmentor_path)

            if "cellpose" in model_name.lower():
                from models.segmentors.cellpose_model import model_wrapper as segmentor_wrapper
            elif "maskrcnn" in model_name.lower() or "maskcnn" in model_name.lower():
                from models.segmentors.maskrcnn_model import model_wrapper as segmentor_wrapper
            elif "yolo" in model_name.lower():
                from models.segmentors.yolov8_model import model_wrapper as segmentor_wrapper

            if 'segmentor_wrapper' in locals():
                
                self.segmentor = segmentor_wrapper(
                    model_path=self.segmentor_path,
                    verbose=self.verbose,
                    gpu_int=self.gpu_int,
                    )
                
                
    def initialise_classifier(self):
        
        if os.path.exists(self.classifier_path) == False:
            
            print(f"Classifier model does not exist")
        
        else:
            
            if self.tensorflow_model == False:
            
                from models.classifiers.classifier_model import model_wrapper as classifier_wrapper
                    
                self.classifier = classifier_wrapper(
                    model_path = self.classifier_path,
                    verbose = self.verbose,
                    gpu_int=self.gpu_int,
                    )
                
            else:
                
                from models.classifiers.tensorflow_classifier import model_wrapper as classifier_wrapper
                
                self.classifier = classifier_wrapper(
                    model_path = self.classifier_path,
                    verbose = self.verbose,
                    gpu_int=self.gpu_int,
                    )
            
            
    def segment(self, images = [], labels = []):
        
        masks = self.segmentor.segment(images)
        
        pipeline_data = get_pipeline_data(images,
                                          masks, 
                                          labels)
        
        return pipeline_data
        

    def segment_and_classify(self, images = [], labels = [], label_names = [], train_mode = "",
                             condition = "", experiment_id = "", experiment_folder = "", 
                             clinical_isolate = "", concentration = "",
                             plot_cm = False, plot_histogram = False, plot_detections=False, augment=False):
        
        if experiment_folder != "":
            if os.path.exists(experiment_folder) == False:
                os.mkdir(experiment_folder)
        
        
        masks = self.segmentor.segment(images)
        
        pipeline_data = get_pipeline_data(images,
                                          masks, 
                                          labels)
        
        pipeline_data, prediction_data = self.classifier.classify_pipeline(pipeline_data, augment=augment)
        
        prediction_summary = dict(clinical_isolate=clinical_isolate,
                                  experiment_id=experiment_id,
                                  condition=condition,
                                  concentration=concentration,
                                  train_mode = train_mode,
                                  label_names=label_names)
        
        prediction_summary = {**prediction_summary, **prediction_data}
        
        if plot_detections:
            plot_pipeline_detections(
                pipeline_data,
                label_names,
                condition,
                experiment_folder,
                clinical_isolate,
                experiment_id,
                train_mode,
                )
            
        if len(labels) > 0:
            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                accuracy = accuracy_score(prediction_data["target_labels"],prediction_data["pred_labels"])
                balanced_accuracy = balanced_accuracy_score(prediction_data["target_labels"],prediction_data["pred_labels"])
            
            unique_labels, counts = np.unique(prediction_data["pred_labels"], return_counts=True)
            prediction_summary["accuracy"] = accuracy
            prediction_summary["balanced_accuracy"] = balanced_accuracy
            
            for label, count in zip(unique_labels, counts):
                prediction_summary[f"{label_names[label]}_classification_ratio"] = count/len(prediction_data["pred_labels"])
            
            prediction_summary["confusion_matrix"] = confusion_matrix(prediction_data["target_labels"],prediction_data["pred_labels"], normalize=None)
            prediction_summary["confusion_matrix_norm"] = confusion_matrix(prediction_data["target_labels"],prediction_data["pred_labels"], normalize="true")
            
            if plot_cm:
                plot_publication_confusion_matrix(
                    prediction_data["target_labels"],
                    prediction_data["pred_labels"],
                    label_names,
                    condition,
                    experiment_folder,
                    clinical_isolate,
                    )
                
            if plot_histogram:
                plot_pipeline_histogram(
                    prediction_data["pred_labels"],
                    np.array(prediction_data["pred_confidence"]),
                    label_names,
                    condition,
                    experiment_folder,
                    clinical_isolate,
                    experiment_id,
                    )
                
        return prediction_summary
            
            
            
            
            

            
            
        
        

        
        
        














