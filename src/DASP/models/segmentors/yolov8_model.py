import torch
import pickle
import os
from torch.utils import data
import torch.optim as optim
import tqdm
import numpy as np
from datetime import datetime
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from torch import nn
from torchvision.ops import box_iou
from torch.utils.tensorboard import SummaryWriter
from glob2 import glob
import pandas as pd
import cv2
import optuna
import traceback
from utils.dataloader import load_dataset
import time

from utils.metrics import calculate_average_precision
from ultralytics import YOLO
from utils.yolo_utils import export_yolo_dataset
import shutil
from utils.metrics import calculate_average_precision
from utils.file_io import extract_bboxes
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import balanced_accuracy_score, accuracy_score


class model_wrapper:

    def __init__(self,
                 train_data: dict = {},
                 val_data: dict = {},
                 epochs: int = 500,
                 batch_size: int = 9,
                 learning_rate: int = 0.0016,
                 model_path: str = "",
                 augment: bool = True,
                 tensorboard: bool = True,
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 gpu_int: int = 0,
                 verbose: bool = True,
                 yolo_model: str = "yolov8x-seg.pt",
                 single_class: bool = False,
                 model_dir: str = "",
                 train_mode: str = "holdout",
                 multiclass_mode: str = "binary",
                 ):
        
        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.augment = augment
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.tensorboard = tensorboard
        self.timestamp = timestamp
        self.gpu_int = gpu_int
        self.verbose = verbose
        self.yolo_model = yolo_model
        self.single_class = single_class
        self.model_dir = model_dir
        self.train_mode = train_mode
        self.multiclass_mode = multiclass_mode
        
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.learning_rate_values = []
        self.validation_iou = []
        self.freeze_mode_list = []
        self.epoch = 0
        self.training_loss_dict = {}
        self.validation_loss_dict = {}
        self.hyperparameter_tuning = False
        self.hyperparameter_study = []
        self.hyperparameter_plots = []

        
        if "label_list" in self.train_data.keys():
            self.label_list = self.train_data["label_list"]
        else:
            self.label_list = []

        if self.model_dir != "":
            
            self.model_dir = os.path.join(os.getcwd(),"models", self.model_dir + "_" + str(self.timestamp))
            
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
        else:
            self.model_dir = os.path.join(os.getcwd(),"models", "yolov8")
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        if self.model_path == "":
            
            self.train_condition = "[" + "+".join([dat.replace("+ETOH","") for dat in self.label_list]) + "]"
            
            self.model_name = f"DASP-segmentor-yolov8-{self.train_condition}-[{self.train_mode}]-[{self.multiclass_mode}]-{self.timestamp}"
            self.model_weights = os.path.join(self.model_dir, self.model_name + "_weights.pt")
            self.model_path = os.path.join(self.model_dir, self.model_name)
            
        self.yolo_folder = "yolo_model"
        
    def initialise_model(self, num_classes = ""):
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.device = torch.device(f'cuda:{self.gpu_int}')
            self.gpu = True
            self.gpu_mode = "GPU"
        else:
            self.device = torch.device('cpu')
            self.gpu = False
            self.gpu_mode = "CPU"

        custom_model = False
        
        if self.model_path != "":
            if os.path.exists(self.model_path):
                custom_model = True

        if type(num_classes) == int:
            self.num_classes = num_classes
        else:
            self.num_classes = len(np.unique(self.train_data["labels"]))
    
        custom_model = False
        
        if self.model_path != "":
            if os.path.exists(self.model_path) == True:
                
                if "_weights.pt" in self.model_path:
                    self.model_weights = self.model_path
                    self.model_path = self.model_path.replace("_weights.pt","")
                else:
                    self.model_weights = self.model_path + "_weights.pt"
                                
                self.model = YOLO(self.model_weights)
                
                self.model_name = os.path.basename(self.model_path)
                custom_model = True
                
        if custom_model == False:
            
            self.model = YOLO("yolov8x-seg.pt")
            
        if self.verbose:
            if custom_model:       
                print(f"loaded pretrained yolov8 segmentation model '{self.model_name}' on {self.gpu_mode}")
            else:
                print(f"loaded yolov8 segmentation model '{self.yolo_model}' on {self.gpu_mode}")

        self.bar_format = '{l_bar}{bar:2}{r_bar}{bar:-10b} [{remaining}]'
    
        (self.num_model_layers, self.num_model_parameters, _,_) = self.model.info()
    
    def create_yolo_dataset(self, generate_dataset = True):
        
        yolo_dataset = {}
        
        if self.train_data !={}:
            yolo_dataset["train"] = self.train_data
            self.num_train_images = len(self.train_data["images"])
        if self.val_data != {}:
            yolo_dataset["val"] = self.val_data
            self.yolo_validation = True
            self.num_validation_images = len(self.val_data["images"])
        else:
            self.yolo_validation = False
            self.num_validation_images = 0
            
        if generate_dataset:
            
            if self.verbose: 
                print("Generating Yolo dataset on disk...\n")
            
            self.yaml_path = export_yolo_dataset(yolo_dataset, yolo_dir="yolo_dataset")
        else:
            self.yaml_path = "yolo_dataset/data.yaml"
            

    def create_tune_dataset(self, num_images = 100):  
      
        if self.verbose: 
            print("Generating Yolo dataset on disk...\n")
        
        yolo_tune_dataset = {"train":{},"val":{}}
        
        for key, value in self.train_data.items():
            yolo_tune_dataset["train"][key] = value[:num_images]
        for key, value in self.val_data.items():
            yolo_tune_dataset["val"][key] = value[:5]
                
        if not os.path.exists("yolo_tune_dataset"):
            os.makedirs("yolo_tune_dataset")
    
        self.tune_yaml_path = export_yolo_dataset(yolo_tune_dataset, yolo_dir="yolo_tune_dataset")

        
    def optuna_objective(self, trial = None, num_images = 100):
    
        os.environ["YOLO_VERBOSE"] = "False"
        
        torch.cuda.empty_cache()
        
        batch_size = trial.suggest_int("batch_size", 2, 20, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        
        self.initialise_model()

        results = self.model.train(
                batch=batch_size,
                lr0=learning_rate,
                device=str(self.gpu_int),
                data=self.tune_yaml_path,
                epochs=self.num_tune_epochs,
                warmup_epochs = 0,
                imgsz=420,
                project = "yolo_tune_results",
                cache = False,
                single_cls = self.single_class,
                verbose = False,
                save=False,
                val = False,
            )
        
        results_path = os.path.join("yolo_tune_results","train","results.csv")
        results = pd.read_csv(results_path)
        
        loss = []
        
        for col in results.columns:
            if "train/" in col:
                loss.append(results[col].tolist()[-1])
                
        return np.sum(loss)
        
        
    def tune_hyperparameters(self, num_trials=5, num_images = 500, num_epochs = 1):
        
        if os.path.exists("yolo_tune_dataset"):
            shutil.rmtree("yolo_tune_dataset")
        if os.path.exists("yolo_tune_results"):
            shutil.rmtree("yolo_tune_results")
        
       
        self.initialise_model()
        self.create_tune_dataset(num_images = num_images)
           
        self.num_tune_epochs = num_epochs
       
        study = optuna.create_study(direction='minimize')
        study.optimize(self.optuna_objective, n_trials=num_trials)
        
        if self.verbose:
            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
    
            print("\nSetting hyperparameters to best values...\n")
            
        self.batch_size = int(trial.params["batch_size"])
        self.learning_rate = float(trial.params["learning_rate"])
        self.hyperparameter_tuning = False
        self.hyperparameter_study = study
   
    
    def generate_model_file(self):
        
        self.num_train_images = len(self.train_data["images"])
        self.num_validation_images = len(self.val_data["images"])
        
        yolo_dir = os.path.join(self.yolo_folder,"train")
        
        plot_paths = glob(yolo_dir+"*.png")
        
        model_data = {
            'num_epochs': self.epochs,
            'label_list': self.label_list,
            'num_train_images': self.num_train_images,
            'num_validation_images': self.num_validation_images,
            "batch_size": self.batch_size,
            "plots":{},
            "results":self.results,
            "start_time":self.start_time,
            "end_time":self.end_time,
            "training_speed": self.train_speed,
            "num_model_parameters": self.num_model_parameters,
            "hyperparameter_tuning": self.hyperparameter_tuning,
            "hyperparameter_study": self.hyperparameter_study,
            "hyperparameter_plots": self.hyperparameter_plots,
            }
        
        for path in plot_paths:
            
            plot_name = os.path.basename(path).replace(".png","")
            
            img = Image.open(path)
            img = np.array(img)
            
            model_data["plots"][plot_name] = img
            
        results = pd.read_csv(yolo_dir +"/results.csv")
        
        model_data["yolo_results"] = results
        
        for col in results.columns:
            
            if "train/" in col:
                
                if "training_dict" not in model_data.keys():
                    model_data["training_dict"] = {}
                    
                model_data["training_dict"][col.replace("train/","").strip()] = results[col].tolist()
            
            elif "val/" in col:
                
                if "validation_dict" not in model_data.keys():
                    model_data["validation_dict"] = {}
                    
                model_data["validation_dict"][col.replace("val/","").strip()] = results[col].tolist()     
                
            elif "metrics/" in col:
                
                if "yolo_metrics" not in model_data.keys():
                    model_data["yolo_metrics"] = {}
                    
                if "(B)" in col:
                    dict_name = "bbox " + col.replace("metrics/","").replace("(B)","").strip()
                if "(M)" in col:
                    dict_name = "mask " + col.replace("metrics/","").replace("(B)","").strip()
                        
                model_data["yolo_metrics"][dict_name] = results[col].tolist() 
                
            elif "lr/" in col:
                
                if "yolo_lr" not in model_data.keys():
                    model_data["yolo_lr"] = {}
                
                model_data["yolo_lr"][col.replace("lr/","").strip()] = results[col].tolist() 
                
            elif "epoch/" in col:
                
                model_data["num_epochs"] = len(results[col].tolist())
                
        torch.save(model_data, self.model_path)

        return model_data
    
    def export_model(self):
        
        print(f"saving model to {self.model_path}...")
        
        model_data = self.generate_model_file()
        
        best_model = os.path.join(self.yolo_folder,"train", "weights","best.pt")
        
        if os.path.exists(best_model):
            shutil.move(best_model, self.model_weights)
        
        return model_data
        
    def train(self, epochs = "", batch_size = "", single_class = "", generate_dataset = True):
        
        if os.path.exists(self.yolo_folder):
            shutil.rmtree(self.yolo_folder)
            os.mkdir(self.yolo_folder)
        
        if type(epochs) == int:
            self.epochs = epochs

        if type(batch_size) == int:
            self.batch_size = batch_size
            
        if type(self.single_class) == bool:
            self.single_class = single_class
            
            
        self.initialise_model()
        self.create_yolo_dataset(generate_dataset = generate_dataset)
        
        self.start_time = time.process_time()
        
        self.results = self.model.train(
                batch=self.batch_size,
                device=str(self.gpu_int),
                data=self.yaml_path,
                epochs=self.epochs,
                imgsz=420,
                val=True,
                project = self.yolo_folder,
                cache = True,
                single_cls = False,
            )
            
        self.end_time = time.process_time()
        self.train_duration = self.end_time - self.start_time
        self.train_speed = self.train_duration/self.epochs
        
        self.export_model()

        return 



    def read_reuslts(self):
        
              
        from ultralytics.yolo.utils.metrics import ConfusionMatrix
        
        model_data = torch.load(self.model_path)
        
        results = self.model.val(data=self.yaml_path)
        
        
        confmatrix = ConfusionMatrix(nc = results.box.nc, task="classify")
        
        eval_dict = dict( 
        precision = results.box.p,
        recall = results.box.r,
        f1 = results.box.f1,
        all_ap = results.box.all_ap,
        ap_class_index = results.box.ap_class_index,
        nc = results.box.nc,)
 
    def correct_predictions(self, label, pred_label):

        if len(label.shape) > 1:
            correct = (label.data.argmax(dim=1) == pred_label.data.argmax(dim=1)).float().sum().cpu()
        else:
            correct = (label.data == pred_label.data).float().sum().cpu()

        accuracy = correct / label.shape[0]

        return accuracy.numpy()


    def evaluate(self, test_data):
        
        self.initialise_model()
        self.create_yolo_dataset(generate_dataset = False)
        
        model_data = torch.load(self.model_path)
        
        test_image_path = os.path.join(self.yolo_folder, "test_images")
        image_path = os.path.join(test_image_path,"test.png")
        
        if not os.path.exists(test_image_path):
            os.makedirs(test_image_path)
        
        test_images = test_data["images"]
        test_masks = test_data["masks"]
        test_labels = test_data["labels"]
        
        batch_iter = tqdm.tqdm(enumerate(zip(test_images,test_labels, test_masks)), "Evaluating", total=len(test_images), position=0,leave=False)
        
        pred_mask_list = []
        true_mask_list = []
        true_labels_list = []
        pred_labels_list = []
        pred_confidences = []
        pred_boxes = []
        true_boxes = []
        
        torch_preds = []
        torch_target = []
        
        for i, (image, label, mask) in batch_iter:
            
            try:
                
                bboxes = extract_bboxes(mask)
            
                image = np.transpose(image, (1, 2, 0))
                plt.imsave(image_path, image, format='png')
                
                results = self.model.predict(source=image_path, device=str(self.gpu_int), verbose=False)
                
                xylist = results[0].masks.xy
                orig_shape = results[0].orig_shape
                boxes = results[0].boxes.xyxy
                
                pred_mask = np.zeros(results[0].orig_shape,dtype=np.uint16)
            
                pred_labels = results[0].boxes.cls.cpu().numpy().astype(int).tolist()
                true_labels = [label]*len(pred_labels)
                
                conf = results[0].boxes.conf.cpu().numpy().tolist()
            
                pred_labels_list.extend(pred_labels)
                true_labels_list.extend(true_labels)
                true_mask_list.append(mask)
                pred_confidences.extend(conf)
                
                contours = []
                
                cnt_int = 1
                
                for i, (cnt, box) in enumerate(zip(xylist,boxes)):
                    
                    if len(cnt) > 0:
                    
                        box = box.cpu().numpy().astype(int).tolist()
                    
                        pred_boxes.append(box)
                    
                        cnt = cnt.reshape(-1,1,2).astype(int)
                        cv2.drawContours(pred_mask, [cnt], -1, cnt_int, -1)

                        cnt_int += 1

                        
                pred_mask_list.append(pred_mask)
                
            except:
                pass

        evaluation_dict = {}

        ap_results = calculate_average_precision(true_mask_list, pred_mask_list)
        
        evaluation_dict = {**evaluation_dict, **ap_results}
        
        cm = confusion_matrix(true_labels_list, pred_labels_list, normalize='pred')
        cm_counts = confusion_matrix(true_labels_list, pred_labels_list,normalize=None)
        accuracy = accuracy_score(true_labels_list, pred_labels_list)
        balanced_accuracy = balanced_accuracy_score(true_labels_list, pred_labels_list)
        
        TN, FP, FN, TP = cm_counts.ravel()
        
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP) 
        accuracy = (TP + TN) / (TN + FP + FN + TP)
        
        
        evaluation_dict["confusion_matrix"] = cm
        evaluation_dict["confusion_matrix_counts"] = cm_counts
        
        evaluation_dict["TN"] = TN
        evaluation_dict["FP"] = FP
        evaluation_dict["FN"] = FN
        evaluation_dict["TP"] = TP
        evaluation_dict["sensitivity"] = sensitivity
        evaluation_dict["specificity"] = specificity
        evaluation_dict["accuracy"] = accuracy
        evaluation_dict["balanced_accuracy"] = balanced_accuracy
        

        class_accuracy = cm.diagonal()/cm.sum(axis=0)
        
        label_list = self.label_list
        label_list = [str(lab) for lab in label_list]
        
        classification_summary = classification_report(true_labels_list,
                                        pred_labels_list,
                                        target_names=label_list,
                                        digits = 5,
                                        output_dict=True)
        
        for i, label in enumerate(label_list):
            classification_summary[label]["accuracy"] = class_accuracy[i]
            
        accuracy = classification_summary.pop("accuracy")

        evaluation_dict["true_labels"] = true_labels_list
        evaluation_dict["pred_labels"] = pred_labels_list
        
        evaluation_dict["mean_pred_confidence"] = pred_confidences
        evaluation_dict["classification_summary"] = classification_summary
        
        model_data["evaluation_dict"] = evaluation_dict

        torch.save(model_data, self.model_path)
            
        return evaluation_dict, true_mask_list, pred_mask_list
       
    
    def segment(self, images):
        
        self.initialise_model(num_classes=2)
        
        model_data = torch.load(self.model_path)
        
        temp_image_path = os.path.join(self.yolo_folder, "segmentation_images")
        image_path = os.path.join(temp_image_path,"test.png")
        
        if not os.path.exists(temp_image_path):
            os.makedirs(temp_image_path)
        
        if self.verbose:
            silence_tqdm = False
        else:
            silence_tqdm = True
        
        pred_mask_list = []
        
        batch_iter = tqdm.tqdm(enumerate(images), "Segmenting", total=len(images), position=0, leave=False, disable=silence_tqdm)
        
        for i, image in batch_iter:
            
            try:
            
                image = np.transpose(image, (1, 2, 0))
                plt.imsave(image_path, image, format='png')
                
                results = self.model.predict(source=image_path, device=str(self.gpu_int), verbose=False)
                
                xylist = results[0].masks.xy
                orig_shape = results[0].orig_shape
                
                pred_mask = np.zeros(results[0].orig_shape,dtype=np.uint16)
                
                contours = []
                
                for i, cnt in enumerate(xylist):
                    
                    cnt = cnt.reshape(-1,1,2).astype(int)
                    
                    cv2.drawContours(pred_mask, [cnt], -1, i+1, -1)

            except:
                pass

            pred_mask_list.append(pred_mask)
        
        return pred_mask_list
        
        
                

            
            
            