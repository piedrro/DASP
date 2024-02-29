
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
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# from visualise import generate_plots, process_image, get_image_predictions, normalize99,rescale01
import pandas as pd
from utils.dataloader import load_dataset
from utils.visualise import normalize99,rescale01, process_image

import timm
import copy
import time


class model_wrapper:

    def __init__(self,
                 train_data: dict = {},
                 val_data: dict = {},
                 test_data: dict = {},
                 timm_model_backbone: str = "",
                 epochs: int = 10,
                 batch_size: int = 30,
                 learning_rate: int = 0.002,
                 model_path: str = "",
                 augment: bool = True,
                 tensorboard: bool = True,
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 optimizer_step_size: int = 25,
                 optimizer_gamma: int = 0.5,
                 train_mode: str = "ensemble",
                 verbose:bool = True,
                 dir_name:str = "",
                 gpu_int:int = 0,
                 optuna_path: str = "optuna_data.csv",
                 drop_rate: float = 0.1,
                 weight_decay = 1e-5,
                 test_experiment_id: int = -1,
                 ):

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.timm_model_backbone = timm_model_backbone
        self.epochs = epochs
        self.augment = augment
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.tensorboard = tensorboard
        self.timestamp = timestamp
        self.optimizer_step_size = optimizer_step_size
        self.optimizer_gamma = optimizer_gamma
        self.train_mode = train_mode
        self.verbose = verbose
        self.dir_name = dir_name
        self.gpu_int = gpu_int
        self.optuna_path = optuna_path
        self.drop_rate = drop_rate
        self.weight_decay = weight_decay
        self.test_experiment_id = test_experiment_id
        
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.learning_rate_values = []
        self.epoch = 0
        self.training_loss_dict = {}
        self.validation_loss_dict = {}
        self.test_loss_dict = {}
        self.fold_summary = {}
        self.fold_values = []
        self.timestamp_list = []

        self.hyperparameter_tuning = False
        self.hyperparameter_study = None
        self.hyperparameter_plots = None
        
        if "label_list" in self.train_data.keys():
            self.label_list = self.train_data["label_list"]
        else:
            self.label_list = []
            
        if self.model_path == "":
            
            if self.dir_name == "":
                self.dir_name = self.timm_model_backbone
            else:
                self.dir_name = self.dir_name + f"_{self.timestamp}" 
            
            self.model_dir = os.path.join(os.getcwd(),"models", self.dir_name)
            
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            
            self.train_condition = "[" + "+".join([dat.replace("+ETOH","") for dat in self.label_list]) + "]"
            
            if self.test_experiment_id == -1:
                self.model_name = f"DASP-classifier-[{self.timm_model_backbone}]-{self.train_condition}-[{self.train_mode}]-{self.timestamp}"
            else:
                self.model_name = f"DASP-classifier-[{self.timm_model_backbone}]-{self.train_condition}-[{self.train_mode}{self.test_experiment_id}]-{self.timestamp}"
            
            self.model_path = os.path.join(self.model_dir, self.model_name)


        self.kfold = False
        self.kfold_reset = False

    def initialise_model(self, num_classes = []):
        
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
                model_data = torch.load(self.model_path)
                
                self.timm_model_backbone = model_data["timm_model_backbone"]
                
                self.model = timm.create_model(self.timm_model_backbone,
                                               pretrained=False,
                                               num_classes=self.num_classes)
                
                self.model.load_state_dict(model_data['model_state_dict'])
                self.model_name = os.path.basename(self.model_path)
                custom_model = True
                
        if custom_model == False:
            
            pretrained_model_list = timm.list_models(pretrained=True)
            pretrained_model_list = [model.split(".")[0] for model in pretrained_model_list]

            if self.timm_model_backbone in pretrained_model_list:
                pretrained=True
            else:
                pretrained = False
            
            self.model = timm.create_model(self.timm_model_backbone,
                                           pretrained=pretrained,
                                           num_classes=self.num_classes, drop_rate=self.drop_rate)
            

        if self.verbose:
            if custom_model:       
                print(f"loaded pretrained timm classifier model '{self.model_name}' on {self.gpu_mode}{self.device}")
            else:
                print(f"loaded timm classifier model '{self.timm_model_backbone}' on {self.gpu_mode}{self.device}")

        self.model.to(self.device)
    
        params = [p for p in self.model.parameters() if p.requires_grad]
    
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=self.optimizer_step_size,
                                                            gamma=self.optimizer_gamma)
    
        self.bar_format = '{l_bar}{bar:2}{r_bar}{bar:-10b} [{remaining}]'
        
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def reset_model(self):
        
        pretrained_model_list = timm.list_models(pretrained=True)
        pretrained_model_list = [model.split(".")[0] for model in pretrained_model_list]

        if self.timm_model_backbone in pretrained_model_list:
            pretrained=True
        else:
            pretrained = False
        
        self.model = timm.create_model(self.timm_model_backbone,
                                       pretrained=pretrained,
                                       num_classes=self.num_classes)
    
    def filter_dataloader(self, dataset, experiment_id):
        
        filtered_dataset = {}
        
        experiment_ids = np.array(dataset["experiment_ids"])
        
        experiment_indices = np.argwhere(experiment_ids == experiment_id)[:,0].tolist()
        
        for key, value in dataset.items():
            if len(value) == len(experiment_ids):
                
                filtered_dataset[key] = []
                
                for index in experiment_indices:
                    filtered_dataset[key].append(value[index]) 
 
            else:
                filtered_dataset[key] = value
        
        # print(experiment_id, np.unique(filtered_dataset["experiment_ids"]))
        
        return filtered_dataset

    def initialise_evaluation_dataloader(self, dataset, augment=False):
        
        num_classes = len(dataset["label_list"])
        
        eval_dataset = load_dataset(images = dataset["images"],
                                    labels = dataset["labels"],
                                    num_classes=num_classes,
                                    augment = False,
                                    mode = "classifier")
  
        evaloader = data.DataLoader(dataset=eval_dataset,
                                    batch_size=100, 
                                    shuffle=False)
        
        return evaloader


    def visualise_augmentations(self,  n_examples = 1, save_plots=False, show_plots=True):
    
        model_dir = pathlib.Path(self.model_dir)
    
        self.initialise_dataloaders()
    
        for example_int in range(n_examples):
    
            from random import randint
            
            random_index = randint(0, len(self.train_data["images"])-1)
    
            img = self.train_data["images"][random_index]
            
            augmentation_images = [img]*100
            
            dataset = load_dataset(
                images=augmentation_images,
                labels=[0]*100,
                num_classes=2,
                augment=True,
                mode = "classifier",
                )
            
            loader = data.DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False)
   
            centre_image = np.swapaxes(img, 0, 2)
            centre_image = rescale01(centre_image)*255
            centre_image = centre_image.astype(np.uint8)
    
            augmented_images = []
    
            for img, label in loader:
                
                img = img[0].numpy()
    
                img = process_image(img)
    
                augmented_images.append(img)
    
            fig, ax = plt.subplots(5, 5, figsize=(10, 10))
            for i in range(5):
                for j in range(5):
                    if i ==2 and j == 2:
                        centre_image = cv2.copyMakeBorder(centre_image, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=[255,0,0])
                        ax[i,j].imshow(centre_image)
                    else:
                        ax[i, j].imshow(augmented_images[i*5+j])
                    ax[i, j].axis('off')
    
            fig.suptitle('Example Augmentations', fontsize=16)
            fig.tight_layout()
            plt.tight_layout()
    
            # if save_plots:
            #     plot_save_path = pathlib.Path('').joinpath(*model_dir.parts, "example_augmentations", f"example_augmentation{example_int}.tif")
            #     print(plot_save_path)
            #     if not os.path.exists(os.path.dirname(plot_save_path)):
            #         os.makedirs(os.path.dirname(plot_save_path))
            #     plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
    
            # if show_plots:
            #     plt.show()
            # plt.close()
            
    def initialise_dataloaders(self, experiment_id = None):
        
        num_classes = len(self.train_data["label_list"])
        
        if type(experiment_id) == int:
            train_dataloader = self.filter_dataloader(self.train_data, experiment_id)
            val_dataloader= self.filter_dataloader(self.val_data, experiment_id)
        else:
            train_dataloader = self.train_data
            val_dataloader= self.val_data
            
        test_dataloader = self.test_data
        
        self.train_dataset = load_dataset(images = train_dataloader["images"],
                                          labels = train_dataloader["labels"],
                                          num_classes = num_classes,
                                          augment = True,
                                          mode = "classifier")
        
        self.val_dataset = load_dataset(images = val_dataloader["images"],
                                          labels = val_dataloader["labels"],
                                          num_classes = num_classes,
                                          augment = False,
                                          mode = "classifier")

        self.trainloader = data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.batch_size,
                                            shuffle=True)
        
        self.valoader = data.DataLoader(dataset=self.val_dataset,
                                        batch_size=self.batch_size, 
                                        shuffle=False)
        
        self.num_train_batches = len(self.trainloader)
        self.num_validation_batches= len(self.valoader)

        self.num_train_images = len(train_dataloader["images"])
        self.num_validation_images = len(val_dataloader["images"])
        
        self.train_statisitcs = self.train_data["statistics"]
        self.val_statistics = self.val_data["statistics"]

        return self.trainloader, self.valoader

    def correct_predictions(self, label, pred_label):

        if len(label.shape) > 1:
            correct = (label.data.argmax(dim=1) == pred_label.data.argmax(dim=1)).float().sum().cpu()
        else:
            correct = (label.data == pred_label.data).float().sum().cpu()

        accuracy = correct / label.shape[0]

        return accuracy.numpy()


    def load_tune_dataset(self, num_images="", num_epochs = 1, augment=False, experiment_id=""):
        
        num_classes = len(np.unique(self.train_data["labels"]))

        tune_images = []
        tune_labels = []
        
        if type(experiment_id) == int:
            tune_dataset = self.filter_dataloader(self.train_data, experiment_id)
        else:
            tune_dataset = self.train_data

        if type(num_images) != int:
            num_images = len(tune_dataset["images"])

        self.tune_train_dataset = load_dataset(images=tune_dataset["images"][:num_images],
                                          labels=tune_dataset["labels"][:num_images],
                                          num_classes=num_classes,
                                          augment=augment,
                                          mode = "classifier")
        
    def optuna_objective(self, trial):

        batch_size = trial.suggest_int("batch_size", 10, 200, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)

        tune_trainloader = data.DataLoader(dataset=self.tune_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=learning_rate, 
                                    weight_decay=self.weight_decay)

        model = copy.deepcopy(self.model)
        model.to(self.device)

        running_loss = []
        
        start_time = time.process_time()

        step = 0

        for epoch in range(self.num_tune_epochs):
            for i, (images, labels) in enumerate(tune_trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if not torch.isnan(images).any():
                    
                    self.optimizer.zero_grad()
                    
                    pred_label = model(images)
                    loss = self.criterion(pred_label, labels)
                    
                    running_loss.append(loss.item())
                    
                    mean_running_loss = np.mean(running_loss[-3:])
                                              
                    trial.report(np.mean(running_loss[-3:]), step=step)
                                 
                    step +=1
                    
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                    loss.backward()
                    self.optimizer.step()

        end_time = time.process_time()
        duration = end_time - start_time

        return mean_running_loss

    def tune_hyperparameters(self, num_trials=50, num_images = 'None', num_epochs = 1, augment=False):
        
        if self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.initialise_model()
        self.criterion = nn.CrossEntropyLoss()
        
        self.load_tune_dataset(num_images = num_images, num_epochs = num_epochs, augment = augment)

        self.num_tune_images = num_images
        self.num_tune_epochs = num_epochs
        
        pruner=optuna.pruners.MedianPruner(n_startup_trials=num_trials/4, n_warmup_steps=30, interval_steps=10)
        
        study = optuna.create_study(direction="minimize", pruner=pruner)
        study.optimize(self.optuna_objective, n_trials=num_trials)
        best_trial = study.best_trial
     
        if self.verbose:
            print("Best trial:")
            print("  Value: ", best_trial.value)
            print("  Params: ")
            for key, value in best_trial.params.items():
                print("    {}: {}".format(key, value))
    
            print("\nSetting hyperparameters to best values...\n")
            
        self.batch_size = int(best_trial.params["batch_size"])
        self.learning_rate = float(best_trial.params["learning_rate"])
        self.hyperparameter_tuning = False
        self.hyperparameter_study = study
        
        self.get_hyperperamter_study_plots(study)
        
        trial_data = dict(timestamp = self.timestamp,
                          model_name = self.timm_model_backbone,
                          train_mode = self.train_mode,
                          label_list = str(self.label_list),
                          num_trials = num_trials,
                          num_epochs = num_epochs,
                          num_images = num_images,
                          best_loss = best_trial.value,
                          best_duration = best_trial.value,
                          best_batch_size = best_trial.params["batch_size"],
                          best_learning_rate = best_trial.params["learning_rate"],
                          )
        trial_data = pd.DataFrame.from_dict([trial_data])
        
        if os.path.exists(self.optuna_path) == False:
            trial_data.to_csv(self.optuna_path, sep = ",", index=False)
        else:
            optuna_data = pd.read_csv(self.optuna_path, sep=",", index_col=None)
            optuna_data = pd.concat((optuna_data, trial_data))
            optuna_data.to_csv(self.optuna_path, sep = ",", index=False)

        return self.batch_size, self.learning_rate


    def tune_num_workers(self, num_images=100, augment=True):
        
        self.initialise_model()
        
        num_classes = len(np.unique(self.train_data["labels"]))

        tune_images = []
        tune_labels = []
        
        start_time = time.process_time()
        
        num_workers_list = np.arange(0,10).tolist()
        duration_list = []
        
        dataset = load_dataset(images=self.train_data["images"],
                                          labels=self.train_data["labels"],
                                          num_classes=num_classes,
                                          augment=augment,
                                          mode = "classifier")
        
        for num_workers in num_workers_list:
            
            start_time = time.process_time()
        
            dataloader = data.DataLoader(dataset=dataset,
                                         batch_size=300,
                                         shuffle=True,
                                         num_workers=num_workers)
            
            for i, (images, labels) in enumerate(dataloader):
                images, labels = images.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)
    
                # checks if any images contains a NaN
                if not torch.isnan(images).any():
    
                    self.optimizer.zero_grad()  # zerograd the parameters
                    pred_label = self.model(images)
            
            end_time = time.process_time()
            
            duration = end_time - start_time
            duration_list.append(duration)

        best_num_workers = num_workers_list[duration_list.index(np.min(duration_list))]
        

    def get_hyperperamter_study_plots(self, study, show = True):
        
        self.hyperparameter_plots = {}
        
        try:
        
            from PIL import Image
    
            from optuna.visualization import (plot_optimization_history, plot_slice,
                                              plot_parallel_coordinate, plot_contour, 
                                              plot_param_importances)
            
            plot_functions = {"plot_optimization_history":plot_optimization_history,
                              "plot_slice":plot_slice,
                              "plot_parallel_coordinate":plot_parallel_coordinate,
                              "plot_contour":plot_contour,
                              "plot_param_importances":plot_param_importances}
                
            
            plot_path = "temp.png"
            
            for plot_name, plot_function in plot_functions.items():
                plot_function(study).write_image(plot_path)
                
                plot_img = np.asarray(Image.open(plot_path))
                
                self.hyperparameter_plots[plot_name] = plot_img
                
                if show == True:
                    plt.imshow(plot_img)
                    plt.axis('off')
                    plt.show()
                    
        except:
            pass
        

    def train(self, kfold = "", epochs = None, learning_rate = None, batch_size = None, initialise_loggers = False):
        
        if type(learning_rate) == float:
            self.learning_rate = learning_rate

        if type(kfold) == bool:
            self.kfold = kfold

        if type(epochs) == int:
            self.epochs = epochs

        if type(batch_size) == int:
            self.batch_size = batch_size

        if self.verbose:
            silence_tqdm = False
        else:
            silence_tqdm = True

        if initialise_loggers == True:
            self.training_loss = []
            self.training_accuracy = []
            self.validation_loss = []
            self.validation_accuracy = []
            self.learning_rate_values = []
            self.epoch = 0
            self.training_loss_dict = {}
            self.validation_loss_dict = {}
            self.fold_summary = {}
            self.fold_values = []
            self.timestamp_list = []

        torch.cuda.empty_cache()
        
        self.initialise_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.learning_rate, 
                                    weight_decay=self.weight_decay)
        
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                            step_size=self.optimizer_step_size,
                                                            gamma=self.optimizer_gamma)

        self.criterion = nn.CrossEntropyLoss()

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir="runs/" + self.model_name)
        else:
            self.writer = False
        
        if self.kfold == True:
            
            fold_experiment_ids = np.unique(self.train_data["experiment_ids"])
            fold_experiment_ids = [int(id) for id in fold_experiment_ids]
            n_fold_cross_validation = True
        
        else:
            fold_experiment_ids = [None]

        best_model_state_dict = []

        self.start_time = time.process_time()

        for fold, experiment_id in enumerate(fold_experiment_ids):
            
            if self.kfold_reset:
                self.reset_model()
                
            if self.kfold == True:
                print(fold)
                self.initialise_dataloaders(experiment_id=experiment_id)
            else:
                self.initialise_dataloaders()
            
            
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.learning_rate,
                                        weight_decay=self.weight_decay)
            
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=self.optimizer_step_size,
                                                                gamma=self.optimizer_gamma)
        
            best_fold_model = []
        
            if self.verbose:
                print(f"\nTraining timm model:\n\
                      model = {self.timm_model_backbone} \n\
                      fold = {fold} \n\
                      experiment_id = {experiment_id} \n\
                      num_epochs = {self.epochs} \n\
                      learning_rate = {self.learning_rate} \n\
                      batch_size = {self.batch_size} \n\
                      num_train_images = {self.num_train_images} \n\
                      num_validation_images = {self.num_validation_images} \n\
                          ")
            
            progressbar = tqdm.tqdm(range(self.epochs), 'Progress', total=self.epochs, position=0, leave=True, bar_format=self.bar_format, disable=silence_tqdm)

            for i in progressbar:
                """Epoch counter"""
                self.epoch += 1  # epoch counter
                
                """Training block"""
                self.train_step(mode="train", silence_tqdm=silence_tqdm)

                """Validation block"""
                if self.valoader is not None:
                    self.train_step(mode="validation", silence_tqdm=silence_tqdm)

                self.learning_rate_values.append(self.optimizer.param_groups[0]['lr'])
                
                self.fold_values.append(fold)
                self.timestamp_list.append(datetime.now().strftime("%y%m%d_%H%M"))
                
                validation_loss = self.validation_loss_dict["loss_list"]
                fold_indices = np.argwhere(np.array(self.fold_values)==fold)[:,0].tolist()
                fold_validation_loss = list(np.take(validation_loss,fold_indices))
                
                if fold_validation_loss[-1] == np.min(fold_validation_loss):
                    self.model.eval()
                    best_fold_model = copy.deepcopy(self.model.state_dict())
                    
                if validation_loss[-1] == np.min(validation_loss):
                    self.model.eval()
                    best_model_state_dict = copy.deepcopy(self.model.state_dict())

                self.fold_summary[fold] = {"experiment_id":experiment_id,
                                            "num_train_images": self.num_train_images,
                                            "num_validation_images": self.num_validation_images,
                                            "model_state_dict": best_fold_model}

                self.end_time = time.process_time()
                self.train_duration = self.end_time - self.start_time
                self.train_speed = self.train_duration/self.epochs

                model_data = {
                    'epoch': self.epoch,
                    'num_epochs': self.epochs,
                    'model_state_dict': best_model_state_dict,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'learning_rate_values': self.learning_rate_values,
                    'label_list': self.label_list,
                    'fold_dict':self.fold_summary,
                    'num_train_images': self.num_train_images,
                    'num_validation_images': self.num_validation_images,
                    "training_dict": self.training_loss_dict,
                    "validation_dict": self.validation_loss_dict,
                    "hyperparameter_tuning": self.hyperparameter_tuning,
                    "hyperparameter_study": self.hyperparameter_study,
                    "hyperparameter_plots": self.hyperparameter_plots,
                    "timm_model_backbone" : self.timm_model_backbone,
                    "batch_size": self.batch_size,
                    "learning_rate":self.learning_rate,
                    "kfold": kfold,
                    "optimizer_step_size": self.optimizer_step_size,
                    "optimizer_gamma": self.optimizer_gamma,
                    "train_statisitcs" : self.train_statisitcs,
                    "val_statistics" : self.val_statistics,
                    "timestamp_list": self.timestamp_list,
                    "start_time":self.start_time,
                    "end_time":self.end_time,
                    "training_speed": self.train_speed,
                    "num_model_parameters": self.num_model_parameters,
                    }
                
                torch.save(model_data,self.model_path)
                
                if self.writer:
                    for key,value in self.training_loss_dict.items():
                        self.writer.add_scalar("train_" + key, value[-1], self.epoch)
                    for key,value in self.validation_loss_dict.items():
                        self.writer.add_scalar("val_" + key, value[-1], self.epoch)
                    self.writer.add_scalar("learning rate", self.learning_rate_values[-1], self.epoch)
                    self.writer.add_scalar("batch size", self.learning_rate_values[-1], self.batch_size)
                    self.writer.add_scalar("fold", fold, self.epoch)
        
                self.lr_scheduler.step()  # learning rate scheduler step
               
        progressbar.close() 
               
        if self.verbose:
            print(f"\n\ntimm model saved to: {self.model_path}\n\n")
        
        return self.model_path
        
    def train_step(self, mode = "train", silence_tqdm=False):
        
        if mode == "train":
            self.model.train()
            dataloader =self.trainloader
            pbar_mode = "Training"
            main_loss_dict = getattr(self, 'training_loss_dict')
        else:
            self.model.eval()
            dataloader = self.valoader
            pbar_mode = "Validation"
            main_loss_dict = getattr(self, 'validation_loss_dict')
        
        loss_dict = {"loss_list":[],"accuracy_list":[]}

        batch_iter = tqdm.tqdm(enumerate(dataloader), pbar_mode, total=len(dataloader), position=0,leave=False, bar_format=self.bar_format, disable=silence_tqdm)

        for i, (images, labels) in batch_iter:
            images, labels = images.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)

            # checks if any images contains a NaN
            if not torch.isnan(images).any():

                self.optimizer.zero_grad()  # zerograd the parameters
                pred_label = self.model(images)  # one forward pass

                loss = self.criterion(pred_label, labels)
                loss_dict["loss_list"].append(loss.item())

                accuracy = self.correct_predictions(pred_label, labels)
                loss_dict["accuracy_list"].append(accuracy)
            
                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters
                
                batch_iter.set_description(f'{pbar_mode}[{self.epoch}\\{self.epochs}]:(loss {np.mean(loss_dict["loss_list"]):.3f}, Acc {np.mean(loss_dict["accuracy_list"]):.2f}')  # update progressbar

        batch_iter.close()
        
        for key,value in loss_dict.items():
            if key not in main_loss_dict.keys():
                main_loss_dict[key] = []
            main_loss_dict[key].append(np.mean(value))

    def compare_models(self, state_dict1, state_dict2):
        
        models_differ = 0
        for key_item_1, key_item_2 in zip(state_dict1.items(), state_dict2.items()):
            if torch.equal(key_item_1[1], key_item_2[1]):
                pass
            else:
                models_differ += 1
                if (key_item_1[0] == key_item_2[0]):
                    print('Mismtach found at', key_item_1[0])
                else:
                    raise Exception
        if models_differ == 0:
            models_matched = True
        else:
            models_matched = False
        
        return models_matched

    def evaluate(self, model_path = "", fold = None, test_dataset = {}, holdout_dataset = {}, 
                 evalate_test_folds = False, evalate_holdout_folds = False):
        
        if model_path != "":
            self.model_path = model_path
            self.initialise_model()
            
        if hasattr(self, 'model') == False:
            self.initialise_model()
            
        model_data = torch.load(self.model_path)
        
        model_state_dict = model_data["model_state_dict"]
        self.model.load_state_dict(model_data['model_state_dict'])
        
        self.criterion = nn.CrossEntropyLoss()
        
        if test_dataset != {}:
            dataloader = self.initialise_evaluation_dataloader(test_dataset)
            evaluation_dict = self.evaluate_step(dataloader=dataloader)
            model_data["test_dict"] = evaluation_dict
            
            if evalate_test_folds:
                model_data = self.evalate_folds_step(
                    model_data,
                    dataloader, 
                    dict_name = "test_dict")
        
        if holdout_dataset != {}:
            dataloader = self.initialise_evaluation_dataloader(holdout_dataset)
            evaluation_dict = self.evaluate_step(dataloader=dataloader)
            model_data["holdout_dict"] = evaluation_dict
            
            if evalate_holdout_folds:
                model_data = self.evalate_folds_step(
                    model_data,
                    dataloader, 
                    dict_name = "holdout_dict"
                    )
        
        torch.save(model_data, self.model_path)
                
        return model_data 
    
    def evalate_folds_step(self, model_data, dataloader, dict_name = "test_dict"):
        
        if "fold_dict" in model_data.keys():
            
            fold_summary = []
            
            for fold, fold_dict in model_data['fold_dict'].items():
                
                fold_model_state_dict = fold_dict['model_state_dict']
                
                self.model.load_state_dict(fold_model_state_dict)
                
                evaluation_dict = self.evaluate_step(dataloader=dataloader, fold=fold)
                
                model_data['fold_dict'][fold][dict_name] = evaluation_dict
                
                fold_summary.append({"fold":fold,
                                      "loss":evaluation_dict["mean_loss"],
                                      "accuracy":evaluation_dict["mean_accuracy"],
                                      "balanced_accuracy":evaluation_dict["balanced_accuracy"]})
                
            fold_summary = pd.DataFrame.from_dict(fold_summary)
            model_data["fold_summary"] = fold_summary
            
        return model_data
            
                
    def evaluate_step(self, dataloader, fold = ""):
        
        self.model.eval()
        self.criterion = nn.CrossEntropyLoss()
        
        self.initialise_dataloaders()
        
        if self.verbose== True:
            silence_tqdm = False
        else:
            silence_tqdm = True
        
        evaluation_dict = {}
        
        if type(fold) == int:
            pbar_mode = f"Evaluating Fold {fold}"
        else:
            pbar_mode = "Evaluating Best Model"
        
        loss_dict = {
            "pred_confidence_list":[],
            "true_labels":[],
            "pred_labels":[],
            }
        
        batch_iter = tqdm.tqdm(enumerate(dataloader), pbar_mode, total=len(dataloader), position=0,leave=False, bar_format=self.bar_format, disable=silence_tqdm)
        
        for i, (image, label) in batch_iter:
            image, label = image.to(self.device), label.to(self.device)  # send to device (GPU or CPU)

            # checks if any images contains a NaN
            if not torch.isnan(image).any():
                
                self.optimizer.zero_grad()  # zerograd the parameters
                output = self.model(image)  # one forward pass
            
                accuracy = self.correct_predictions(output, label)
            
                loss = self.criterion(output, label)
                
                confidence, pred_label = torch.nn.functional.softmax(output, dim=1).max(dim=1)
                
                pred_confidence = confidence.detach().cpu().numpy().astype(float).tolist()
                pred_label = pred_label.detach().cpu().numpy().astype(int).tolist()
                true_label = label.detach().cpu().argmax(dim=1).numpy().astype(int).tolist()
                
                loss_dict["pred_confidence_list"].extend(pred_confidence)
                loss_dict["true_labels"].extend(true_label)
                loss_dict["pred_labels"].extend(pred_label)
                
        cm = confusion_matrix(loss_dict["true_labels"], loss_dict["pred_labels"], normalize='pred')
        cm_counts = confusion_matrix(loss_dict["true_labels"], loss_dict["pred_labels"],normalize=None)
        accuracy = accuracy_score(loss_dict["true_labels"], loss_dict["pred_labels"])
        balanced_accuracy = balanced_accuracy_score(loss_dict["true_labels"], loss_dict["pred_labels"])
        
        TN, FP, FN, TP = cm_counts.ravel()
        
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP) 
        accuracy = (TP + TN) / (TN + FP + FN + TP)


        for key,value in loss_dict.items():
            evaluation_dict[key] = value
            if key in ["pred_confidence_list"]:
                save_key = "mean_" + key.replace("_list","")
                if save_key not in evaluation_dict.keys():
                    evaluation_dict[save_key] = []
                evaluation_dict[save_key] = np.mean(value)
                
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
        
        evaluation_dict["true_labels"] = loss_dict["true_labels"]
        evaluation_dict["pred_labels"] = loss_dict["pred_labels"]
        
        return evaluation_dict
        
    def classify_and_evaluate(self, images = [], labels = []):
        
        num_classes = len(np.unique(labels))
        
        torch.cuda.empty_cache()
        self.initialise_model(num_classes=num_classes)
        
        self.model.eval()
        
        if self.verbose:
            self.silence_tqdm = False
        else:
            self.silence_tqdm = True
        
        batch_iter = tqdm.tqdm(enumerate(zip(images,labels)), 'Classifying', total=len(images), position=0,leave=False, bar_format=self.bar_format, disable=self.silence_tqdm)
        
        pred_label_list = []
        target_label_list = []
        
        for i, (img,label) in batch_iter:
            
            img = torch.as_tensor(img).to(self.device)
            
            img = torch.unsqueeze(img, 0)
            
            with torch.no_grad():
                
                pred_label = self.model(img)
                
                pred_label = pred_label.data.cpu().argmax().numpy().tolist()
                
                pred_label_list.append(pred_label)
                target_label_list.append(label)
             
                
        balanced_accuracy = balanced_accuracy_score(target_label_list, pred_label_list)
        
        return balanced_accuracy
        
    
    def classify_images(self, images = [], num_classes=2, augment=False):
        
        torch.cuda.empty_cache()
        self.initialise_model(num_classes=num_classes)
        
        self.model.eval()
        
        if self.verbose:
            self.silence_tqdm = False
        else:
            self.silence_tqdm = True
            
        classify_dataset = load_dataset(images = images,
                                        augment = augment,
                                        mode = "pipeline")
        
        classify_loader = data.DataLoader(
            dataset=classify_dataset,
            batch_size=100, 
            shuffle=False,
            )
        
        batch_iter = tqdm.tqdm(enumerate(classify_loader), 'Classifying', 
                               total=len(classify_loader),
                               position=0,leave=False, 
                               bar_format=self.bar_format, 
                               disable=self.silence_tqdm)
        
        pred_label_list = []
        pred_confidence_list = []

        for i, img in batch_iter:
            
            img = torch.as_tensor(img).to(self.device)

            with torch.no_grad():
                
                output = self.model(img)
                
                confidence, pred_label = torch.max(output, 1)
                
                confidence = confidence.cpu().numpy().astype(float).tolist()
                pred_label = pred_label.cpu().numpy().astype(int).tolist()
                    
                pred_label_list.extend(pred_label)
                pred_confidence_list.extend(confidence)
                
        return pred_label_list, pred_confidence_list


    def classify_pipeline(self, pipeline_data, num_classes=2, augment=False):
        
        torch.cuda.empty_cache()
        self.initialise_model(num_classes=num_classes)
        
        self.model.eval()
        
        if self.verbose:
            self.silence_tqdm = False
        else:
            self.silence_tqdm = True
        
        target_label_list = []
        pred_label_list = []
        pred_confidence_list = []
        
        batch_iter = tqdm.tqdm(enumerate(pipeline_data), 'Classifying', 
                               total=len(pipeline_data),
                               position=0,leave=False, 
                               bar_format=self.bar_format, 
                               disable=self.silence_tqdm)
    
        for i, image_data in batch_iter:
            
            classify_dataset = load_dataset(images = image_data["cell_images"],
                                            augment = augment,
                                            mode = "pipeline")
            
            classify_loader = data.DataLoader(
                dataset=classify_dataset,
                batch_size=100, 
                shuffle=False,
                )
            
            image_data["pred_confidences"] = []
            image_data["pred_labels"] = []
            
            target_label_list.extend(image_data["cell_labels"])
            
            for images in classify_loader:
                
                images = torch.as_tensor(images).to(self.device)
                
                with torch.no_grad():
                    
                    output = self.model(images)
                    
                    confidence, pred_label= torch.nn.functional.softmax(output, dim=1).max(dim=1)
                    
                    confidence = confidence.cpu().numpy().astype(float).tolist()
                    pred_label = pred_label.cpu().numpy().astype(int).tolist()
                    
                    image_data["pred_confidences"].extend(confidence)
                    image_data["pred_labels"].extend(pred_label)
                    
                    pred_label_list.extend(pred_label)
                    pred_confidence_list.extend(confidence)
                    
            pipeline_data[i] = image_data 
            

                
        cm = confusion_matrix(target_label_list, pred_label_list, normalize='pred')
        cm_counts = confusion_matrix(target_label_list, pred_label_list,normalize=None)
        accuracy = accuracy_score(target_label_list, pred_label_list)
        balanced_accuracy = balanced_accuracy_score(target_label_list, pred_label_list)
        
        TN, FP, FN, TP = cm_counts.ravel()
        
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP) 
        accuracy = (TP + TN) / (TN + FP + FN + TP)


        prediction_data = dict(target_labels = target_label_list,
                               pred_labels = pred_label_list,
                               pred_confidence = pred_confidence_list)


        prediction_data["confusion_matrix"] = cm
        prediction_data["confusion_matrix_counts"] = cm_counts
        
        prediction_data["TN"] = TN
        prediction_data["FP"] = FP
        prediction_data["FN"] = FN
        prediction_data["TP"] = TP
        prediction_data["sensitivity"] = sensitivity
        prediction_data["specificity"] = specificity
        prediction_data["accuracy"] = accuracy
        prediction_data["balanced_accuracy"] = balanced_accuracy

        return pipeline_data, prediction_data
                        

                
            
            
            










