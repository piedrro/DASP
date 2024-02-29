import torch
import os
from cellpose import models,metrics
from cellpose import utils, models, io, dynamics
from cellpose.io import logger_setup
import logging
import shutil
from datetime import datetime
from utils.metrics import calculate_average_precision
import optuna
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time



class model_wrapper:

    def __init__(self,
                 train_data: dict = {},
                 val_data: dict = {},
                 epochs: int = 50,
                 batch_size: int = 5,
                 learning_rate: float = 0.0094,
                 diam_mean: int = 15,
                 model_type: str = "nuclei",
                 model_path: str = "",
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 verbose: bool = True,
                 gpu_int: int = 0,
                 ):
        

        self.train_data = train_data
        self.val_data = val_data
        self.epochs = epochs
        self.diam_mean = diam_mean
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model_type = model_type
        self.model_path = model_path
        self.timestamp = timestamp
        self.log_path = "cellpose_logger.log"
        self.verbose = verbose
        self.gpu_int = gpu_int
        
        
        self.hyperparameter_tuning = False
        self.hyperparameter_study = []
        self.hyperparameter_plots = []
        
        self.cellpose_folder = "cellpose_model"
        
        if not os.path.exists(self.cellpose_folder):
            os.makedirs(self.cellpose_folder)
    
        self.model_dir = os.path.join(os.getcwd(), "segmentors", "cellpose")
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.model_dir = os.path.join(os.getcwd(),"models", "cellpose")
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        if self.model_path == "":
            
            self.model_name = f"DASP-segmentor-cellpose-{self.timestamp}"
            self.model_path = os.path.join(self.model_dir, self.model_name)
            
            self.model_weights_path = self.model_path + "_weights"
            self.model_weights_name = self.model_name + "_weights"
        else:
            
            model_name = os.path.basename(self.model_path)
            
            if "_weights" in model_name:
                self.model_weights_name = model_name
                self.model_weights_path = self.model_path
                self.model_name = model_name.replace("_weights","")
                self.model_path = self.model_path.replace("_weights","")
            else:
                self.model_name = os.path.basename(self.model_path)
                self.model_weights_path = self.model_path + "_weights"
                self.model_weights_name = self.model_name + "_weights"
        
        if "label_list" in self.train_data.keys():
            self.label_list = self.train_data["label_list"]
        else:
            self.label_list = []

    def initialise_logger(self, silence_logger=False):
        
        if os.path.exists(self.log_path):
            os.remove(self.log_path)

        # from cellpose.io import logger_setup
        # logger_setup()

        logger = logging.getLogger('cellpose')
        
        if self.verbose == True and silence_logger == False:
            logger.propagate = True
        else:
            logger.propagate = False
            
        logger.setLevel(logging.INFO)
    
        # create file handler which logs even debug messages
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        
        
            
    def get_logger_data(self):
    
        with open(self.log_path, "r") as f:
            log = f.read()
        
        log_entries = log.split("\n")
        
        epoch_list = []
        train_loss_list = []
        val_loss_list = []
        
        for log_entry in log_entries:
            
            if "Epoch" in log_entry:
                epoch = int(log_entry.split("Epoch")[-1].split(",")[0].strip())
                epoch_list.append(epoch)

                if "Loss" in log_entry:
                    train_loss = float(log_entry.split("Loss")[1].split(",")[0].strip())
                    train_loss_list.append(train_loss)
                    
                if "Loss Test" in log_entry:
                    val_loss = float(log_entry.split("Loss Test")[-1].split(",")[0].strip())
                    val_loss_list.append(val_loss)

        return epoch_list, train_loss_list, val_loss_list
        
        
    def initialise_model(self):
        
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
                
        if custom_model == True:
            
            if self.verbose:
                print(f"Initialised custom Cellpose Model on {self.gpu_mode}: {os.path.basename(self.model_path)}")
            
            self.model = models.CellposeModel(
                pretrained_model=self.model_weights_path,
                diam_mean=self.diam_mean,
                model_type=None,
                gpu = self.gpu,
                device = self.device,
                net_avg=False,
                )
        else:
            
            if self.verbose:
                print(f"Initialised built-in Cellpose Model on {self.gpu_mode}: {self.model_type}")
            
            self.model = models.CellposeModel(
                diam_mean = self.diam_mean,
                model_type=self.model_type,
                gpu = self.gpu,
                device = self.device,
                net_avg = False,
                )
        
        self.num_model_parameters = sum(p.numel() for p in self.model.net.parameters() if p.requires_grad)
        
    def cellpose_weights_io(self, path, mode = "read"):
        
        if mode == "read":

            model_weights = torch.load(path)

        if mode == "write":
            
            model_data = torch.load(path)
            
            model_weights = None

            if type(model_data) == dict:
                if "model_state_dict" in model_data.keys():
                    
                    model_weights = model_data["model_state_dict"]
                    
                    model_dir = os.path.dirname(path)
                    model_name = os.path.basename(path) + "_weights"
                    
                    path = os.path.join(model_dir, model_name)
                    
                    torch.save(model_weights, path)
            
        return model_weights, path
                
    def generate_flows(self, masks):
        
        flows = dynamics.labels_to_flows(
            masks,
            use_gpu=self.gpu, 
            device=self.device, 
            redo_flows=False)
             
        return flows
        
    def tune_hyperparamters(self, num_trials=10, num_images = 1000, num_epochs = 3):
        
        self.initialise_model()
        
        self.tune_images = self.train_data["images"][:num_images]
        self.tune_masks = self.train_data["masks"][:num_images]
        
        self.tune_flows = self.generate_flows(self.tune_masks)
        self.tune_epochs = num_epochs
        
        study = optuna.create_study(direction='minimize')
        study.optimize(self.optuna_objective, n_trials=num_trials)
        
        trial = study.best_trial
        
        if self.verbose:
            print("Best trial:")
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print("    {}: {}".format(key, value))
    
            print("\nSetting hyperparameters to best values...\n")
            
        self.batch_size = int(trial.params["batch_size"])
        self.learning_rate = float(trial.params["learning_rate"])
        self.hyperparameter_tuning = True
        self.hyperparameter_study = study
        
        self.get_hyperperamter_study_plots(study)
        
    def optuna_objective(self, trial = None):
    
        torch.cuda.empty_cache()
        
        batch_size = trial.suggest_int("batch_size", 2, 20, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2)
        
        self.initialise_logger(silence_logger=True)
        self.initialise_model()
        
        pretrained_model = self.model.train(
            self.tune_images,
            self.tune_flows,
            channels = [0,0],
            batch_size = batch_size,
            learning_rate = learning_rate,
            min_train_masks = 0,
            n_epochs = self.tune_epochs,
            save_every=1,
            )
        
        _, train_loss, _ = self.get_logger_data()
        
        return train_loss[-1]
    
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
            
        
    def train(self, epochs = None, learning_rate = None):
        
        if type(learning_rate) == float:
            self.learning_rate = learning_rate

        if type(epochs) == int:
            self.epochs = epochs
        
        self.initialise_logger()
        self.initialise_model()
        
        self.num_train_images = len(self.train_data["images"])
        self.num_validation_images = len(self.val_data["images"])
        
        if os.path.exists(self.cellpose_folder):
            shutil.rmtree(self.cellpose_folder)

        print("Training model...")
        
        self.start_time = time.process_time()
  
        pretrained_model = self.model.train(
            self.train_data["images"],
            self.train_data["masks"],
            test_data = self.val_data["images"],
            test_labels = self.val_data["masks"],
            channels = [0,0],
            save_path =  self.cellpose_folder,
            model_name = self.model_weights_name,
            batch_size = self.batch_size,
            learning_rate = self.learning_rate,
            min_train_masks = 0,
            n_epochs = self.epochs,
            save_every=1,
            )
        
        self.end_time = time.process_time()
        self.train_duration = self.end_time - self.start_time
        self.train_speed = self.train_duration/self.epochs
        
        print(f"Cellpose model weights saved to: {pretrained_model}")
        
        epoch_list, train_loss, val_loss = self.get_logger_data()
        
        training_dict = {"loss":train_loss}
        validation_dict = {"loss":val_loss}
        
        if os.path.exists(pretrained_model):
            shutil.move(pretrained_model, self.model_weights_path)
        
        torch.save({'num_epochs': self.epochs,
                    'learning_rate': self.learning_rate,
                    'batch_size':self.batch_size,
                    'learning_rate_values': [self.learning_rate]*self.epochs,
                    'training_dict': training_dict,
                    'validation_dict': validation_dict,
                    'epoch_list': epoch_list,
                    'label_list': self.label_list,
                    'num_train_images': self.num_train_images,
                    'num_validation_images': self.num_validation_images,
                    "hyperparameter_tuning": self.hyperparameter_tuning,
                    "hyperparameter_study": self.hyperparameter_study,
                    "hyperparameter_plots": self.hyperparameter_plots,
                    "start_time":self.start_time,
                    "end_time":self.end_time,
                    "training_speed": self.train_speed,
                    "num_model_parameters": self.num_model_parameters,
                    },
                    self.model_path,)
        
        print(f"Cellpose model info saved to: {self.model_path}")
        
    def evaluate(self, test_data, model_path = ""):
        
        image_list = test_data["images"]
        mask_list = test_data["masks"]
        
        if model_path != "" and os.path.exists(model_path):
            if "_weights" in model_path:
                self.model_weights_path = model_path
                self.model_path = model_path.replace("_weights")
            else:
                self.model_path = model_path
                self.model_weights_path = model_path + "_weights"

        self.initialise_model()
            
        print(f"Cellpose evaluating {len(test_data['masks'])} images on {self.gpu_mode}")
            
        model_data = torch.load(self.model_path)

        pred_mask_list, _, _ = self.model.eval(
            image_list,
            diameter=self.diam_mean,
            channels=[0, 0],
            flow_threshold=0.9,
            cellprob_threshold=0,
            min_size=15,
            batch_size=1000,
            )
    
        evalutation_dict = {}
    
        ap_results = calculate_average_precision(mask_list, pred_mask_list)
        
        evalutation_dict = {**evalutation_dict, **ap_results}
        
        model_data["evaluation_dict"] = evalutation_dict
        model_data["num_test_images"] = len(test_data['masks'])
        
        torch.save(model_data, self.model_path)
        
        return model_data
        
        
    def segment(self, images: list = []):
        
        self.initialise_model()
        
        if self.verbose:
            print(f"Cellpose Segmenting {len(images)} on {self.gpu_mode}")
        
        masks, flows, diams = self.model.eval(
            images,
            diameter=self.diam_mean,
            channels=[0, 0],
            flow_threshold=0.9,
            cellprob_threshold=0,
            min_size=15,
            batch_size=1000)

        return masks
    
    def segment_pipeline(self, dataset, model_path = ""):
        
        image_list = dataset["images"]
        mask_list = dataset["masks"]
        
        
        
        if model_path != "" and os.path.exists(model_path):
            if "_weights" in model_path:
                self.model_weights_path = model_path
                self.model_path = model_path.replace("_weights")
            else:
                self.model_path = model_path
                self.model_weights_path = model_path + "_weights"

        self.initialise_model()
        self.initialise_logger(silence_logger=True)
            
        model_data = torch.load(self.model_path)

        pred_mask_list, _, _ = self.model.eval(
            image_list,
            diameter=self.diam_mean,
            channels=[0, 0],
            flow_threshold=0.9,
            cellprob_threshold=0,
            min_size=15,
            batch_size=1000,
            )
        
        dataset["pred_masks"] = pred_mask_list
        
        return dataset

        
        
    
    
    
    
    
    
    