import torch
import pickle
import os
from torch.utils import data
import torch.optim as optim
import tqdm
import numpy as np
from datetime import datetime
import torchvision
import time
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import mobilenet_v2
from utils.dataloader import load_dataset
import matplotlib.pyplot as plt
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import FasterRCNN
from torch import nn
from torchvision.ops import box_iou
import copy
import optuna

from torch.utils.tensorboard import SummaryWriter
from utils.metrics import calculate_average_precision

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import balanced_accuracy_score, accuracy_score

class model_wrapper:

    def __init__(self,
                 train_data: dict = {},
                 val_data: dict = {},
                 epochs: int = 10,
                 batch_size: int = 10,
                 learning_rate: int = 0.0005,
                 model_path: str = "",
                 transfer_model_path: str = "",
                 augment: bool = True,
                 tensorboard: bool = True,
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 verbose: bool = True,
                 model_dir: str =  "",
                 gpu_int: int = 0, 
                 train_mode: str = "holdout",
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
        self.verbose = verbose
        self.model_dir = model_dir
        self.gpu_int = gpu_int
        self.transfer_model_path = transfer_model_path
        self.train_mode = train_mode
        
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
        
        if self.verbose:
            self.silence_tqdm = False
        else:
            self.silence_tqdm = True
        
        
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
            
            self.model_name = f"DASP-segmentor-maskcnn-{self.train_condition}-[{self.train_mode}]-{self.timestamp}"
            self.model_path = os.path.join(self.model_dir, self.model_name)
        
        
    def initialise_backbone(self, backbone_name="resnet_50"):
        
        if backbone_name == 'resnet_18':
        
            resnet_net = torchvision.models.resnet18(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 512
         
        elif backbone_name == 'resnet_34':
         
            resnet_net = torchvision.models.resnet34(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 512
        
        elif backbone_name == 'resnet_50':
         
            resnet_net = torchvision.models.resnet50(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
         
        elif backbone_name == 'resnet_101':
         
            resnet_net = torchvision.models.resnet101(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
         
        elif backbone_name == 'resnet_152':
         
            resnet_net = torchvision.models.resnet152(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
         
        elif backbone_name == 'resnet_50_modified_stride_1':
            resnet_net = torchvision.models.resnet50(pretrained=True)
            modules = list(resnet_net.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
         
        elif backbone_name == 'resnext101_32x8d':
        
            resnet_net = torchvision.models.resnext101_32x8d(weights="DEFAULT")
            modules = list(resnet_net.children())[:-1]
            backbone = nn.Sequential(*modules)
            backbone.out_channels = 2048
        
        elif backbone_name == "mobilenet_v2":
            
            backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
            backbone.out_channels = 1280
        
        return backbone
        
    def build_custom_maskrcnn(self, num_classes, backbone_name="resnet_50"):
        
        backbone = self.initialise_backbone(backbone_name = backbone_name)
        
        # let's make the RPN generate 5 x 3 anchors per spatial
        # location, with 5 different sizes and 3 different aspect
        # ratios. We have a Tuple[Tuple[int]] because each feature
        # map could potentially have different sizes and
        # aspect ratios
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))
        
        # let's define what are the feature maps that we will
        # use to perform the region of interest cropping, as well as
        # the size of the crop after rescaling.
        # if your backbone returns a Tensor, featmap_names is expected to
        # be ['0']. More generally, the backbone should return an
        # OrderedDict[Tensor], and in featmap_names you can choose which
        # feature maps to use.
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=7,
                                                        sampling_ratio=2)
        
        # put the pieces together inside a FasterRCNN model
        model = FasterRCNN(backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    
        return model 
    

    def initialise_model(self, num_classes = "", model_type = "FasterRCNN", freeze = None):
        
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
            num_classes = len(np.unique(self.train_data["labels"]))
        
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")  # load an instance segmentation model pre-trained pre-trained on COCO
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=num_classes)  # replace the pre-trained head with a new one  
        
        custom_model = False
        
        if self.model_path != "":
            if os.path.exists(self.model_path) == True:
                model_data = torch.load(self.model_path)
                self.model.load_state_dict(model_data['model_state_dict'])
                
                custom_model = True
                
            
        if self.transfer_model_path != "":
            if os.path.exists(self.transfer_model_path) == True:
                
                model_data = torch.load(self.transfer_model_path)
                model_num_classes = int(model_data['model_state_dict']["roi_heads.box_predictor.cls_score.weight"].shape[0])
                
                if model_num_classes != num_classes:
                    in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
                    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=model_num_classes)  # replace the pre-trained head with a new one  
                
                self.model.load_state_dict(model_data['model_state_dict'])
                
                if model_num_classes != num_classes:
                    in_features = self.model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
                    self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=num_classes)  # replace the pre-trained head with a new one  
                
                custom_model = True
                

            
        if self.verbose:
            if custom_model:       
                print(f"loaded custom maskrcnn model '{os.path.basename(self.model_path)}' on {self.gpu_mode}")
            else:
                print(f"loaded maskrcnn model on {self.gpu_mode}")
            
            
        self.model.to(self.device)
        
        
        self.bar_format = '{l_bar}{bar:2}{r_bar}{bar:-10b} [{remaining}]'
        
        self.num_model_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
    def freeze_model(self, freeze_list = []):
        
        freezable_layers = ["backbone","rpn","roi_heads", "box_head", "box_predictor", "mask_head", "mask_predictor"]
        
        freeze_list = [layer for layer in freeze_list if layer in freezable_layers]
        
        if len(freeze_list) > 0:
            
            if self.verbose:
                print(f"freezing layers: {freeze_list}")

        params = [p for p in self.model.parameters() if p.requires_grad]

        for name, parameters in self.model.named_parameters():
            
            freeze = False
            for layer in freeze_list:
                if layer in name:
                    freeze = True

            if freeze:
                parameters.requires_grad = False
            else:
                parameters.requires_grad = True
            
    def collate_fn(self, items):
        
     x = []
     y = []
     
     for x_, y_ in items:
         x.append(x_)
         y.extend(y_)
         
     return x, y
  
        
    def initialise_dataloaders(self):
        
        num_classes = len(self.train_data["label_list"])
        
        self.train_dataset = load_dataset(images = self.train_data["images"],
                                          masks = self.train_data["masks"],
                                          labels = self.train_data["labels"],
                                          num_classes=num_classes,
                                          image_size=(256,256),
                                          maskrcnn=True,
                                          augment=True)
        
        self.val_dataset = load_dataset(images = self.val_data["images"],
                                        masks = self.val_data["masks"],
                                        labels = self.val_data["labels"],
                                        num_classes=num_classes,
                                        image_size=(256,256),
                                        maskrcnn=True,
                                        augment=False)

        self.trainloader = data.DataLoader(dataset=self.train_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           collate_fn=self.collate_fn)
        
        self.valoader = data.DataLoader(dataset=self.val_dataset,
                                        batch_size=self.batch_size, 
                                        shuffle=False,
                                        collate_fn=self.collate_fn)
        
        self.num_train_images = len(self.trainloader)
        self.num_validation_images = len(self.valoader)
        
        return self.trainloader, self.valoader
        
    
    
    def load_tune_dataset(self, num_images=100, num_epochs = 10):
        
        num_classes = len(np.unique(self.train_data["labels"]))

        tune_images = []
        tune_masks = []
        tune_labels = []

        tune_train_dataset = load_dataset(images = self.train_data["images"][:num_images],
                                          masks = self.train_data["masks"][:num_images],
                                          labels = self.train_data["labels"][:num_images],
                                          num_classes=num_classes,
                                          image_size=(256,256),
                                          maskrcnn=True,
                                          augment=True)
        
        tuneloader = data.DataLoader(dataset=tune_train_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     collate_fn=self.collate_fn)

        for i in range(num_epochs):
            for images, labels in tuneloader:
                
                for image, label in zip(images,labels):
                    
                    img_dim = len(image.shape)
                    mask_dim = len(label["masks"].numpy().shape)
                    
                    if img_dim == 3 and mask_dim == 3:
                    
                        image = image.numpy()
                        tune_images.append(image)
                    
                        tune_labels.extend(label["labels"].numpy())
                        tune_masks.extend(label["masks"].numpy())

        if self.verbose:
            print(f"\nLoaded {len(tune_images)} augmented images for hyperparameter tuning.\n")

        self.tune_train_dataset = load_dataset(images=tune_images,
                                               masks=tune_masks,
                                               labels=tune_labels,
                                               num_classes=num_classes,
                                               image_size=(256,256),
                                               maskrcnn=True,
                                               augment=False)

    def optuna_objective(self, trial):

        batch_size = trial.suggest_int("batch_size", 2, 50, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1)

        tune_trainloader = data.DataLoader(dataset=self.tune_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           collate_fn=self.collate_fn)
    
    
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        model = copy.deepcopy(self.model)
        model.to(self.device)

        running_loss = []
        
        for i, (images, labels) in enumerate(tune_trainloader):
            
            try:
            
                images = list(image.to(self.device) for image in images)
                labels =[{k: v.to(self.device) for k,v in t.items()} for t in labels]
            
                self.optimizer.zero_grad()  # zerograd the parameters
                loss_dict = model(images, labels)  # one forward pass
            
                loss = sum(loss for loss in loss_dict.values())
                
                running_loss.append(loss.item())
                
                mean_running_loss = np.mean(running_loss[-3:])
            
                trial.report(np.mean(running_loss[-3:]), step=i)
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
                loss.backward()
                self.optimizer.step()
                
            except:
                mean_running_loss = None
                pass

        return mean_running_loss
    
    
    def tune_hyperparameters(self, num_trials=5, num_images = 500, num_epochs = 4):
        
        if self.verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        self.initialise_model()
        self.criterion = nn.CrossEntropyLoss()
        
        self.load_tune_dataset(num_images = num_images, num_epochs = num_epochs)

        self.num_tune_images = num_images
        self.num_tune_epochs = num_epochs
        
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
        
        study = optuna.create_study(direction='minimize', pruner=pruner)
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
        self.hyperparameter_tuning = True
        self.hyperparameter_study = study
        
        self.get_hyperperamter_study_plots(study)

        return
    
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
    
    def train(self, epochs = None, model = "FasterRCNN", freeze_list = [], learning_rate = None, batch_size = None, initialise_loggers = False):
        
        if type(learning_rate) == float:
            self.learning_rate = learning_rate

        if type(epochs) == int:
            self.epochs = epochs

        if type(batch_size) == int:
            self.batch_size = batch_size

        if hasattr(self, "model") == False:
            if self.verbose:
                print("initialising model")
            self.initialise_model()
            
        self.freeze_model(freeze_list=freeze_list)
        
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(params=params, lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=100,gamma=0.1)
        
        self.start_time = time.process_time()

        if initialise_loggers == True:
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
        
        torch.cuda.empty_cache()
    
        if self.tensorboard:
            self.writer = SummaryWriter(log_dir="runs/" + "maskrcnn" + "_" + self.timestamp)
        
        self.initialise_dataloaders()
        
        if self.verbose:
            print(f"\nTraining maskrcnn model: \n\t freeze_list = {freeze_list} \n\t learning_rate = {self.learning_rate} \n\t num_epochs = {self.epochs} \n\n")
        
        progressbar = tqdm.tqdm(range(self.epochs), 'Progress', total=self.epochs, position=0, leave=True, bar_format=self.bar_format, disable=self.silence_tqdm)

        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self.train_step()

            """Validation block"""
            if self.valoader is not None:
                self.val_step()
        
            self.learning_rate_values.append(self.optimizer.param_groups[0]['lr'])
            self.freeze_mode_list.append(freeze_list)
        
            self.lr_scheduler.step()  # learning rate scheduler step
        
            if self.validation_loss[-1] == np.min(self.validation_loss):
                
                self.model.eval()
                
                self.end_time = time.process_time()
                self.train_duration = self.end_time - self.start_time
                self.train_speed = self.train_duration/self.epochs
                
                torch.save({'epoch': self.epoch,
                            'num_epochs': self.epochs,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'learning_rate_values': self.learning_rate_values,
                            'freeze_list': self.freeze_mode_list,
                            'training_loss': self.training_loss,
                            'validation_loss': self.validation_loss,
                            'label_list': self.label_list,
                            'num_train_images': self.num_train_images,
                            'num_validation_images': self.num_validation_images,
                            "training_loss_dict": self.training_loss_dict,
                            "validation_loss_dict": self.validation_loss_dict,
                            "start_time":self.start_time,
                            "end_time":self.end_time,
                            "training_speed": self.train_speed,
                            "num_model_parameters": self.num_model_parameters,
                            "hyperparameter_tuning": self.hyperparameter_tuning,
                            "hyperparameter_study": self.hyperparameter_study,
                            "hyperparameter_plots": self.hyperparameter_plots,
                            },
                            self.model_path,)
            
            if self.writer:
                for key,value in self.training_loss_dict.items():
                    self.writer.add_scalar("train_" + key, value[-1], self.epoch)
                for key,value in self.validation_loss_dict.items():
                    self.writer.add_scalar("val_" + key, value[-1], self.epoch)
                self.writer.add_scalar("learning rate", self.learning_rate_values[-1], self.epoch)
                
            progressbar.set_description(f'(Training Loss {self.training_loss[-1]:.5f}, Validation Loss {self.validation_loss[-1]:.5f})')  # update progressbar
        
        if self.verbose:
            print(f"Maskrcnn model saved to: {self.model_path}")
            
        return self.model_path
            
    def train_step(self):
        
        self.model.train()
        train_losses = []
        train_losses_dict = {"loss":[]}

        batch_iter = tqdm.tqdm(enumerate(self.trainloader), 'Training', total=len(self.trainloader), position=1,leave=False, bar_format=self.bar_format, disable=self.silence_tqdm)
        
        for i, (images, labels) in batch_iter:
            
            try:

                images = list(image.to(self.device) for image in images)
                labels =[{k: v.to(self.device) for k,v in t.items()} for t in labels]
            
                self.optimizer.zero_grad()  # zerograd the parameters
                loss_dict = self.model(images, labels)  # one forward pass
            
                losses = sum(loss for loss in loss_dict.values())
                train_losses.append(losses.item())
                
                for key,value in loss_dict.items():
                    if key not in train_losses_dict.keys():
                        train_losses_dict[key] = []
                    train_losses_dict[key].append(loss_dict[key].item())
                train_losses_dict["loss"] = losses.item()
    
                losses.backward()
                self.optimizer.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
    
                batch_iter.set_description(f'Training[{self.epoch}\\{self.epochs}]:(loss {np.mean(train_losses):.3f}')  # update progressbar
                
            except:
                pass

        for key,value in train_losses_dict.items():
            if key not in self.training_loss_dict.keys():
                self.training_loss_dict[key] = []
            self.training_loss_dict[key].append(np.mean(value))

        self.training_loss.append(np.mean(train_losses))
        batch_iter.close()
        
        
    def calculate_box_iou(self, target, prediction):
        
        iou_list = []
        
        for i in range(len(target)):
            
            boxes = target[i]["boxes"].detach().cpu()
            
            pred_scores = prediction[i]["scores"].detach().cpu()
            pred_boxes = prediction[i]["boxes"].detach().cpu()

            
            if len(pred_boxes) > 0 and len(boxes) > 0:
            
                filtered_pred_boxes = []
                
                for j, box in enumerate(pred_boxes):
                    
                    score = pred_scores[i]

                    if score > 0.5:
                        
                        filtered_pred_boxes.append(np.array(box))
                        
                filtered_pred_boxes = torch.tensor(np.array(filtered_pred_boxes))
                
                iou = np.nanmean(np.max(box_iou(boxes, pred_boxes).numpy(),axis=0)) 
                
                iou_list.append(iou)
            
        return iou_list
                    
    
    def val_step(self):
        
        self.model.train()
        val_losses = []
        val_losses_dict = {"loss":[]}

        batch_iter = tqdm.tqdm(enumerate(self.valoader), 'Validating', total=len(self.valoader), position=1,leave=False, bar_format=self.bar_format, disable=self.silence_tqdm)
        
        for i, (images, labels) in batch_iter:
            
            try:

                images = list(image.to(self.device) for image in images)
                labels =[{k: v.to(self.device) for k,v in t.items()} for t in labels]
            
                self.optimizer.zero_grad()  # zerograd the parameters
                loss_dict = self.model(images, labels)  # one forward pass
            
                losses = sum(loss for loss in loss_dict.values())
                val_losses.append(losses.item())
                        
                for key,value in loss_dict.items():
                    if key not in val_losses_dict.keys():
                        val_losses_dict[key] = []
                    val_losses_dict[key].append(loss_dict[key].item())
                val_losses_dict["loss"] = losses.item()
                
                batch_iter.set_description(f'Validating[{self.epoch}\\{self.epochs}]:(loss {np.mean(val_losses):.3f}')  # update progressbar
            
            except:
                pass
            
        self.validation_loss.append(np.mean(val_losses))
        
        for key,value in val_losses_dict.items():
            if key not in self.validation_loss_dict.keys():
                self.validation_loss_dict[key] = []
            self.validation_loss_dict[key].append(np.mean(value))

        batch_iter.close()
    
    def mask_from_stack(self, mask_stack):
            
        instance_mask = np.zeros(mask_stack.shape[-2:], dtype=np.uint16)
        
        for i, mask in enumerate(mask_stack):
            
            if len(mask.shape) == 3:
                instance_mask[mask[0]>0.5] = i + 1
            if len(mask.shape) == 2:
                instance_mask[mask>0.1] = i + 1
                
        return instance_mask
    
    
    def correct_predictions(self, label, pred_label):

        if len(label.shape) > 1:
            correct = (label.data.argmax(dim=1) == pred_label.data.argmax(dim=1)).float().sum().cpu()
        else:
            correct = (label.data == pred_label.data).float().sum().cpu()

        accuracy = correct / label.shape[0]

        return accuracy.numpy()

    
    def evaluate(self, test_data, model_path = ""):
        
        if self.verbose:
            self.silence_tqdm = True
        else:
            self.silence_tqdm = False
        
        num_classes = len(self.train_data["label_list"])
        
        test_dataset = load_dataset(images = test_data["images"],
                                        masks = test_data["masks"],
                                        labels = test_data["labels"],
                                        num_classes=num_classes,
                                        maskrcnn=True,
                                        augment=False)

        testloader = data.DataLoader(dataset=test_dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     collate_fn=self.collate_fn)
        
        if model_path != "" and os.path.exists(model_path):
            print(f"loading {os.path.basename(model_path)}")
            self.model_path = model_path
            
        model_data = torch.load(self.model_path)
        self.model.load_state_dict(model_data['model_state_dict'])

        self.model.eval()
        
        train_losses = []
        train_losses_dict = {"loss":[]}
        
        iou_list = []

        batch_iter = tqdm.tqdm(enumerate(testloader),'Evaluating', total=len(testloader), position=1,leave=False, bar_format=self.bar_format, disable=self.silence_tqdm)
        
        prediction_list = []
        label_list = []
        image_list = []
        
        mask_list = []
        pred_mask_list = []
        
        true_labels_list = []
        pred_labels_list = []

        for i, (images, labels) in batch_iter:
            
            images = list(image.to(self.device) for image in images)
            
            if len(labels[0]["boxes"].shape) > 1:
                
                predictions = self.model(images)
                
                predictions = [{k: v.detach().cpu().numpy() for k,v in prediction.items()} for prediction in predictions]
                images = [image.detach().cpu().numpy() for image in images]
                
                masks = [self.mask_from_stack(label["masks"].cpu().numpy()) for label in labels]
                pred_masks = [self.mask_from_stack(pred["masks"]) for pred in predictions]
                
                pred_labels = np.array([pred["labels"] for pred in predictions]).flatten().tolist()
                true_labels = [labels[0]["labels"].cpu().numpy()[0]]*len(pred_labels)

                mask_list.extend(masks)
                pred_mask_list.extend(pred_masks)
                
                pred_labels_list.extend(pred_labels)
                true_labels_list.extend(true_labels)


        evaluation_dict = {}
        
        ap_results = calculate_average_precision(mask_list, pred_mask_list)
        
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
        
        evaluation_dict["classification_summary"] = classification_summary
        
        model_data["evaluation_dict"] = evaluation_dict
        
        torch.save(model_data, self.model_path)

        return evaluation_dict
        
        
    def detect(self, images = []):
        
        self.model.eval()
        
        masks = []
        
        batch_iter = tqdm.tqdm(enumerate(images), 'Detecting', total=len(images), position=0,leave=True, bar_format=self.bar_format, disable=self.silence_tqdm)
        
        result_dict = {"images" : []}
        
        for i, image in batch_iter:
            
            result_dict["images"].append(image)
            
            img = torch.as_tensor(image).to(self.device)
            
            with torch.no_grad():
                
                predictions = self.model([img])
                
                for pred in predictions:
                    
                    for key,value in pred.items():
                        
                        if key not in result_dict.keys():
                            
                            result_dict[key] = []
                            
                        value = pred[key].detach().cpu().numpy()
                            
                        result_dict[key].append(value)
    
        return result_dict
    
    
    def segment(self, images: list = []):
        
        self.initialise_model(num_classes=5)
        
        self.model.eval()
        
        masks = []
        
        batch_iter = tqdm.tqdm(enumerate(images), 'Segmenting', total=len(images), position=0,leave=False, bar_format=self.bar_format, disable=self.silence_tqdm)
        
        for i, img in batch_iter:
            
            img = torch.as_tensor(img).to(self.device)
            
            with torch.no_grad():
                
                pred = self.model([img])
                
                masks_stack = pred[0]['masks'].detach().cpu().numpy()
                pred_mask = self.mask_from_stack(masks_stack)
                
                masks.append(pred_mask)
                
        return masks
        
        
        
        
        
        # batch_iter = tqdm.tqdm(enumerate(self.valoader), 'Segmenting', total=len(self.valoader), position=0,leave=False, bar_format=self.bar_format)
        
        # for i, (images, labels) in batch_iter:
            
        #     images = list(image.to(self.device) for image in images)
            
        #     with torch.no_grad():
        #     pred = model(images)
            
            
        
        
        
        
        
        
        
        
    # def train(self, epochs=None):
        
    #     if type(epochs) == int:
    #         self.epochs = epochs
        
    #     self.initialise_dataloaders()
        
    #     for epoch in range(self.epochs):
            
    #         print(f"Epoch {epoch+1}/{self.epochs }")
            
    #         for i, (images, targets) in enumerate(self.trainloader):
                
    #            images = list(image.to(self.device) for image in images)
    #            targets=[{k: v.to(self.device) for k,v in t.items()} for t in targets]
                
    #            loss_dict = self.model(images, targets)
                
    #            losses = sum(loss for loss in loss_dict.values())
    
    #            self.optimizer.zero_grad()
    #            losses.backward()
    #            self.optimizer.step()
               
    #            print(f"Iteration {i}, Loss: {losses.item()}")
    
    #            if i % 50 == 0:
    #                print(f"Iteration {i}, Loss: {losses.item()}")

        
        