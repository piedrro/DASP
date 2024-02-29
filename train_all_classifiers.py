
import sys
import pickle
from DASP.utils.file_io import get_metadata, cache_images
from DASP.utils.dataset import get_training_data
from DASP.models.classifiers.classifier_model import model_wrapper
from datetime import datetime


dataset = "MG1655"
class_labels = "treatment"
holdout_experiment_id = 7
import_limit = 100
image_channels = ["Nile Red", "DAPI"]
mask_background = True
colicoords = False
resize = False
normalise = False
align = True
stats = True
dataset_ratios = [0.7,0.2,0.1]

gpu_int = 0

model_dir = "classifier_ensemble_comparison"

timestamp = datetime.now().strftime("%y%m%d_%H%M")


batch_size = 10
learning_rate = 0.01
epochs = 100
verbose = True

file_metadata = get_metadata(dataset_directory = "PATH TO DASP DATA")

model_permuations = ["densenet121","densenet201",
                     "efficientnet_b0","efficientnet_b7",
                     "resnet18","resnet50","resnet101","resnet152",
                     "vgg11","vgg19", 
                     ]

train_permutations = ["ensemble"]

class_list = file_metadata[class_labels].unique().tolist()
class_list.remove("WT+ETOH")

class_permutations = []
for class_label in class_list:
    class_permutations.append(["WT+ETOH",class_label])
    

if gpu_int == 1:
    model_permuations = model_permuations[:len(model_permuations)//2]
else:
    model_permuations = model_permuations[len(model_permuations)//2:]


if __name__ == '__main__':
    
    cached_data = cache_images(file_metadata,
                                mode = "cells",
                                dataset = dataset,
                                class_labels = class_labels,
                                normalise=True,
                                mask_background=True)

    for label_names in class_permutations:
        
        for train_mode in train_permutations:
    
            for timm_model_backbone in model_permuations:
                
                print(f"training {train_mode} {timm_model_backbone} on labels: {label_names}")
                
                try:
                        
                    sys.stdout.write("\r loading...")
                    sys.stdout.flush()
                    
                    datasets = get_training_data(
                        cached_data,
                        label_names = label_names,
                        holdout_experiment_id=holdout_experiment_id,
                        xvalidation_label_limit='None',
                        label_limit=500,
                        mode = train_mode,
                        verbose = verbose,
                        )
       
                    model = model_wrapper(
                        train_data = datasets["train_data"],
                        val_data = datasets["val_data"],
                        timm_model_backbone = timm_model_backbone,
                        train_mode=train_mode,
                        verbose = verbose,
                        timestamp=timestamp,
                        dir_name = model_dir,
                        tensorboard=True,
                        gpu_int=gpu_int,
                        )
            
                    sys.stdout.write("\r tuning...")
                    sys.stdout.flush()
            
                    model.tune_hyperparameters(num_trials = 30,
                                                augment = True, 
                                                num_images= 3000,
                                                num_epochs = 5)
            
                    sys.stdout.write("\r training...")
                    sys.stdout.flush()
            
                    model_path = model.train(epochs = epochs)
                    
                    sys.stdout.write("\r evaluating...")
                    sys.stdout.flush()
    
                    if train_mode == "xvalidation": 
                        model_data = model.evaluate(test_dataset=datasets["test_data"], evalate_test_folds=True)
                    elif train_mode == "ensemble":
                        model_data = model.evaluate(test_dataset=datasets["test_data"])
                    else:
                        model_data = model.evaluate(holdout_dataset=datasets["holdout_data"])
  
                except:
                    import traceback
                    print(traceback.format_exc())
                    pass

        
        