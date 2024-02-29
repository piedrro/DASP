

from datetime import datetime
from keras.applications.densenet import DenseNet121
from keras import layers, activations, losses, optimizers, metrics, Model, callbacks
from keras.optimizers import SGD,Adam, Nadam, Adagrad
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
import numba
import tqdm
import skimage
from imgaug import augmenters as iaa
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import confusion_matrix
import cv2
import albumentations as A
from albumentations.augmentations.blur.transforms import Blur
from albumentations.augmentations.transforms import RGBShift, GaussNoise, PixelDropout, ChannelShuffle
from albumentations import RandomBrightnessContrast, RandomRotate90, Flip, Affine


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, 
                 images: list = [],
                 labels: list = [],
                 image_size = (64,64),
                 num_classes: int = 0,
                 batch_size=32, 
                 shuffle=False, 
                 augment=False):
        
        self.images = images 
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.image_size = image_size
        self.num_classes = num_classes
        
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch
        return int(np.floor(len(self.images) / self.batch_size))


    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        images = [self.images[k] for k in indexes]
        labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(images, labels, self.image_size, self.num_classes)

        return X, y
    

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def rescale01(self, x):
        """ normalize image from 0 to 1 """
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

    def augmentor(self, img):
        
        # geometric transforms
        seq1 = iaa.Sequential(
            [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                rotate=(-90, 90),
                mode="constant",
                cval=0),
            ],
            random_order=True)
        
        #pixel transforms
        seq2 = iaa.Sequential(
        [
        iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.5, 1.5))), #Random sharpness increae
        iaa.Sometimes(0.5, iaa.WithChannels(0, iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}))), #Random up to 10% misalignment
        iaa.Sometimes(0.5, iaa.MultiplyBrightness((0.5, 2.0))), #Brightness correction
        iaa.Sometimes(0.5, iaa.imgcorruptlike.GaussianNoise(severity=(1,2))), #Random gaussian noise
        # iaa.Sometimes(0.5, iaa.imgcorruptlike.DefocusBlur(severity=(1,2))), #Defocus correction
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 2.5)))
        ],
        )


        img = np.moveaxis(img,0,-1)

        seq_det1 = seq1.to_deterministic()
        img = seq_det1.augment_images([img])[0]
        
        mask = img.copy()
        mask[mask==0] = 1
    
        seq_det2 = seq2.to_deterministic()
        img = seq_det2.augment_images([img])[0]
        
        img[mask==1]=0
        
        img = np.moveaxis(img,-1,0)

        return img
    
    def __data_generation(self, images, labels, image_size, num_classes):
        
        X = np.empty((self.batch_size, image_size[0], image_size[1], 3), dtype=np.float32)
        Y = np.empty((self.batch_size, 2),  dtype=np.float32)

        for i, (img, label) in enumerate(zip(images, labels)):
            
            if img.shape[:-1] != image_size:
                
                if img.shape[-1] != 3:
                    img = np.moveaxis(img, 0, -1)
                    
                n_chan = img.shape[0]
                reshape_size = (n_chan,) + image_size
                img = cv2.resize(img, image_size) 
            
            if img.shape[-1] == 3:
                img = np.moveaxis(img, -1, 0)
                
            if self.augment:
                img = self.augmentor(img)
            
            img = self.rescale01(img)
        
            img = np.moveaxis(img, 0, -1)
            
            X[i,:,:,:] = img
            Y[i] = tf.one_hot([label], 2,dtype="int32")
            
        X = np.asarray(X)
        X = skimage.img_as_ubyte(X)  # Cast between 0-1, resize
        histeq = iaa.Sequential([iaa.AllChannelsHistogramEqualization()]) #Equalize histogram
        X = histeq(images = X)
            
        return X, Y

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
                 image_size: tuple() = (64,64,3),
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
        self.image_size = image_size
        
        
        
    def initialise_model(self, num_classes = [], opt = "Adam"):
        
        tf.config.set_visible_devices = str(self.gpu_int)
        
        if self.model_path != "" and os.path.exists(self.model_path):
            weights = self.model_path
        else:
            weights = None
        
        if type(num_classes) == int:
            self.num_classes = num_classes
        else:
            self.num_classes = len(np.unique(self.train_data["labels"]))
    
        #Select model from supported modes
        self.model = DenseNet121(include_top=True, weights=weights, input_shape=self.image_size, classes=self.num_classes)
    
        for layer in self.model.layers:
            layer.trainable = True #Ensure all layers are trainable
    
        #Select optimimzer
        if opt == 'SGD+N': #SGD with nestrov
            optimizer = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=True) #SGD with nesterov momentum, no vanilla version
        elif opt == 'SGD': #SGD with ordinary momentum
            optimizer = SGD(learning_rate=self.learning_rate, momentum=0.9, nesterov=False)  # SGD with nesterov momentum, no vanilla version
        elif opt == 'NAdam':
            optimizer = Nadam(learning_rate=self.learning_rate)  # Nestrov Adam
        elif opt == 'Adam':
            optimizer = Adam(learning_rate=self.learning_rate)  # Adam
        else:
            raise TypeError('Optimizer {} not supported'.format(opt))

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.bar_format = '{l_bar}{bar:2}{r_bar}{bar:-10b} [{remaining}]'
        
    def initialise_dataloaders(self):
         
        self.trainloader = DataGenerator(
            images = self.train_data["images"],
            labels = self.trian_data["labels"],
            num_classes=self.num_classes,
            image_size = self.image_size,
            batch_size = self.batch_size,
            augment = self.augment)   
        
        self.valoader = DataGenerator(
            images = self.val_data["images"],
            labels = self.val_data["labels"],
            num_classes=self.num_classes,
            image_size = self.image_size,
            batch_size = self.batch_size,
            augment = False)   
         
        
        
    def classify_and_evaluate(self, images = [], labels = [], num_classes=2, augment=False):
        
        self.initialise_model(num_classes=num_classes)
        
        if self.verbose:
            self.silence_tqdm = False
            self.tf_verbose = 1
        else:
            self.silence_tqdm = True
            self.tf_verbose = 0
        
        target_label_list = []
        pred_label_list = []
        pred_confidence_list = []
        
        generator = DataGenerator(
            images = images,
            labels = labels,
            num_classes=self.num_classes,
            image_size = self.image_size,
            batch_size = self.batch_size, 
            augment = augment)
        
        batch_iter = tqdm.tqdm(enumerate(generator), 'Classifying', 
                               total=len(generator),
                               position=0,leave=False, 
                               bar_format=self.bar_format, 
                               disable=self.silence_tqdm)
        
        for i, (images, labels) in batch_iter:
            
            with tf.device(f'/device:GPU:{self.gpu_int}'):
                result = self.model.predict(images,verbose=0)
            
                target_labels = np.argmax(labels, axis=1).tolist()
                pred_labels = np.argmax(result,axis=1).tolist()
                pred_confidence = np.max(result,axis=1).tolist()
                
                target_label_list.extend(target_labels)
                pred_label_list.extend(pred_labels)
                pred_confidence_list.extend(pred_confidence)
            
        
        cm = confusion_matrix(target_label_list, pred_label_list, normalize='pred')
        cm_counts = confusion_matrix(target_label_list, pred_label_list,normalize=None)
        accuracy = accuracy_score(target_label_list, pred_label_list)
        balanced_accuracy = balanced_accuracy_score(target_label_list, pred_label_list)
        
        try:
            TN, FP, FN, TP = cm_counts.ravel()
            
            sensitivity = TP/(TP+FN)
            specificity = TN/(TN+FP) 
            accuracy = (TP + TN) / (TN + FP + FN + TP)
        except:
            TN, FP, FN, TP, sensitivity, specificity = 0,0,0,0,0,0
            

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
        
        return prediction_data
        

    def classify(self, images, labels = [], num_classes=2, augment=False, batch_size=10):
        
        self.initialise_model(num_classes=num_classes)
        
        if len(labels) == 0:
            labels = [0]*len(images)
            
        if self.verbose:
            self.silence_tqdm = False
        else:
            self.silence_tqdm = True
        
        target_label_list = []
        pred_label_list = []
        pred_confidence_list = []
        
        generator = DataGenerator(
            images = images,
            labels = labels,
            num_classes = num_classes,
            image_size = (64,64),
            batch_size = batch_size, 
            augment = False)   

        batch_iter = tqdm.tqdm(enumerate(generator), 'Classifying', 
                                total=len(generator),
                                position=0,leave=False, 
                                bar_format=self.bar_format, 
                                disable=self.silence_tqdm)

        target_label_list = []
        pred_label_list = []
        pred_confidence_list = []

        for i, (images, labels) in batch_iter:
            
            with tf.device(f'/device:GPU:{self.gpu_int}'):
                
                result = self.model.predict(images, verbose=0)
                
                target_labels = np.argmax(labels, axis=1).tolist()
                pred_labels = np.argmax(result,axis=1).tolist()
                confidence = np.max(result,axis=1).tolist()
            
                pred_label_list.extend(pred_labels)
                target_label_list.extend(target_labels)
                pred_confidence_list.extend(confidence)

        return pred_label_list, pred_confidence_list


    def classify_pipeline(self, pipeline_data, num_classes=2, augment=False):
        
        self.initialise_model(num_classes=num_classes)
        
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
    
            generator = DataGenerator(
                images = image_data["cell_images"],
                labels = image_data["cell_labels"],
                num_classes=self.num_classes,
                image_size = self.image_size,
                batch_size = self.batch_size, 
                augment = augment)
            
            image_data["pred_confidences"] = []
            image_data["pred_labels"] = []

            
            for (images, labels) in generator:
                
                with tf.device(f'/device:GPU:{self.gpu_int}'):
                    
                    result = self.model.predict(images, verbose=0)
                    
                    target_labels = np.argmax(labels, axis=1).tolist()
                    pred_labels = np.argmax(result,axis=1).tolist()
                    confidence = np.max(result,axis=1).tolist()
                    
                    image_data["pred_confidences"].extend(confidence)
                    image_data["pred_labels"].extend(pred_labels)
                    
                    pred_label_list.extend(pred_labels)
                    target_label_list.extend(target_labels)
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

            
            
        
        
        
        
        
        
        