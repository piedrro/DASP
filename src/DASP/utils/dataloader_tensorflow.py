import tensorflow as tf
import numpy as np




class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, 
                 images: list = [],
                 labels: list = [],
                 image_size = (64,64),
                 num_classes: int = 0,
                 batch_size=32, 
                 shuffle=True, 
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

    def augmentor(self, img):
        
        return img
    
    def __data_generation(self, images, labels, image_size, num_classes):
        
        X = np.empty((self.batch_size, image_size[0], image_size[1], 3), dtype=np.float32)
        Y = np.empty((self.batch_size, num_classes),  dtype=np.float32)

        for i, (img, label) in enumerate(zip(images, labels)):
            
            print(img.shape)
            
            if self.augment:
                img = self.augmentor(img)
            
            # X[i,:,:,:] = img
            # Y[i,] = label

        return X, Y
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    