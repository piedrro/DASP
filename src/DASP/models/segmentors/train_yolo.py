import sys
sys.path.append("C:\AK-DASP\src")

import os
print(os.getcwd())

import pickle
import matplotlib.pyplot as plt
import cv2

from utils.file_io import get_metadata, cache_images
from utils.dataset import get_training_data
from segmentors.yolov8_model import model_wrapper



file_metadata = get_metadata(dataset_directory = "/home/turnerp/PycharmProjects/AK-DASP/dataset")



if __name__ == '__main__':
    
    # cached_data = cache_images(file_metadata,
    #                             mode = "images",
    #                             dataset = "MG1655", 
    #                             class_labels = "treatment",
    #                             normalise=True,
    #                             mask_background=False)
    
    # with open('cached_images.pickle', 'wb') as handle:
    #     pickle.dump(cached_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cached_images.pickle', 'rb') as handle:
        cached_data = pickle.load(handle)

    datasets  = get_training_data(
        cached_data,
        mode="ensemble",
        verbose=True,
        )

    model = model_wrapper(train_data = datasets["train_data"],
                          val_data = datasets["val_data"],
                          verbose=True)

    # # model.train(epochs=100, batch_size=9)
    # # model_data = model.evaluate(test_data = datasets["test_data"])

    model.export_model()

    
    






