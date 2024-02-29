import sys
sys.path.append("C:\AK-DASP\src")


import pickle
import matplotlib.pyplot as plt
import cv2

from utils.file_io import get_metadata, cache_images
from utils.dataset import get_training_data

from segmentors.maskrcnn_model import model_wrapper

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

    # with open('cached_images.pickle', 'rb') as handle:
    #     cached_data = pickle.load(handle)

    # datasets  = get_training_data(
    #     cached_data,
    #     mode="ensemble",
    #     verbose=False,
    #     )
    
    # with open('datasets.pickle', 'wb') as handle:
    #     pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('datasets.pickle', 'rb') as handle:
        datasets = pickle.load(handle)
    
    model_path = r"/home/turnerp/PycharmProjects/AK-DASP/models/maskrcnn/DASP-segmentor-maskcnn-500"
    
    model = model_wrapper(train_data = datasets["train_data"],
                          val_data = datasets["val_data"],
                           model_path=model_path,
                          verbose=True)
    
    # model_data = model.evaluate(datasets["test_data"])
    # model.tune_hyperparameters(num_trials=50, num_images = 1000, num_epochs = 10)
    # masks = model.segment(datasets["test_data"]["images"])

    