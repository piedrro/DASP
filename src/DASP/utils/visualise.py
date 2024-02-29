
import numpy as np
from skimage import exposure
import itertools
from PIL import Image
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import io
from sklearn.metrics import balanced_accuracy_score
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pathlib
import os
import cv2

def process_image(img):
    for i in range(img.shape[0]):
        try:
            im = img[i]
            im = rescale01(im) * 255
            im = im.astype(np.uint8)

            im = normalize99(im)
            im = rescale01(im)
            img[i] = im
        except:
            pass
    img = np.swapaxes(img, 0, 2)

    return img

def normalize99(X):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """

    if len(X.shape) == 3:
        for i, img in enumerate(X):
            if np.max(img) > 0:
                img = img.copy()
                v_min, v_max = np.percentile(img[img != 0], (0.001, 99.999))
                img = exposure.rescale_intensity(img, in_range=(v_min, v_max))

                X[i] = img

    else:
        if np.max(X) > 0:
            X = X.copy()
            v_min, v_max = np.percentile(X[X != 0], (1, 99))
            X = exposure.rescale_intensity(X, in_range=(v_min, v_max))

    return X


def rescale01(x):
    """ normalize image from 0 to 1 """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x



def get_image_predictions(images, saliency, test_labels, pred_labels, pred_confidences, antibiotic_list):

    images_TP = []
    images_TN = []
    images_FP = []
    images_FN = []

    saliency_TP = []
    saliency_TN = []
    saliency_FP = []
    saliency_FN = []

    label_TP = None
    label_TN = None
    label_FP = None
    label_FN = None

    predicted_label_TP = None
    predicted_label_TN = None
    predicted_label_FP = None
    predicted_label_FN = None

    confidence_TN = []
    confidence_TP = []
    confidence_FN = []
    confidence_FP = []

    for i in range(len(images)):
        test_label = test_labels[i]
        pred_label = pred_labels[i]

        if test_label == 1 and pred_label == 1:
            if pred_confidences[i] not in confidence_TP:
                images_TP.append(images[i])
                saliency_TP.append(saliency[i])

                label_TP = antibiotic_list[test_label]
                predicted_label_TP = antibiotic_list[pred_label]
                confidence_TP.append(pred_confidences[i])

        if test_label == 0 and pred_label == 0:
            if pred_confidences[i] not in confidence_TN:
                images_TN.append(images[i])
                saliency_TN.append(saliency[i])

                label_TN = antibiotic_list[test_label]
                predicted_label_TN = antibiotic_list[pred_label]
                confidence_TN.append(pred_confidences[i])

        if test_label == 0 and pred_label == 1:
            if pred_confidences[i] not in confidence_FP:
                images_FP.append(images[i])
                saliency_FP.append(saliency[i])

                label_FP = antibiotic_list[test_label]
                predicted_label_FP = antibiotic_list[pred_label]
                confidence_FP.append(pred_confidences[i])

        if test_label == 1 and pred_label == 0:
            if pred_confidences[i] not in confidence_FN:
                images_FN.append(images[i])
                saliency_FN.append(saliency[i])

                label_FN = antibiotic_list[test_label]
                predicted_label_FN = antibiotic_list[pred_label]
                confidence_FN.append(pred_confidences[i])

    miss_predictions = {}

    if len(images_TP) > 0:
        images_TP, saliency_TP, confidence_TP = [list(x) for x in zip(*sorted(zip(images_TP, saliency_TP, confidence_TP), key=lambda x: x[2]))]
    if len(images_TN) > 0:
        images_TN, saliency_TN, confidence_TN = [list(x) for x in zip(*sorted(zip(images_TN, saliency_TN, confidence_TN), key=lambda x: x[2]))]
    if len(images_FP) > 0:
        images_FP, saliency_FP, confidence_FP = [list(x) for x in zip(*sorted(zip(images_FP, saliency_FP, confidence_FP), key=lambda x: x[2]))]
    if len(images_FN):
        images_FN, saliency_FN, confidence_FN = [list(x) for x in zip(*sorted(zip(images_FN, saliency_FN, confidence_FN), key=lambda x: x[2]))]


    miss_predictions["True Positives"] = {"images": images_TP, "saliency_map": saliency_TP, "true_label": label_TP, "predicted_label": predicted_label_TP, "prediction_confidence": confidence_TP}

    miss_predictions["True Negatives"] = {"images": images_TN, "saliency_map": saliency_TN, "true_label": label_TN, "predicted_label": predicted_label_TN, "prediction_confidence": confidence_TN}

    miss_predictions["False Positives"] = {"images": images_FP, "saliency_map": saliency_FP, "true_label": label_FP, "predicted_label": predicted_label_FP, "prediction_confidence": confidence_FP}

    miss_predictions["False Negatives"] = {"images": images_FN, "saliency_map": saliency_FN, "true_label": label_FN, "predicted_label": predicted_label_FN, "prediction_confidence": confidence_FN}

    return miss_predictions


def generate_shap_image(deep_explainer, test_image):

    shap_values = deep_explainer.shap_values(test_image)

    shap_values = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2)[0].sum(-1) for s in shap_values]
    test_image = np.swapaxes(np.swapaxes(test_image.cpu().numpy(), 1, -1), 1, 2)

    shap_img = np.zeros(test_image.shape[1:])

    for i in range(len(shap_values)):
        sv = shap_values[i]

        v_min, v_max = np.nanpercentile(sv[sv > 0], (1, 99))
        sv = exposure.rescale_intensity(sv, in_range=(v_min, v_max))

        sv = (sv - np.min(sv)) / (np.max(sv) - np.min(sv))

        if i == 0:
            index = 2
        if i == 1:
            index = 0

        shap_img[:, :, index] = sv

    return shap_img


def plot_pipeline_histogram(classifications, confidences, 
                            label_names = [],
                            condition = "", experiment_folder = "", clinical_isolate="", experiment_id = "",
                            show = True, legend = True):

    sns.reset_orig()
    
    total = len(classifications)
    
    classifications = np.asarray(classifications)
    confidences = np.asarray(confidences)

    cids = np.unique(classifications)
    
    if len(label_names) == 0:
        legend = False
        label_names = np.arange(len(cids))


    colours = ['orangered', 'dodgerblue']

    plot_text = ""

    f, ax = plt.subplots()

    for cid in cids:

        idx_cid = np.where(classifications == cid,True,False)
        confidences_cid = confidences[idx_cid]
        
        #Compute histogram
        values_cid,bins = np.histogram(confidences_cid, bins=40, range=(0.5,1.0), density=False)

        #Normalise by total counts in all classes
        density_cid = values_cid / (total * np.diff(bins))

        #Plot using default utilties. Renormalise weight such that we're not rebinning the already binned histogram.

        colour = colours[cid]
        name = label_names[cid]
        
        plt.hist(bins[:-1], 
                 bins, 
                 weights=density_cid,
                 edgecolor=colour,
                 color=colour,
                 histtype='stepfilled', 
                 alpha=0.2, 
                 label=name)

        proportion = np.sum(density_cid * np.diff(bins))
        
        plot_text += f"{len(confidences_cid)}  ({proportion*100:.0f}%)  {name}\n"
        
    plt.xlabel('Detection Confidence', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Normalised Frequency Density', fontsize=14)
    plt.grid(False)
    
    plt.text(0.05, 0.1, plot_text, fontsize=16,
              horizontalalignment='left',
              verticalalignment='center',
              transform = ax.transAxes)

    if legend:
        plt.legend(loc = 'upper left', fontsize=14)

    if experiment_folder == "":
        save_dir = "histogram"
    else:
        save_dir = os.path.join(experiment_folder,"histogram")
        
    save_dir = os.path.abspath(save_dir)
    
    if save_dir != "":
        
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
            
        file_name = f"AKDASP_histogram_{label_names}"
        
        if clinical_isolate != "":
            file_name = file_name + f"_[{clinical_isolate}]"
            save_dir = os.path.join(save_dir,str(clinical_isolate))
            
            if os.path.exists(save_dir) == False:
                os.mkdir(save_dir)
            
        if experiment_id != "":
            file_name = file_name + f"_[{experiment_id}]"

        if condition != "":
            file_name = file_name + f"_[{condition}]"

            
        file_name = file_name + ".png"
        
        save_path = os.path.join(save_dir,file_name)
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    
    if show:
        plt.show()

    plt.close()


def plot_pipeline_detections(pipeline_data, label_names, 
                             condition = "", experiment_folder = "", clinical_isolate="", experiment_id="", train_mode="",
                             show = False):
    
    for index,data in enumerate(pipeline_data):
    
        image = data["image"]
        cell_labels = data["cell_labels"]
        
        if len(cell_labels) > 0:
            
            label = data["cell_labels"][0]
            label_name = label_names[label]
            
            contours = data["cell_contours"]
            pred_labels = data["pred_labels"]
            
            img = np.moveaxis(np.stack([image[1]*0.8]*3), 0, -1)
            
            mask = np.zeros(image.shape[-2:], dtype=np.uint8)
            
            for i, (cnt, label) in enumerate(zip(contours, pred_labels)):
                
                if label == 0:
                      cv2.drawContours(mask, [cnt], -1, 1, 1)
                if label == 1:
                    cv2.drawContours(mask, [cnt], -1, 2, 1)
             
        img[mask==1,1] = 0.9
        img[mask==2,0] = 0.9
        
        plt.imshow(img)
        plt.axis(False)
        
        if experiment_folder == "":
            save_dir = "detections"
        else:
            save_dir = os.path.join(experiment_folder,"detections")
            
        save_dir = os.path.abspath(save_dir)
        
        if save_dir != "":
            
            if os.path.exists(save_dir) == False:
                os.mkdir(save_dir)
                
            save_dir = os.path.join(save_dir,str(label_names))
            
            if os.path.exists(save_dir) == False:
                os.mkdir(save_dir)
                
            file_name = f"AKDASP_detections_{label_name}"
            
            if train_mode != "":
                file_name = file_name + f"_[{train_mode}]"
                save_dir = os.path.join(save_dir,str(train_mode))
                
                if os.path.exists(save_dir) == False:
                    os.mkdir(save_dir)
            
            if clinical_isolate != "":
                file_name = file_name + f"_[{clinical_isolate}]"
                save_dir = os.path.join(save_dir,str(clinical_isolate))
                
                if os.path.exists(save_dir) == False:
                    os.mkdir(save_dir)
                
            if experiment_id != "":
                file_name = file_name + f"_[{experiment_id}]"

            if condition != "":
                file_name = file_name + f"_[{condition}]"
                
            file_name = file_name + f"_[{index}].png"
            
            save_path = os.path.join(save_dir,file_name)
        
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        
        if show:
            plt.show()
            
        plt.close()
        
        
def plot_publication_confusion_matrix(y_test, result, label_names = [], 
                                      condition = "", experiment_folder = "", clinical_isolate="",
                                      save=True, show=True):

    CM = confusion_matrix(y_test,result, normalize='true')
    CM_counts = confusion_matrix(y_test,result,normalize=None)
    
    #Seaborn plots sequential figures on top of each other. Use this to get multiple annotations

    CM_percentage = 100*CM
    processed_counts = CM_counts.flatten().tolist()
    processed_counts = ['({})'.format(elm) for elm in processed_counts]
    processed_counts = np.asarray(processed_counts).reshape((2,2))
    
    processed_percentage = np.asarray(np.rint(CM_percentage.flatten()),dtype='int').tolist()
    processed_percentage = ['{}%'.format(elm) for elm in processed_percentage]
    processed_percentage = np.asarray(processed_percentage).reshape((2,2))

    formatted_text = (np.asarray(["{}\n\n{}".format(
        data,text) for text, data in zip(processed_counts.flatten(), processed_percentage.flatten())])).reshape(2, 2)

    sns.set(font_scale=2.0)
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    plt.tight_layout()
    
    for i in range(CM.shape[0]):
        
        if i == 0:
            cmap = sns.light_palette((0, 75, 60), input="husl")
        else:
            cmap = sns.light_palette((260, 75, 60), input="husl")

        mask = np.ones(CM_percentage.shape)
        mask[:,i] = 0
        sns.heatmap(CM_percentage,linewidths=2, linecolor="black",  ax=ax,annot=formatted_text, cbar=False, cmap=cmap, vmin=0,vmax=100, fmt='', mask=mask)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels', fontsize=20)
    ax.set_ylabel('True labels', fontsize=20)
    
    if len(label_names) > 0:
        ax.xaxis.set_ticklabels(label_names, fontsize=20)
        ax.yaxis.set_ticklabels(label_names, fontsize=20)
    
    ax.axhline(y=0, color='k', linewidth=3)
    ax.axhline(y=CM_percentage.shape[1], color='k', linewidth=3)
    ax.axvline(x=0, color='k', linewidth=3)
    ax.axvline(x=CM_percentage.shape[0], color='k', linewidth=3)
    
    plt.tight_layout()
    
    if experiment_folder == "":
        save_dir = "confusion_matrix"
    else:
        save_dir = os.path.join(experiment_folder,"confusion_matrix")
        
    save_dir = os.path.abspath(save_dir)
    
    if save_dir != "":
        
        if os.path.exists(save_dir) == False:
            os.mkdir(save_dir)
        
        file_name = f"AKDASP_confusion_matrix_{label_names}"
        
        if clinical_isolate != "":
            file_name = file_name + f"_[{clinical_isolate}]"
        if condition != "":
            file_name = file_name + f"_[{condition}]"
            
        file_name = file_name + ".png"
        
        save_path = os.path.join(save_dir,file_name)
        
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    
    if show:
        plt.show()
        
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, classes, show_percent = True, num_samples=1, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, save_path=None):
    
    cm = confusion_matrix(true_labels, pred_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    accuracy = len(np.where(np.array(true_labels) == np.array(pred_labels))[0]) / len(true_labels)

    balanced_accuracy = balanced_accuracy_score(true_labels, pred_labels)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title + "\n" + f"N: {len(true_labels)}\nAccuracy: {accuracy:.2f}\nBalanced Accuracy: {balanced_accuracy:.2f}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90, ha='center', rotation_mode='anchor')
    plt.tick_params(axis='y', which='major', pad=10)

    # colour_mapping={'Untreated':sns.light_palette((0, 75, 60), input="husl"), 'GENT':sns.light_palette((260, 75, 60), input="husl")}

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        positions = np.where(np.array(true_labels) == i)[0]
        conf_pred_labels = np.array(pred_labels)[positions]
        num_labels = len(np.where(conf_pred_labels == j)[0])

        if i == 0:
            cmap = sns.light_palette((0, 75, 60), input="husl")
        else:
            cmap = sns.light_palette((260, 75, 60), input="husl")
                
        if len(positions) == 0:
            accuracy = 0
        else:
            accuracy = num_labels / len(positions)

        if show_percent:
            plt.text(j, i, f"{accuracy*100:.0f} %" + "\n(" + str(num_labels) + ")",
                     verticalalignment="center",
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, f"{accuracy:.2f}" + " \n(" + str(num_labels) + ")",
                     verticalalignment="center",
                     horizontalalignment="center", 
                     color="white" if cm[i, j] > thresh else "black")
            

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    #     plt.show()
    #     image = Image.open(save_path)
    #     ar = np.asarray(image)
    # else:
    #     with io.BytesIO() as buffer:
    #         plt.savefig(buffer, format="raw", bbox_inches='tight', pad_inches=0, dpi=300)
    #         plt.show()
    #         image = Image.open(buffer)
    #         ar = np.asarray(image)

    # plt.close()

    return


def process_image(img):
    for i in range(img.shape[0]):
        try:
            im = img[i]
            im = rescale01(im) * 255
            im = im.astype(np.uint8)

            im = normalize99(im)
            im = rescale01(im)
            img[i] = im
        except:
            pass
    img = np.swapaxes(img, 0, 2)

    return img


def generate_prediction_images(miss_predictions, model_dir):

    for prediction_type, data in miss_predictions.items():


        images, saliency_map, confidence = data["images"], data["saliency_map"], data["prediction_confidence"]
        predicted_label, true_label = data['predicted_label'], data['true_label']

        if len(images) > 0:
            if true_label == 'None':
                true_label = 'Untreated'
            if predicted_label == 'None':
                predicted_label = 'Untreated'

            images, saliency_map, confidence = [list(x) for x in zip(*sorted(zip(images, saliency_map, confidence), key=lambda x: x[2]))]

            images_highconferr = images[-5:]
            saliency_highconferr = saliency_map[-5:]
            confidence_highconferr = confidence[-5:]
            images_lowconferr = images[:5]
            saliency_lowconferr = saliency_map[:5]
            confidence_lowconferr = confidence[:5]

            images_highconferr = np.hstack(images_highconferr)
            saliency_highconferr = np.hstack(saliency_highconferr)
            images_lowconferr = np.hstack(images_lowconferr)
            saliency_lowconferr = np.hstack(saliency_lowconferr)

            combined_image = np.concatenate((images_highconferr, saliency_highconferr, images_lowconferr, saliency_lowconferr))

            name_mod = ''.join([word[0] for word in prediction_type.split(" ")])
            name_mod = name_mod + '_figs.tif'

            plot_save_path = pathlib.Path('').joinpath(*model_dir.parts, "test_prediction_images", name_mod)
            if not os.path.exists(os.path.dirname(plot_save_path)):
                os.makedirs(os.path.dirname(plot_save_path))

            plt.imshow(combined_image)
            tickmarks = [(combined_image.shape[0] / 4) * 1, (combined_image.shape[0] / 4) * 3]
            plt.yticks(tickmarks, ["Highest Confidence", "Lowest Confidence"], rotation=90, ha='center', rotation_mode='anchor', fontsize=8)
            plt.xticks([])
            plt.tick_params(axis='y', which='major', pad=20)

            plt.title(f"{prediction_type}. Label: {true_label}, Predicted Label: {predicted_label}", fontsize=10)
            plt.savefig(plot_save_path, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.show()
            plt.close()


def generate_plots(model_data, save_path, model_directory):

    antibiotic = model_data["antibiotic"]
    channel_list = model_data["channel_list"]
    cm = model_data["confusion_matrix"]
    num_samples = model_data["num_test_images"]
    true_labels, pred_labels = model_data["test_labels"], model_data["pred_labels"]

    test_predictions = model_data["test_predictions"]

    condition = [antibiotic] + channel_list
    condition = '[' + '-'.join(condition) + ']'
    classes = ["Untreated", antibiotic]

    cm_path = save_path + "_confusion_matrix.tif"
    loss_graph_path = save_path + "_loss_graph.tif"
    accuracy_graph_path = save_path + "_accuracy_graph.tif"

    plot_confusion_matrix(true_labels, pred_labels, classes, num_samples=num_samples, normalize=True, title="Confusion Matrix: " + condition, save_path=cm_path)

    train_loss = model_data["training_loss"]
    validation_loss = model_data["validation_loss"]
    train_accuracy = model_data["training_accuracy"]
    validation_accuracy = model_data["validation_accuracy"]

    plt.plot(train_loss, label="training loss")
    plt.plot(validation_loss, label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.legend(loc="upper right")
    plt.title("Loss Graph: " + condition)
    plt.savefig(loss_graph_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    plt.plot(train_accuracy, label="training accuracy")
    plt.plot(validation_accuracy, label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    plt.title("Accuracy Graph: " + condition)
    plt.savefig(accuracy_graph_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    generate_prediction_images(test_predictions, model_directory)
    plot_test_stat_correlations(model_data, model_directory)


def plot_test_stat_correlations(model_data, model_dir):

    test_stats = model_data["test_stats"]
    test_labels = model_data["test_labels"]
    pred_confidences = model_data["pred_confidences"]

    if "antibiotic_list" in model_data.keys():
        label_names = model_data["antibiotic_list"]
    else:
        label_names = np.unique(test_labels)

    for stat_name in test_stats[0].keys():

        model_dir = pathlib.Path(model_dir)

        plot_save_path = pathlib.Path('').joinpath(*model_dir.parts, "test_stat_correlations", f"{stat_name}_distribution.tif")
        if not os.path.exists(os.path.dirname(plot_save_path)):
            os.makedirs(os.path.dirname(plot_save_path))

        fig, ax = plt.subplots()

        for label in np.unique(test_labels):
            label_indices = np.where(test_labels == label)[0]

            plot_confidences = np.take(pred_confidences, label_indices)
            plot_stats = [dat[stat_name] for dat in np.take(test_stats, label_indices)]

            ax.scatter(plot_stats, plot_confidences, label=label_names[label])
            ax.set_xlabel(stat_name)
            ax.set_ylabel('Prediction Confidence')

        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()

