
import numpy as np
from scipy.optimize import linear_sum_assignment
from multiprocessing import Pool
import tqdm
from sklearn.metrics import confusion_matrix,f1_score, recall_score, precision_score, accuracy_score


def generate_results_summary(true_labels, pred_labels, label_list):

    cm = confusion_matrix(true_labels, pred_labels)
    
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # F1
    F1 = (2*PPV*TPR)/(PPV+TPR)
    
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    result_dict = {}
    
    for i, label in enumerate(label_list):
        
        label = str(label)
        
        if label not in result_dict.keys():
            result_dict[label] = {}
            
        result_dict[f"{label}"]["Accuracy"] = ACC[i]
        result_dict[f"{label}"]["Recall"] = TPR[i]
        result_dict[f"{label}"]["Specificity"] = TNR[i]
        result_dict[f"{label}"]["F1"] = F1[i]
        result_dict[f"{label}"]["False Negative Rate"] = FNR[i]
        result_dict[f"{label}"]["False Positive Rate"] = FPR[i]
        
    result_dict["Mean Accuracy"] = np.mean(ACC)
    result_dict["Mean Recall"] = np.mean(TPR)
    result_dict["Mean Specificity"] = np.mean(TNR)
    result_dict["Mean False Negative Rate"] = np.mean(FNR)
    result_dict["Mean False Positive Rate"] = np.mean(FPR)

    print(accuracy_score(true_labels, pred_labels))
    print(precision_score(true_labels, pred_labels, average="micro"))
    print(recall_score(true_labels, pred_labels, average="micro"))
    print(f1_score(true_labels, pred_labels, average="micro"))

    return result_dict


def calculate_mask_ap(dat):
    
    try:
    
        mask_true, mask_pred, threshold_list = dat
        
        n_true = np.max(mask_true)
        n_pred = np.max(mask_pred)
        
        ap_list = []
        ar_list = []

            
        if n_pred > 0:
            
            iou = intersection_over_union(mask_true, mask_pred)[1:, 1:]  
            
            for threshold in threshold_list:
                
                tp = true_positive(iou, threshold)
                
                fp = n_pred - tp
                fn = n_true - tp
                
                ar = tp / (tp + fn)
                ap = tp / (tp + fp)
                
                ap_list.append(ap)
                ar_list.append(ar)
            
        else:
            ap, ar = 0,0
    
    except:
        threshold, ap_list, ar_list = [None], [None], [None]
        
    return threshold_list, ap_list, ar_list

def calculate_average_precision(masks_true, masks_pred, threshold_list = []):
    
    try:
    
        if len(threshold_list) == 0:
            threshold_list = np.arange(0.5, 1, 0.01)
        
        data = []
        
        for index, (mask_true, mask_pred) in enumerate(zip(masks_true, masks_pred)):
            data.append([mask_true, mask_pred, threshold_list])
    
    
        print(f"\nCalculating mAP for {len(data)} masks...")
    
        with Pool() as pool:
            
            results = tqdm.tqdm(pool.imap(calculate_mask_ap, data), total = len(data))
            
            pool.close()
            pool.join()
        
        results = list(results)
        
        threshold_list, ap_list, ar_list = zip(*results)
        
        threshold_list = np.array([item for sublist in threshold_list for item in sublist if item != None])
        ap_list = np.array([item for sublist in ap_list for item in sublist if item != None])
        ar_list = np.array([item for sublist in ar_list for item in sublist if item != None])
        
    
        ap_results = []
        ar_results = []
    
        for threshold in np.unique(threshold_list):
            
            ap = np.nanmean(np.take(ap_list, np.argwhere(threshold_list==threshold)))
            ar = np.nanmean(np.take(ar_list, np.argwhere(threshold_list==threshold)))
            
            ap_results.append([threshold, ap])
            ar_results.append([threshold, ar])
    
        ap_results = np.stack(ap_results)
        ar_results = np.stack(ar_results)
        
        
        mAP50 = np.take(ap_results[:,1],
                        np.argwhere(np.array(ap_results)[:,0] == 0.5)[0])[0]
        
        mAP5095 = np.take(ap_results[:,1],
                          np.argwhere((np.array(ap_results)[:,0] >= 0.5) & 
                                      (np.array(ap_results)[:,0] <= 0.95))[:,0]).mean()
    
        ar_results = np.stack(ar_results)
        
        
        mAR50 = np.take(ar_results[:,1],
                        np.argwhere(np.array(ar_results)[:,0] == 0.5)[0])[0]
        
        mAR5095 = np.take(ar_results[:,1],
                          np.argwhere((np.array(ar_results)[:,0] >= 0.5) & 
                                      (np.array(ar_results)[:,0] <= 0.95))[:,0]).mean()
    
        results = {}
        
        results["average_precision_data"] = ap_results
        results["mAP50"] = mAP50
        results["mAP5095"] = mAP5095
        results["average_recall_data"] = ar_results
        results["mAR50"] = mAR50
        results["mAR5095"] = mAR5095
        
    except:
        results = {}
        pass
    
    return results



















def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9]):
    
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    
    if len(masks_true) != len(masks_pred):
        raise ValueError('metrics.average_precision requires len(masks_true)==len(masks_pred)')

    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    n_true = np.array(list(map(np.max, masks_true)))
    n_pred = np.array(list(map(np.max, masks_pred)))
    
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])  
        
    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
        
    return ap, tp, fp, fn


def intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]
    
    ------------
    How it works:
        The overlap matrix is a lookup table of the area of intersection
        between each set of labels (true and predicted). The true labels
        are taken to be along axis 0, and the predicted labels are taken 
        to be along axis 1. The sum of the overlaps along axis 0 is thus
        an array giving the total overlap of the true labels with each of
        the predicted labels, and likewise the sum over axis 1 is the
        total overlap of the predicted labels with each of the true labels.
        Because the label 0 (background) is included, this sum is guaranteed
        to reconstruct the total area of each label. Adding this row and
        column vectors gives a 2D array with the areas of every label pair
        added together. This is equivalent to the union of the label areas
        except for the duplicated overlap area, so the overlap matrix is
        subtracted to find the union matrix. 

    """
    overlap = label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou


def true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold
        
    ------------
    How it works:
        (1) Find minimum number of masks
        (2) Define cost matrix; for a given threshold, each element is negative
            the higher the IoU is (perfect IoU is 1, worst is 0). The second term
            gets more negative with higher IoU, but less negative with greater
            n_min (but that's a constant...)
        (3) Solve the linear sum assignment problem. The costs array defines the cost
            of matching a true label with a predicted label, so the problem is to 
            find the set of pairings that minimizes this cost. The scipy.optimize
            function gives the ordered lists of corresponding true and predicted labels. 
        (4) Extract the IoUs fro these parings and then threshold to get a boolean array
            whose sum is the number of true positives that is returned. 

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp


def label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    # put label arrays into standard form then flatten them 
#     x = (utils.format_labels(x)).ravel()
#     y = (utils.format_labels(y)).ravel()
    x = x.ravel()
    y = y.ravel()
    
    # preallocate a 'contact map' matrix
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    
    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image 
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap



