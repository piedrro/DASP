import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split

def balance_dataset(dataset):
    
    if len(dataset["images"]) > 0:
        
        data_sort = dataset["labels"]
        unique, counts = np.unique(data_sort, return_counts=True)
        
        max_count = np.min(counts)
    
        balanced_dataset = {}
    
        for unique_key in unique:
    
            unique_indices = np.argwhere(np.array(data_sort) == unique_key)[:,0].tolist()[:max_count]
    
            for key, value in dataset.items():
    
                if key not in balanced_dataset.keys():
                        balanced_dataset[key] = []
    
                balanced_dataset[key].extend([value[index] for index in unique_indices])

    else:
        
        balanced_dataset = dataset

    return balanced_dataset


def shuffle_train_data(train_data):
    
    if len(train_data["labels"]) > 0:
        
        dict_names = list(train_data.keys())     
        dict_values = list(zip(*[value for key,value in train_data.items()]))
        
        random.seed(4)
                
        random.shuffle(dict_values)
        
        dict_values = list(zip(*dict_values))
        
        train_data = {key:list(dict_values[index]) for index,key in enumerate(train_data.keys())}
    
    return train_data
                    

def limit_train_data(train_data, num_files):
    
    for key,value in train_data.items():
        
        train_data[key] = value[:num_files]
        
    return train_data

def check_label_format(dataset, label_names):
    
    if len(dataset["images"]) > 0:
    
        dataset_label_names = dataset["label_names"]
            
        dataset_labels = [label_names.index(label) for label in dataset_label_names]
        
        dataset["labels"] = dataset_labels
    
    return dataset


def get_training_data(cached_dataset, mode = "",
                      holdout_experiment_id = 7, test_experiment_id = "", label_names = [], condition_list = [], experiment_list = [], concentration_list = [], species_list = [],
                      dataset_ratios = [0.8, 0.1, 0.1], test_file_names = [],
                      label_limit = "None", xvalidation_label_limit = "None", holdout_label_limit = "None",
                      shuffle = True,  balance = False, verbose=True, check_labels = True):

    experiment_ids = np.array(cached_dataset["experiment_ids"])
    label_list = cached_dataset.pop("label_list")
    
    if label_names == []:
        label_names = sorted(list(np.unique(cached_dataset["label_names"])))
                    
    if condition_list == []:
        condition_list = sorted(list(np.unique(cached_dataset["condition"])))
        
    if experiment_list == []:
        experiment_list = sorted(list(np.unique(cached_dataset["experiment_ids"])))
    
    if species_list == []:
        species_list = sorted(list(np.unique(cached_dataset["species_ids"])))
        
    if concentration_list == []:
        concentration_list = sorted(list(np.unique(cached_dataset["treatment_concentrations"])))

    if holdout_experiment_id not in experiment_list:
        experiment_list.append(holdout_experiment_id)
        
    if type(label_limit) != int:
        label_limit = "None"
        
        
    train_indcies_list = []
    val_indices_list = []
    test_indcies_list = []
    holdout_indices_list = []
    evaluation_indicies_list = []
    
    train_data = {}
    val_data = {}
    test_data = {}
    holdout_data = {}
    evaluation_data = {}

    if mode == "xvalidation" or mode == "holdout":
        
        data_sort = pd.DataFrame(cached_dataset)
        
        data_sort = data_sort[data_sort["label_names"].isin(label_names)]
        data_sort = data_sort[data_sort["condition"].isin(condition_list)]
        data_sort = data_sort[data_sort["experiment_ids"].isin(experiment_list)]
        data_sort = data_sort[data_sort["species_ids"].isin(species_list)]
        data_sort = data_sort[data_sort["treatment_concentrations"].isin(concentration_list)]
        
        if label_names.index("WT+ETOH") != 0:
            label_names.insert(0, label_names.pop(label_names.index("WT+ETOH")))
            
        if "images" in cached_dataset.keys():
            data_sort.drop(labels = ["images"], axis=1, inplace=True)
        if "masks" in cached_dataset.keys():
            data_sort.drop(labels = ["masks"], axis=1, inplace=True)
            
        holdout_data_sort = data_sort.copy()
        
        data_sort = data_sort[data_sort["experiment_ids"] != holdout_experiment_id]
        holdout_data_sort = holdout_data_sort[holdout_data_sort["experiment_ids"] == holdout_experiment_id]
        
        data_sort_groups = data_sort.groupby(["experiment_ids","labels"])
        holdout_data_sort_groups = holdout_data_sort.groupby(["labels"])
        
        experiment_list = sorted(list(data_sort.experiment_ids.unique()))
        
        if test_experiment_id == "":
            test_experiment_id = holdout_experiment_id
    
        for i in range(len(data_sort_groups)):
        
            data = data_sort_groups.get_group(list(data_sort_groups.groups)[i])
            
            experiment_id = data.experiment_ids.unique()[0]
            
            if experiment_id != test_experiment_id:
                
                train_indices = data.index.values.tolist()
                             
                train_size = dataset_ratios[0] / sum(dataset_ratios[:2])
                
                train_indices, val_indices = train_test_split(train_indices,
                                                              train_size=train_size,
                                                              random_state=42,
                                                              shuffle=True)
                
                if label_limit != "None":
                    train_indices = train_indices[:label_limit]
                    val_indices = val_indices[:label_limit]
                
                train_indcies_list.extend(train_indices)
                val_indices_list.extend(val_indices)
    
            else:
            
                test_indices = data.index.values.tolist()
                test_indcies_list.extend(test_indices)
              
        for i in range(len(holdout_data_sort_groups)):
        
            data = holdout_data_sort_groups.get_group(list(holdout_data_sort_groups.groups)[i])
            
            holdout_indices = data.index.values.tolist()
            
            if holdout_label_limit != "None":
                holdout_indices = holdout_indices[:holdout_label_limit]
            
            holdout_indices_list.extend(holdout_indices)
             
            
    elif mode == "ensemble":
        
        data_sort = pd.DataFrame(cached_dataset)
        
        data_sort = data_sort[data_sort["label_names"].isin(label_names)]
        data_sort = data_sort[data_sort["condition"].isin(condition_list)]
        data_sort = data_sort[data_sort["experiment_ids"].isin(experiment_list)]
        data_sort = data_sort[data_sort["species_ids"].isin(species_list)]
        data_sort = data_sort[data_sort["treatment_concentrations"].isin(concentration_list)]
        
        label_names = data_sort["label_names"].unique().tolist()
        
        if label_names.index("WT+ETOH") != 0:
            label_names.insert(0, label_names.pop(label_names.index("WT+ETOH")))
        
        if "images" in cached_dataset.keys():
            data_sort.drop(labels = ["images"], axis=1, inplace=True)
        if "masks" in cached_dataset.keys():
            data_sort.drop(labels = ["masks"], axis=1, inplace=True)
        
        holdout_data_sort = data_sort.copy()
        
        holdout_data_sort = holdout_data_sort[holdout_data_sort["experiment_ids"] == holdout_experiment_id]
        data_sort = data_sort[data_sort["experiment_ids"] != holdout_experiment_id]

        data_sort_groups = data_sort.groupby(["labels","experiment_ids"])
        holdout_data_sort_groups = holdout_data_sort.groupby(["labels"])
        
        for i in range(len(data_sort_groups)):
            
            data = data_sort_groups.get_group(list(data_sort_groups.groups)[i])
            
            experiment_ids = list(data.experiment_ids.unique())

            if holdout_experiment_id not in experiment_ids:
            
                dataset_indices = data.index.tolist()
                
                train_indices, test_indices = train_test_split(dataset_indices,
                                                               train_size = 1 -dataset_ratios[-1],
                                                               random_state=42,
                                                               shuffle=True)
                
                train_size = dataset_ratios[0] / sum(dataset_ratios[:2])
            
                train_indices, val_indices = train_test_split(train_indices,
                                                              train_size=train_size,
                                                              random_state=42,
                                                              shuffle=True)
                
                if label_limit != "None":
                    
                    train_indices = train_indices[:label_limit]
                    val_indices = val_indices[:label_limit]
                    test_indices = test_indices[:label_limit]    
                
                train_indcies_list.extend(train_indices)
                val_indices_list.extend(val_indices)
                test_indcies_list.extend(test_indices)

        for i in range(len(holdout_data_sort_groups)):
        
            data = holdout_data_sort_groups.get_group(list(holdout_data_sort_groups.groups)[i])
            
            experiment_ids = list(data.experiment_ids.unique())
            
            holdout_indices = data.index.values.tolist()

            if holdout_label_limit != "None":
                holdout_indices = holdout_indices[:holdout_label_limit]
                
            holdout_indices_list.extend(holdout_indices)

    elif mode == "evaluate":
        
        data_sort = pd.DataFrame(cached_dataset)
        
        data_sort = data_sort[data_sort["label_names"].isin(label_names)]
        # print(len(data_sort),label_names)
        data_sort = data_sort[data_sort["condition"].isin(condition_list)]
        # print(len(data_sort),condition_list)
        data_sort = data_sort[data_sort["experiment_ids"].isin(experiment_list)]
        # print(len(data_sort),experiment_list)
        data_sort = data_sort[data_sort["species_ids"].isin(species_list)]
        # print(len(data_sort),species_list)
        data_sort = data_sort[data_sort["treatment_concentrations"].isin(concentration_list)]
        # print(len(data_sort),concentration_list)
        
        if "images" in cached_dataset.keys():
            data_sort.drop(labels = ["images"], axis=1, inplace=True)
        if "masks" in cached_dataset.keys():
            data_sort.drop(labels = ["masks"], axis=1, inplace=True)
        
        data_sort_groups = data_sort.groupby(["experiment_ids","labels"])
        
        for i in range(len(data_sort_groups)):

            data = data_sort_groups.get_group(list(data_sort_groups.groups)[i])
            
            train_indices = data.index.values.tolist()
            
            if label_limit != "None":
                train_indices = train_indices[:label_limit]
                
            evaluation_indicies_list.extend(train_indices)
     
       
    elif mode == "az":
        
        data_sort = pd.DataFrame(cached_dataset)
        
        data_sort = data_sort[data_sort["file_names"].isin(test_file_names)]
        
        data_sort_groups = data_sort.groupby(["experiment_ids","labels"])
        
        for i in range(len(data_sort_groups)):

            data = data_sort_groups.get_group(list(data_sort_groups.groups)[i])
            
            train_indices = data.index.values.tolist()
            
            evaluation_indicies_list.extend(train_indices)

    for key,value in cached_dataset.items():
        
        if key in ["images","masks"]:
            
            train_dat = list(np.array(value)[train_indcies_list])
            val_dat = list(np.array(value)[val_indices_list])
            test_dat = list(np.array(value)[test_indcies_list])
            holdout_dat = list(np.array(value)[holdout_indices_list])
            evaluation_dat = list(np.array(value)[evaluation_indicies_list])
            
        else:

            train_dat = np.take(np.array(value), train_indcies_list).tolist()
            val_dat = np.take(np.array(value), val_indices_list).tolist()
            test_dat = np.take(np.array(value), test_indcies_list).tolist()
            holdout_dat = np.take(np.array(value), holdout_indices_list).tolist()
            evaluation_dat = np.take(np.array(value), evaluation_indicies_list).tolist()
                        
        if key in train_data.keys():
      
          train_data[key].extend(train_dat)
          val_data[key].extend(val_dat)
          test_data[key].extend(test_dat)
          holdout_data[key].extend(holdout_dat)
          evaluation_data[key].extend(evaluation_dat)
  
        else:
          
          train_data[key] = train_dat
          val_data[key] = val_dat
          test_data[key] = test_dat
          holdout_data[key] = holdout_dat
          evaluation_data[key] = evaluation_dat
          
    if balance == True:
        train_data = balance_dataset(train_data)
        val_data = balance_dataset(val_data)
        test_data = balance_dataset(test_data)
        evaluation_data = balance_dataset(evaluation_data)

    holdout_data = balance_dataset(holdout_data)


    if shuffle == True:
        train_data = shuffle_train_data(train_data)    
        val_data = shuffle_train_data(val_data)
        test_data = shuffle_train_data(test_data)
        holdout_data = shuffle_train_data(holdout_data)
        evaluation_data = shuffle_train_data(evaluation_data)
                
    train_data["label_list"] = label_names
    val_data["label_list"] = label_names
    test_data["label_list"] = label_names
    holdout_data["label_list"] = label_names
    evaluation_data["label_list"] = label_names
    
    if check_labels:
        train_data = check_label_format(train_data, label_names)
        val_data = check_label_format(val_data, label_names)
        test_data = check_label_format(test_data, label_names)
        holdout_data = check_label_format(holdout_data, label_names)
        evaluation_data = check_label_format(evaluation_data, label_names)
    
    if verbose:
        num_train = len(train_data["images"])
        num_val = len(val_data["images"])
        num_test = len(test_data["images"])
        num_holdout = len(holdout_data["images"])
        num_eval = len(evaluation_data["images"])
        
        train_labels = list(np.unique(train_data["labels"]))
        val_labels = list(np.unique(val_data["labels"]))
        test_labels = list(np.unique(test_data["labels"]))
        holdout_labels = list(np.unique(holdout_data["labels"]))
        evaluation_labels = list(np.unique(evaluation_data["labels"]))
        
        train_experiments = list(np.unique(train_data["experiment_ids"]))
        val_experiments = list(np.unique(val_data["experiment_ids"]))
        test_experiments = list(np.unique(test_data["experiment_ids"]))
        holdout_experiments = list(np.unique(holdout_data["experiment_ids"]))
        evaluation_experiments = list(np.unique(evaluation_data["experiment_ids"]))
        
        print(f"train size: [{num_train}], validation size: [{num_val}], test_size: [{num_test}], num holdout: [{num_holdout}], num eval: [{num_eval}]")
        print(f"train labels: {train_labels}, validation labels: {val_labels}, test labels: {test_labels}, holdout labels: {holdout_labels}, eval labels: {evaluation_labels}")
        print(f"train experiments: {train_experiments}, validation experiments: {val_experiments}, test experiments: {test_experiments}, holdout experiments: {holdout_experiments}, eval experiments: {evaluation_experiments}")
    
    cached_dataset["label_list"] = label_list
    
    datasets = dict(train_data = train_data,
                    val_data = val_data,
                    test_data = test_data,
                    holdout_data = holdout_data,
                    evaluation_data = evaluation_data)
    
    return datasets



