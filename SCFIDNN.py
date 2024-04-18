
"""
Classifies whether a strain (in this case E.coli) is resistant (1) or susceptible (0) when treated with Kanamycin, Trimethorpim, and Gentamycin, 
based on data gathered from the Sub-Cellular fluctuation technique developed at the University of Bristol.

NEEDED for the code to run: Folder with the saved arrays of the input parameters, as well as the experimental spreadsheets generated from running Arthur's code.

Running this code presents various options - choice of antibiotic, choice of DNN method, choice of ACF fit as well as a choice between different input parameters. 
More details at the bottom of the code.
"""




import os
import shutil
import re
import numpy as np
from matplotlib.pyplot import subplots
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import sys

import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split, SubsetRandomSampler
from torchvision.transforms import ToTensor
#from ISLP.torch import (SimpleDataModule, SimpleModule, ErrorTracker) 
from torchinfo import summary  
#from torch.optim import RMSprop   

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from pytorch_lightning import Trainer           
from pytorch_lightning.loggers import CSVLogger 
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
#from ray.train import Checkpoint
#from ray.tune.suggest.hyperopt import BayesOptSearch

import shap
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit

import time

start_time = time.time()


########################################################################
#Pre-processing
#-----------------------------------------------------------------------
#Needed: the Saved Array folders and the Excel spreadsheets generated after Arthur's fitting code is run
########################################################################

def param_data_processing( save = "no", split_timeseries = False, split_ACF = False, flatten = False, cross_val = False): 

    #intensities added seperately to make sure it's loaded in cases where it's needed for normalisation, even if don't want it as NN input
    #will be removed so that not inputted into NN - if want as input, specify in nn_params. 
    if "intensities" not in nn_params and any(substring in nn_params for substring in ["_m_mono_", "_m1_", "_m2_", "_g0_"]):
        load_intensities = True
        params = nn_params + ["intensities"]

    else: 
        load_intensities = False
        params = nn_params

    #will normalise g0 or the m's with intensity**power
    power = 2

    #directory where the Saved arrays are
    directory = "D:/Saved arrays"

    total_X_list = []
    total_Y_list = []

    #this is for the case where we don't want to cross validate later but want to train on subsection of Gentamycin (the last repeat in each folder)    
    #and then test on the first two repeats in each folder
    total_X_train_list = []
    total_X_test_list = []
    total_Y_train_list = []
    total_Y_test_list = []

    
    #this sets which of the res_sus_datanames (see above) to loop over, depending on the chosen antibiotic
    if antibiotic == "Gentamycin": 
        low = 0
        up = 10
    elif antibiotic == "Kanamycin":
        low = 10
        up = 16
    elif antibiotic == "Trimethoprim":
        low = 16
        up = 22
    elif antibiotic == "all": 
        low = 0
        up = 22
    
    #Main processing loop
    folder_list = list(range(low, up))

    if equal_Genta == True and antibiotic == "Gentamycin": 
        folder_list = [0, 1,  4,  7]

    for s in folder_list:
        #[0, 1, 4, 7]:
        print("Doing preprocessing for: ", res_sus_datanames[s])
        
        #gets the Saved arrays folder name. for cases where "_r" or "_s" si in the name, it removes this 
        #to get the original folder name - whether there's an _r or _s will be important for later when only 
        #either the resistant or the susceptible files are kept
        if re.search(r'_r|_s', res_sus_datanames[s]):
            folder = res_sus_datanames[s][:-2]
        else: 
            folder = res_sus_datanames[s]

        strain_arrays_dir = os.path.join(directory, folder)
        path, dirs, files = next(os.walk(strain_arrays_dir))

        #sorts files by row nr
        p1 = re.compile(r'row_(\d+)')
        sorted_files = sorted(files, key=lambda s: int(p1.search(s).groups()[0]))
    

        #only keeps files of selected parameters
        fit_files = []
        pattern = re.compile("|".join(re.escape( p )for p in params))

        if "R" not in params: 
            #this is necessary because "R_tD" (saved R_squared file name for some) would be mistaken for tD by the pattern finder

            R_pattern = re.compile(r'R')
            fit_files = [file for file in sorted_files if (pattern.search(file) and not (R_pattern.search(file)))]

        else: 
            fit_files = [file for file in sorted_files if (pattern.search(file))]

        #to check for errors
        t = "".join(fit_files[:len(params)])
        if s == low:
            print("Selected files of type: ", fit_files[:len(params)])

        for el in params: 
            if el not in t:
                sys.exit("Parameter files were not selected - probably because they're not in your folder!")
        
        #gets paths for the parameter files
        ordered_array_paths = []
        for el in fit_files:
            ordered_array_paths.append(os.path.join(strain_arrays_dir, el))

        #creates sublists for every param within the total list
        row_list = [ordered_array_paths[i:i+len(params)] for i in range(0, len(ordered_array_paths), len(params))]

        #finds out how many rows there are for each type (T/UT (Gentamycin), tr/ts/ur/ur(kanamycin, trimethoprim)), via the Excel spreadsheets for that saved array folder
        spreadsheets_dir = "D:\Data spreadsheets"
        sheet_dir = os.path.join(spreadsheets_dir, folder +  ".xlsx")
        df = pd.read_excel(sheet_dir)
        col_vals = df["dividers"]
        col_vals = col_vals[col_vals != 0]
        div_list = col_vals.tolist()


        #divides the rows correctly into treated and untreated
        #I will not be using the untreated as normalising the treated with the untreated made the NN classify worse

        #s <10 is Gentamycin data: 
        #treated ones are the first three groups, untreated the last three groups
        if s < 10:
            treated_rows = row_list[:div_list[2]] 
            untreated_rows = row_list[div_list[2]: div_list[-1]] 


        #these are for Kanamycin and Timethoprim
        #order of rows for both is tr, ts, ur, us
        elif re.search(r'_r', res_sus_datanames[s]):
            treated_rows = row_list[:div_list[0]]
            untreated_rows = row_list[div_list[1]:div_list[2]]

        elif re.search(r'_s', res_sus_datanames[s]):
            treated_rows = row_list[div_list[0]:div_list[1]]
            untreated_rows =row_list[div_list[2]: div_list[-1]]

        row_list = treated_rows

        #loads the files and creates array with all data
        loaded_arrays = []
        for row in row_list: 

            loaded_row = [np.load(file_path) for file_path in row]
            loaded_arrays.append(loaded_row)

            
            if "timeseries" in nn_params:
                #some videos have different lengths, especially different for Gentamycin. This gets a uniform length for them.
                if antibiotic == "Gentamycin":
                    loaded_row[0] = loaded_row[0][:150]
                    l = loaded_row[0]
                else:
                    diff = 401- len(loaded_row[0])
                    if diff != 0:  
                        last_el = loaded_row[0][-1]
                        new_row = np.expand_dims(last_el, axis=0)
                        for i in range(diff): 
                            loaded_row[0] = np.concatenate((loaded_row[0], new_row), axis=0)
                        l = loaded_row[0]
            


        arr_loaded_arrays = np.array(loaded_arrays)
        #t = arr_loaded_arrays[0]
        
        #Divides any g0 or m by intensity-squared

        #print(arr_loaded_arrays[0][1][0])
        if load_intensities == True:

            print("Normalising with intensity...")

            #gets index of intensity column in array by seeing at which index intensity first appears in fit files
            for el in fit_files: 
                if "intensities" in el: 
                    i_index = fit_files.index(el)
                    break

            i_squared = (arr_loaded_arrays[0:, i_index])**power

            for sp in params: 
                #indexing is not gonna work if m1 and m2 in params, need to adapt later
                if sp in ["_m_mono_", "_m1_", "_m2_", "_g0_"]:
                    for el in fit_files: 
                        if sp in el: 
                            index = fit_files.index(el)
                            break
                    normed_p = arr_loaded_arrays[0:, index]/i_squared
                    arr_loaded_arrays[0:, index] = normed_p

        

        #print(arr_loaded_arrays[0][1][0])

        """

        test if array transformations/calculations correct. if a/b and c are the same then yes. 
        a = arr_loaded_arrays[0][1]
        b = i_squared[0]
        c = normed_m[0][0]
        print(a/b)
        print(c) 
        """

        #normalise treated with untreated
        """
        #question: in one column: why median of array in one row? and then mean of entire column? (instead of mean both times?)
        #to not skew results
        def produce_norm_val(arrays):
        """
        """

            Takes an ndarray as input. 
            Returns the average value of every column of the array (every parameter).
            For every column (every parameter), the function calculates the median of the array in every row 
            and then takes the mean of these medians, giving one value per column (parameter).
        """
        """

            #prob need to change this for ACF

            #rows = np.array(rows)
            results = {}
            for p in range(len(sorted_params)):
                parameter = sorted_params[p] 
                results[parameter] = [] 
                avgs = []
                for row in range(len(arrays)):
                    #print(rows[row number, column number])
                    arr = arrays[row, p]
                    avgs.append(np.median(arr.flatten()))
                results[parameter].append(np.mean(avgs))
            return results
        
        
        norm_vals = produce_norm_val(untreated_array)
        print(norm_vals)

       
        #divide each param in treated array by average of untreated equivalent.
        for i in range(len(norm_vals)):
            p_val = list(norm_vals.values())[i]
            treated_array[0:, i] = treated_array[0:, i]/p_val
        """
        
            
        #remove intensity from treated array in case it's not wanted as an input to the NN.
        treated_array = arr_loaded_arrays
        if load_intensities == True: 
            treated_array = np.delete(treated_array, i_index, axis=1)


        X_array = treated_array
        Y_array = [y_vals[s]]*len(X_array)


        if "ACF" in nn_params or "timeseries"  in nn_params: 
    
            if split_timeseries == True:
                #frame by frame split
                X_array = X_array.transpose(0, 2, 3, 4, 1)

                frames = []
                for vid in X_array:
                    for frame in vid:
                        frames.append(frame)
                X_array = np.array(frames)
                X_array = X_array.transpose(0, 3, 2, 1)
                Y_array = [y_vals[s]]*len(X_array)


            if split_ACF == True: 
                #this is for the Conv1D
                #gets the ACF for every pixel individually
                X_array = X_array.transpose(0, 2, 3, 4, 1)

                ACF_curves = []
                for vid in range(len(X_array)):
                    for i in range(20):
                        for j in range(20):
                            ACF_curve = X_array[vid, :, i, j]
                            ACF_curves.append(ACF_curve)

                ACF_curves = np.array(ACF_curves)
                
                #tests to see whether ACF curves array is correct
                """
                xaxis = np.linspace(0, 100, 100)
                for i in range(len(ACF_curves)):
                    y = X_array[i]
                    plt.figure()
                    plt.scatter(xaxis, y)
                    plt.show()
                """

                #mean = np.mean(ACF_curves, axis=1, keepdims = True)
              
                #needs to be transposed this way for cnn input
                ACF_curves = ACF_curves.transpose(0, 2, 1)
                X_array = ACF_curves
                Y_array = [y_vals[s]]*len(X_array)  


        if flatten == True: 
            
            flattened = []
            for vid in X_array: 
                flattened.append(vid.flatten())

            X_array = np.array(flattened)
            #X_array = X_array.flatten()
            Y_array = [y_vals[s]]*len(X_array)
            X_array = np.expand_dims(X_array, 1)
            

        total_X_list.append(X_array)
        total_Y_list.extend(Y_array)

        if cross_val == False and antibiotic == "Gentamycin": 
            total_X_train_list.append(X_array[:int(len(X_array)*2/3)])
            total_X_test_list.append(X_array[-int(len(X_array)*1/3):])

            total_Y_train_list.append(Y_array[:int(len(Y_array)*2/3)])
            total_Y_test_list.append(Y_array[-int(len(Y_array)*1/3):])

    total_X_arr = np.concatenate(total_X_list)
    total_Y_arr = np.array(total_Y_list)

    if cross_val == False and antibiotic == "Gentamycin": 
        total_X_tr_arr = np.concatenate(total_X_train_list)
        total_X_ts_arr = np.concatenate(total_X_test_list)

        total_Y_tr_arr = np.concatenate(total_Y_train_list)
        total_Y_ts_arr = np.concatenate(total_Y_test_list)

    if save == "yes": 
        if "timeseries" in nn_params: 
            np.save("D:/train_test arrays/"+antibiotic +"/Xarr_" + antibiotic + "_" + str(nn_params) + "_split_" + str(split_timeseries), total_X_arr)
            np.save("D:/train_test arrays/"+antibiotic +"/Yarr_" + antibiotic + "_" + str(nn_params) + "_split_" + str(split_timeseries), total_Y_arr)
        else: 
            np.save("D:/train_test arrays/"+antibiotic +"/Xarr_" + antibiotic + "_" + str(nn_params) + "_300x4"  , total_X_arr)
            np.save("D:/train_test arrays/"+antibiotic +"/Yarr_" + antibiotic + "_" + str(nn_params) + "_300x4", total_Y_arr)

    if cross_val == False and antibiotic == "Gentamycin": 
        return total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr 
    
    else:
        return total_X_arr, total_Y_arr

def get_tensor_traintest(X_arr, Y_arr): 
    transform = ToTensor()
    X = torch.stack([transform(x) for x in X_arr])
    Y = torch.LongTensor(Y_arr)
    X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size=0.2, stratify=Y, shuffle=True)
    #trainset, testset = random_split(TensorDataset(X, Y), [0.8, 0.2])
    trainset = TensorDataset(X_tr, Y_tr)
    testset = TensorDataset(X_ts, Y_ts)

    return trainset, testset

class CustomDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample_x = self.X[idx]
        sample_y = self.Y[idx]

        # Apply transformation if provided
        if self.transform:
            sample_x = self.transform(sample_x)
        
        return sample_x, sample_y
#set up of CNN model structure
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()

        if ROI == "20x20": 
            self.layer1 = nn.Sequential(
                nn.Conv2d(nr_params, 32, kernel_size=3, padding="same"),
                #nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2))
            
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding="same"),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2))
            
            self.layer3= nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding="same"),
                #nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2))
            
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding="same"),
                #nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2))
            
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.ReLU())
        
            self.fc3= nn.Sequential(
                nn.Linear(128, nr_classes))
            
        elif ROI == "300x4":
            self.layer1 = nn.Sequential(
                nn.Conv2d(nr_params, 32, kernel_size=(2, 4), padding="same"),
                #nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2))
            
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=(2, 4), padding="same"),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2))
            
            self.layer3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(2, 4), padding="same"),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (1, 4)))
            
            self.layer4 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=(2, 4), padding="same"),
                #nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (1, 4)))
            
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                #nn.Linear(1024, 512),
                nn.Linear(4800, 2400),
                nn.ReLU())
        
            self.fc3= nn.Sequential(
                nn.Linear(2400, nr_classes))
                #nn.Linear(512, nr_classes))
        
        
    def forward(self, x):
        
        x = x.to(torch.float32)

        out = self.layer1(x)
        out = self.layer2(out)
    
        if ROI == "20x20":
            out = self.layer3(out)
            out = self.layer4(out)

        out = torch.flatten(out, start_dim = 1)
        out = self.fc(out)

        #if model is CNN_LSTM then we instead want the feature vector outputted by the previous step.
        if model not in ["CNN_LSTM"]: 
            out = self.fc3(out)

        return out

class LSTM_Model(nn.Module):
    def __init__(self):
        super(LSTM_Model, self).__init__()
    
        self.lstm = nn.LSTM(input_size=seq_length, hidden_size = 64, batch_first= True)
        self.sequential = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(64, nr_classes)
        

    def forward(self, x):

        x = x.to(torch.float32)
        x, _ = self.lstm(x)
        x = self.sequential(x)
        x = x[:, -1, :]
        x = self.fc(x)
      
        return x
    
class GRU_Model(nn.Module):
    def __init__(self):
        super(GRU_Model, self).__init__()
        self.rnn1 = nn.GRU(seq_length, 64, batch_first=True, dropout=0)
        self.rnn2 = nn.GRU(64, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 2)


    def forward(self, x):
        x = x.to(torch.float32)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        # Extract the last time step output
        x = x[:, -1, :]
        x = nn.functional.dropout(x, p=0.4)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        
        return x

class C3D_Model(nn.Module): 
    def __init__(self):

        super(C3D_Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv3d(nr_params, 32, kernel_size=3, padding=1),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            #nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.layer3 = nn.Sequential(
            #nn.Conv3d(128, 256, kernel_size=3, padding=1),
            #nn.BatchNorm3d(256),
            #nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            #nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.layer4 = nn.Sequential(
            #nn.Conv3d(256, 512, kernel_size=3, padding=1),
            #nn.BatchNorm3d(512),
            #nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            #nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))

        #last_duration = int(math.floor(sample_duration / 16))
        #last_size = int(math.ceil(sample_size / 32))
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            #nn.Linear(12800, 4096),
            nn.Linear(4608, 2304),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(2304, nr_classes))         

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, start_dim = 1)
        out = self.fc1(out)
        out = self.fc(out)
        return out

class C1D_Model(nn.Module):

    def __init__(self):

        super(C1D_Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=(3,), padding="same"),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=(1,), padding="same"),
            #nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = (2,)))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(3200, 1600),
            nn.ReLU())
        
        self.fc2 = nn.Sequential(
            nn.Linear(1600, nr_classes)) 
        
    def forward(self, x):
        x = x.to(torch.float32)
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, start_dim = 1)
        #out = self.fc(out)
        out = self.fc2(out)
        return out 
        
temporal_filter_size = 25
momentum = 0.9
in_channels = 32
out_channels = 32

class Shortcut(nn.Module):

    def __init__(self):
        super(Shortcut, self).__init__()
        
    def forward(self, this_input, residual):
        # Truncate the end section of the input tensor
        shortcut = this_input[:, :, :-(2 * temporal_filter_size -2)]

        return shortcut + residual
    
class fcsnet(nn.Module):
    def __init__(self):
        super(fcsnet, self).__init__()

        #input is layer
        self.layer1 = nn.Sequential(nn.Conv1d(1, out_channels, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())
        
        self.shortcut = Shortcut()



        self.layer3 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer4 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=temporal_filter_size), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())
        
        self.shortcut2 = Shortcut()


        #1x1 layers

        self.layer5 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer6 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())


        self.layer7 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer8 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())


        self.layer9 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer10 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())


        self.layer11 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer12 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())


        self.layer13 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer14 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())


        self.layer15 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())

        self.layer16 = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=(1,)), 
                    #nn.BatchNorm1d(out_channels, momentum=momentum), 
                    nn.ReLU())


        self.fc = nn.Sequential(
                    nn.Linear(128, 2))
                

    def forward(self, x):
        x = x.to(torch.float32)
        identity = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.shortcut(this_input = identity, residual = out)

        identity = out
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.shortcut2(this_input = identity, residual = out)

        identity = out
        out = self.layer5(out)
        out = self.layer6(out)
        out = identity + out

        identity = out
        out = self.layer7(out)
        out = self.layer8(out)
        out = identity + out

        identity = out
        out = self.layer9(out)
        out = self.layer10(out)
        out = identity + out

        identity = out
        out = self.layer11(out)
        out = self.layer12(out)
        out = identity + out

        identity = out
        out = self.layer13(out)
        out = self.layer14(out)
        out = identity + out

        identity = out
        out = self.layer15(out)
        out = self.layer16(out)
        out = identity + out

        out = torch.flatten(out, start_dim = 1)
        out = self.fc(out)
        
        return out

     
def get_summary(my_model, dataset, batch_size): 

    dataloader = DataLoader(dataset, shuffle = True, batch_size=batch_size)


    #look at shape of typical batches in data loaders
    for idx, (X_, Y_) in enumerate(dataloader):
        print("X: ", X_.shape)
        print("Y: ", Y_.shape)
        if idx >= 0:
            break

    #model architecture summary
    summary(my_model,
            input_data = X_,
            col_names=["input_size",
                        "output_size",
                        "num_params"])

def get_dataloaders(trainval_idx, test_idx, batch_size, dataset): 
    #splits the 4/5 of original data further into training and validation
    #such that in total, 3/5 of folds are used for training, 1/5 for validation and 1/5 for testing
    #(and splits are different for each fold)

    trainval_dataset = CustomDataset(dataset[trainval_idx][0], dataset[trainval_idx][1])
    

    kfold2 = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    train_idx, val_idx = next(kfold2.split(trainval_dataset, trainval_dataset.Y))

    train_idx = trainval_idx[train_idx]
    val_idx = trainval_idx[val_idx]

        
    trainloader  = DataLoader(dataset, batch_size=batch_size, sampler = SubsetRandomSampler(train_idx))
    validloader  = DataLoader(dataset, batch_size=batch_size, sampler = SubsetRandomSampler(val_idx))
    testloader  = DataLoader(dataset,  batch_size=batch_size, sampler = SubsetRandomSampler(test_idx))

    return trainloader, validloader, testloader

def get_non_sampling_loaders(X_trainval, Y_trainval, X_test, Y_test, batch_size):
    #NOTE this train/val/test splitting is DIFFERENT than the fold method: 
    #Here an extra measure of generalizability: Data is split into train/val set and testset NOT by random splitting. (shuffling does occur though after the splitting) 
    #This splitting only applicable to KANAMYCIN and TRIMETHOPRIM given that they have three repeats and so we know that the last repeat has an equal nr of classe


    testset = CustomDataset(X_test, Y_test)
    
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2, stratify=Y_trainval)
    trainset = CustomDataset(X_train, Y_train)
 
    validset = CustomDataset(X_val, Y_val)
  

    trainloader = DataLoader(trainset, shuffle = True, batch_size=batch_size)
    validloader = DataLoader(validset, shuffle = True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle = True, batch_size=batch_size)

    return trainloader, validloader, testloader

def train_val(_model,trainloader, validloader, max_epochs, loss_fn, my_optimizer, early_stop_thresh, directory1):  

    best_accuracy = -1
    best_epoch = -1

    trainingEpoch_loss = []
    trainingEpoch_acc = []
    valEpoch_loss = []
    valEpoch_acc = []

    for epoch in range(max_epochs):
        
        
        #####################################################
        #training
        #####################################################
        count = 0
        step_loss = []
        step_acc = []

        _model.train()
        for X_batch, Y_batch in trainloader: 
            my_optimizer.zero_grad()
            Y_pred = _model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)
            loss.backward()
            my_optimizer.step()

            fY_pred = torch.argmax(Y_pred, 1)
            
            acc = (fY_pred == Y_batch).float().sum()
            count += len(Y_batch)

            step_loss.append(loss.item())
            step_acc.append(acc)
            
        epoch_loss = np.array(step_loss).mean()
        epoch_acc = (np.array(step_acc).sum())/count

        trainingEpoch_acc.append(epoch_acc)
        trainingEpoch_loss.append(epoch_loss)

    
        
        #####################################################
        #validation
        #####################################################
        val_count = 0
        val_step_loss = []
        val_step_acc = []

        _model.eval()
        for X_batch, Y_batch in validloader: 
            with torch.no_grad(): 
                Y_pred = _model(X_batch)
                val_loss = loss_fn(Y_pred, Y_batch)
                
                fY_pred = torch.argmax(Y_pred, 1)

                val_acc = (fY_pred == Y_batch).float().sum()
                val_count += len(Y_batch)
                val_step_acc.append(val_acc)
                val_step_loss.append(val_loss.item())


        val_epoch_loss = np.array(val_step_loss).mean()
        val_epoch_acc = (np.array(val_step_acc).sum())/val_count

        valEpoch_loss.append(val_epoch_loss)
        valEpoch_acc.append(val_epoch_acc)

        print("Epoch %d: validation loss %.2f" % (epoch, val_epoch_loss))
        print("Epoch %d: validation accuracy %.2f%%" % (epoch, val_epoch_acc*100))

    
    
        if val_epoch_acc > best_accuracy:
            best_accuracy = val_epoch_acc
            best_epoch = epoch

            #temporarily stores the best model's state dict 
            torch.save(_model.state_dict(), directory1)

        #Early stopping
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            break  # terminate the training loop

    return trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc

def test(_model, directory1, testloader, loss_fn, mean_fpr, batchsize):
        
    #loads in state dict belonging to the best model
    #this is also the place to load in state dict of best model that was trained on a DIFFERENT antibiotic, for instance, to test generalizability.
    #_model.load_state_dict(torch.load(directory1))
    #dir = "D:/CNN_res/Kanamycin/95.03%_20x20_double_['_g0_']/best_model.pth"
    #dir = "D:/CNN_res/Trimethoprim/65.25%_20x20_double_['_g0_']/best_model.pth"
    dir = "D:/CNN_res/Trimethoprim/66.40%_20x20_2DTIRF_['_g0_']/best_model.pth"
    
    #dir = directory1
    _model.load_state_dict(torch.load(dir))

    test_count = 0
    test_step_loss = []
    test_step_acc = []

    y_pred_list = []
    y_true_list = []
    y_pos_prob = []

    _model.eval()
    for X, Y in testloader:
        with torch.no_grad(): 
            Y_pred = _model(X)
            test_loss = loss_fn(Y_pred, Y)
            fY_pred = torch.argmax(Y_pred, 1)
            
            test_acc = (fY_pred == Y).float().sum()
            test_count += len(Y)
            test_step_acc.append(test_acc)
            test_step_loss.append(test_loss.item())

        
            pos_prob = Y_pred[:, 1]

            
            y_pred_list.extend(fY_pred)
            y_true_list.extend(Y)
            y_pos_prob.extend(pos_prob)
            
        
    
    test_loss = np.array(test_step_loss).mean()
    test_acc = (np.array(test_step_acc).sum())/test_count

    print("Trained test accuracy: %.2f%%" % ((test_acc.item())*100))
    print("Trained test loss: %.2f" % ((test_loss.item())))

    cm = confusion_matrix(y_true_list, y_pred_list)
    #tp = cm[0][0]
    #fn = cm[0][1]
    #fp = cm[1][0]
    #tn = cm[1][1]

    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]

    #sensitivity (true positive rate) and specificity (true negative rate):  take positive = class 1, negative = class 0
    #f1?
    sens = tp/(tp+fn)
    spec = tn/(fp+tn)
    g_mean = np.sqrt(sens*spec)


    precision = tp/(tp+fp)
    recall = sens
    f1 = 2*((precision*recall)/(precision+recall))

    
    fpr, tpr, _ = roc_curve(y_true_list, y_pos_prob)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0

    roc_auc = roc_auc_score(y_true_list, y_pos_prob)

    if model not in ["CNN_LSTM", "LSTM"] and len(nn_params) == 1:

        #SHAP analysis
        batch = next(iter(testloader))
        images, labels = batch

        batchsize = 16
        background = images[:(batchsize-2)]
        test_images = images[(batchsize -2):batchsize]

        test_labels = labels[(batchsize -2):batchsize]
        

        e = shap.DeepExplainer(_model, background)
        shap_values = e.shap_values(test_images )
        shap_values = np.swapaxes(shap_values, 1, -1)
        shap_values = np.swapaxes(shap_values, 0, 1)

        shap_numpy = [s for s in shap_values]

        test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

        p = shap_numpy, test_numpy, test_labels
        
    else: 
        #placeholder as no SHAP (yet) for LSTMs
        #SHAP also only works when one parameter is used as input
        p = 0

    #plt.tight_layout()
    #plt.show()

    return test_acc, sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, p
    

def plot_epochs(directory, trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc): 

    fig, axes= plt.subplots(2, 1, figsize=(10, 8), sharex = False)
    axes[0].plot(trainingEpoch_loss, label='train_loss')
    axes[0].plot(valEpoch_loss,label='val_loss')
    axes[0].legend()

    axes[1].plot(trainingEpoch_acc, label='train_accuracy')
    axes[1].plot(valEpoch_acc,label='val_accuracy')
    axes[1].legend()
    
    plt.suptitle("Accuracy and Loss over epochs (last fold)")
    plt.savefig(directory, dpi = 300)

def shap_plot(shapplot):

    if model not in ["CNN_LSTM", "LSTM"]:


    
        #SHAP plotting
        shapnp, testnp, testlabels = shapplot
        t = shapnp[0]

        """
        #this is for 300x4 plotting - need to save manually

        threshold_low = -0.006
        threshold_high = 0.006

        blue = (0, 0, 1)     
        white = (1, 1, 1) 
        red = (1, 0, 0)      

    
        n_bins = 100  
        cmap_name = 'blue_gray_red'
        cmap = colors.ListedColormap([ blue, white, red])
       
        bounds = [-1, threshold_low, threshold_high, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        fig = plt.figure(figsize=(4, 4), dpi=300)
        gs = fig.add_gridspec(4, 1)


        fig.add_subplot(gs[0, 0])
        plt.imshow(testnp[0], cmap = "Greys_r")
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

    
        fig.add_subplot(gs[1, 0])
        plt.imshow(shapnp[0][0],cmap=cmap, norm=norm, interpolation='nearest')
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

      
        fig.add_subplot(gs[2, 0])
        plt.imshow(testnp[1], cmap = "Greys_r")
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

      
        fig.add_subplot(gs[3, 0])
        plt.imshow(shapnp[0][1], cmap=cmap, norm=norm, interpolation='nearest')
        plt.tick_params(
                    bottom=False,
                    left = False,       
                    labelbottom=False, 
                    labelleft = False) 

        #cbar = plt.colorbar(orientation='vertical')
        #cbar.ax.tick_params(labelsize=5)
          

        oglabels = ["C" + str(testlabels[0].item()), "C" + str(testlabels[1].item())]
        ogx_position = 0.08
        ogy_positions = [0.78, 0.4]  

        for y_pos, label in zip(ogy_positions, oglabels):
            plt.text( ogx_position, y_pos, label, transform=plt.gcf().transFigure, va='center', fontsize=5)



        plt.subplots_adjust(hspace=0.001) 
        plt.tight_layout()
       
        plt.show()
        
        """
       
    
        #This is for when have 2 TEST IMAGES
        #fig = plt.figure()
        shap.image_plot(shapnp, testnp, show = False)
        labels = ["Original Im" + str(nn_params), "Class 0 SHAP", "Class 1 SHAP"]
        x_positions = [0.235, 0.49, 0.76]  
        y_position = 0.93

        oglabels = ["Class " + str(testlabels[0].item()), "Class " + str(testlabels[1].item())]
        ogx_position = 0.15
        ogy_positions = [0.9, 0.6]  

        for y_pos, label in zip(ogy_positions, oglabels):
            plt.text( ogx_position, y_pos, label, transform=plt.gcf().transFigure, va='center', fontsize=12)

        for x_pos, label in zip(x_positions, labels):
            plt.text(x_pos, y_position, label, transform=plt.gcf().transFigure, ha='center', fontsize=12)

        
        return plt.gcf()

def shap_flat_plot(shaplist, directory2): 

        #have shap values for class 0 and for class 1
        #since binary, all info is contained in one. eg shap0 positive contribute to class 0, negative contribute to class 1

    shap0 = []
    testims = []
    for el in shaplist: 
        shapnp, testnp, testlabels = el
        shapclass0 = shapnp[0].flatten()
        shap0.extend(shapclass0)
        testims.extend(testnp.flatten())

    np.save(directory2 + "flat shap vs param data", np.array([shap0, testims]))

    plt.figure(figsize=(10, 6), dpi=300)
    plt.scatter(shap0, testims, c = "b", alpha = 0.1)
    plt.xlabel("SHAP value")
    plt.ylabel(nn_params)
    plt.grid()
    plt.savefig(directory2 + "paramvsshap",dpi=300)
    #plt.show()

def fold_res(k_folds, fold_metrics, tprs, y_true_folds, y_pred_folds,  mean_fpr, trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc, shap_folds_list, shapfigs, directory1, directory3):

    # Print fold results
    print('----------------------------------------------------')
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('----------------------------------------------------')
  
    fold_test_accs = []
    fold_aucs = []
    fold_specs = []
    fold_sens = []
    fold_gmeans = []
    fold_f1s = []

    for m in fold_metrics: 

        fold_test_accs.append(fold_metrics[m]["test accuracy"])
        fold_aucs.append(fold_metrics[m]["auc"])
        fold_specs.append(fold_metrics[m]["spec"])
        fold_sens.append(fold_metrics[m]["sens"])
        fold_gmeans.append(fold_metrics[m]["gmean"])
        fold_f1s.append(fold_metrics[m]["f1"])

        test_acc = fold_metrics[m]["test accuracy"]
        print("Fold %d: %.4f" % (m, test_acc))

    mean_test_acc = np.mean(fold_test_accs)
    sd_test_acc = np.std(fold_test_accs)

    mean_spec = np.mean(fold_specs)
    sd_spec = np.std(fold_specs)

    mean_sens = np.mean(fold_sens)
    sd_sens = np.std(fold_sens)

    mean_gmean = np.mean(fold_gmeans)
    sd_gmean = np.std(fold_gmeans)

    mean_auc = np.mean(fold_aucs)
    sd_auc = np.std(fold_aucs)

    mean_tpr = np.mean(tprs, axis=0)
    sd_tpr = np.std(tprs, axis = 0)
    
    mean_f1 = np.mean(fold_f1s)
    sd_f1 = np.std(fold_f1s)
    
    print("Mean test accuracy: %.4f \u00B1 %.4f" % (mean_test_acc, sd_test_acc))
    print("Mean specificity: %.4f \u00B1 %.4f" % (mean_spec, sd_spec))
    print("Mean sensitivity: %.4f \u00B1 %.4f" % (mean_sens, sd_sens))
    print("Mean G-mean: %.4f \u00B1 %.4f" % (mean_gmean, sd_gmean))
    print("Mean AUC: %.4f \u00B1 %.4f" %  (mean_auc, sd_auc))
    print("Mean F1: %.4f \u00B1 %.4f" % (mean_f1, sd_f1))


    directory2 = "D:/" + model + "_res/" + antibiotic + "/" + "%.2f%%" % ((mean_test_acc*100)) + "_" + ROI + "_" + fit  + "_" + str(nn_params) + "/"
    if os.path.exists(directory2) == False: 
        os.makedirs(directory2)

    #plots training and validaton acc and loss over epochs, for the last fold
    #(doesn't make sense to plot mean curves because different epochs for each fold due to early stopping)
    plot_epochs(directory2 + "metrics_plot", trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc)

    #get confusion matrix of all folds
    cm = confusion_matrix(y_true_folds, y_pred_folds)
    cmfig = ConfusionMatrixDisplay(cm)
    cmfig.plot()
    plt.title("Confusion matrix of 5 folds")
    cmfig.figure_.savefig(directory2 + "conf_mx.png",dpi=300)

    #plot Mean ROC curve with standard deviations
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=("Mean ROC (AUC %.2f $\pm$ %.2f)" %  (mean_auc, sd_auc)),
        lw=2,
        alpha=0.8,
    )

    ax.plot([0, 1], ls="--")

    tprs_upper = np.minimum(mean_tpr + sd_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - sd_tpr, 0)

    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title="Mean ROC curve",
    )
    ax.legend(loc="lower right")
    plt.savefig(directory2 + "roc.png",dpi=300)
    
    rc = np.array([mean_tpr, tprs_upper, tprs_lower])
    a = np.array([mean_auc, sd_auc])
    np.save( directory2 + "mean roc curve data.npy", rc)
    np.save( directory2 + "mean auc data.npy", a)
                 

    report = classification_report(y_true_folds, y_pred_folds )
    print(report)

    if os.path.exists(directory3) == True:
        shutil.move(directory3, directory2 + "featuremaps")

    shutil.move(directory1, directory2 + "best_model.pth")
    

    fold_all_res = {"fold_test_accs": fold_test_accs, "fold_aucs": fold_aucs, "fold_specs": fold_specs, "fold_sens": fold_sens, "fold_g_means": fold_gmeans, "fold_f1s": fold_f1s}
    fold_mean_res = {"Mean test accuracy": "%.4f \u00B1 %.4f" % (mean_test_acc, sd_test_acc), "Mean specificity": "%.4f\u00B1 %.4f" % (mean_spec, sd_spec),
                     "Mean sensitivity": "%.4f \u00B1 %.4f" % (mean_sens, sd_sens), "Mean G-mean": "%.4f \u00B1 %.4f" % (mean_gmean, sd_gmean),
                     "Mean AUC": "%.4f \u00B1 %.4f" %  (mean_auc, sd_auc), "Mean F1": "%.4f \u00B1 %.4f" % (mean_f1, sd_f1)}
 
    with open(directory2 +  "result_data.txt", "wt") as fp:
        fp.write("Input params: " + str(nn_params) + "\n" + str(fold_all_res) + "\n" + str(fold_mean_res))

    if model not in ["CNN_LSTM", "LSTM"] and (method not in ["acf-1DNN"]) and len(nn_params) == 1:
        i = 0
        for fig in shapfigs: 
            fig.savefig(directory2 + "shap" + str(i))
            i += 1

        shap_flat_plot(shap_folds_list, directory2)

       
def plot_featuremaps(directory1, directory3, testloader ):
                        
    if os.path.exists(directory3) == False: 
            os.makedirs(directory3)

    if model == "3DCNN": 
        newmodel = C3D_Model()

    elif model in ["CNN_LSTM","LSTM"]:
        #my_model = GRU_Model()
        newmodel = LSTM_Model()
        #batch_size = 16
        batch_size = 64

    elif model == "1DCNN": 
        #my_model = C1D_Model()
        newmodel = fcsnet()
        batch_size = 5000

    else:
        newmodel = CNN_Model()
        #batch_size = 32
        batch_size = 32

    #loads in state dict belonging to the best model
    newmodel.load_state_dict(torch.load(directory1))
    newmodel.eval()

    for X, Y in testloader: 
        X = X.to(torch.float32)
        for j in range(len(X)): 
            x = X[j]
            y = Y[j]
            

            if "ACF" in nn_params: 
                pass

            else: 
                plt.figure()
                plt.imshow(x[0])
                plt.savefig(directory3  + "/" + str(j) + "_im_class" + str(y.item()), dpi = 300) 
                with torch.no_grad(): 
                    feature_maps = newmodel.layer1(x)
                #fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))

                if ROI == "300x4":
                    fig = plt.figure(figsize=(4, 32))
                    gs = fig.add_gridspec(32, 1)
                    for i in range(32):
                        ax = fig.add_subplot(gs[i, 0])
                        ax.imshow(feature_maps[i])
                        ax.axis("off")

                    plt.tight_layout()  
                    plt.savefig(directory3  + "/" + str(j) + "_1stConvlayer_class" + str(y.item()))
                    plt.show()

                else:
                    fig, ax = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
                    plt.axis("off")
                    for i in range(0, 32):
                        row, col = i//8, i%8
                        ax[row][col].imshow(feature_maps[i])
                    plt.savefig(directory3  + "/" + str(j) + "_1stConvlayer_class" + str(y.item()) , dpi = 300)

            
            """
            with torch.no_grad(): 
                feature_maps = newmodel.layer2(feature_maps)

            fig2, ax2 = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
            for j in range(0, 32):
                row, col = j//8, j%8
                ax2[row][col].imshow(feature_maps[j])
            plt.savefig(directory3 + "/3rdConvlayer_" + str(i), dpi = 300)

            with torch.no_grad(): 
                feature_maps = newmodel.layer3(feature_maps)

            fig2, ax2 = plt.subplots(4, 8, sharex=True, sharey=True, figsize=(16,8))
            for j in range(0, 32):
                row, col = j//8, j%8
                ax2[row][col].imshow(feature_maps[j])
            plt.savefig(directory3 + "/3rdConvlayer_" + str(i), dpi = 300)
        """   
        break

def run_CNN_LSTM_models(cross_val = True, get_featuremaps = False): 

    #try 3D CNN
    #(testing their generalizability is most important part)

    ########################## Feature extraction for CNN_LSTM ##########################
    

    if model == "CNN_LSTM": 

        print("Running feature extraction")
        feature_extractor = CNN_Model()
        if "timeseries" in nn_params: 
            #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Trimethoprim/47.55%_20x20_none_['timeseries']REP/best_model.pth"))
            #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Kanamycin/76.40%_20x20_none_['timeseries']REP/best_model.pth"))
            #feature_extractor.load_state_dict(torch.load( "D:/CNN_res/Gentamycin/66.73%_20x20_none_['timeseries']/best_model.pth"))
           

            feature_extractor.load_state_dict(torch.load("D:/CNN_res/Gentamycin/97.41%_20x20_none_['timeseries']/best_model.pth"))
            #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Kanamycin/99.97%_20x20_none_['timeseries']/best_model.pth"))
            #feature_extractor.load_state_dict(torch.load("D:/CNN_res/Trimethoprim/96.99%_20x20_none_['timeseries']/best_model.pth"))
            
        if cross_val == False and antibiotic == "Gentamycin": 
            global total_X_tr_arr
            global total_X_ts_arr
            total_X_array = np.concatenate((total_X_tr_arr, total_X_ts_arr), axis=0)

        else: 
            total_X_array = total_X_arr 

        frame_features = np.zeros(shape=(len(total_X_array), vid_seq_length, 128), dtype="float32")

        for vid in range(len(total_X_array)): 
            temp_frame_features = np.zeros(shape=( vid_seq_length, 1, 128), dtype="float32")

            
            frames = total_X_array[vid]
            frames = frames.transpose(1, 0, 2, 3)
            frames = frames[None, ...]
            frames = torch.tensor(frames)
            
            
            for i, batch in enumerate(frames):
                for frame in range(vid_seq_length): 
                    #transform = ToTensor()
                    #input = transform(batch[None, frame, :])
                    input = batch[None, frame, :]
                    res = feature_extractor(input)
                    res = res.detach().numpy()
                    temp_frame_features[frame, :] = res

            frame_features[vid,] = temp_frame_features.squeeze()

        new_total_X_arr = frame_features
        np.save("D:/train_test arrays/"+antibiotic +"/Xfeature_arr_" + antibiotic + "_" + str(nn_params), new_total_X_arr)


    ################################# DNN #################################

    #new_total_X_arr = np.load("D:/train_test arrays/"+antibiotic +"/Xfeature_arr_" + antibiotic + "_" + str(nn_params) + ".npy")
        if cross_val == False and antibiotic == "Gentamycin": 
            total_X_tr_arr = new_total_X_arr[:int(len(new_total_X_arr)*2/3)]
            total_X_ts_arr = new_total_X_arr[-int(len(new_total_X_arr)*1/3):]
        
        else: 
            X_arr = new_total_X_arr
    
    else: 
        if cross_val == False and antibiotic == "Gentamycin": 
            pass

        else: 
            X_arr = total_X_arr

    #hyperparameters

    if model in ["CNN_LSTM","LSTM"]:
        print("Running LSTM")
        A_lr=0.005
        early_stop_thresh = 10
        max_epochs = 100

    else: 

        print("Running CNN")
        A_lr = 0.0005
        early_stop_thresh = 30
        max_epochs = 30


    #SGD_lr =  0.0038677245
    optimizer_options = ["Adam", "SGD"]
    optimizer = optimizer_options[0]
    #weight_decay = 6.012330734402105e-05
    #momentum = 0.80728432696454


    loss_fn = nn.CrossEntropyLoss()
   

    if cross_val == False and antibiotic == "Gentamycin":
        total_Y_tr = torch.LongTensor(total_Y_tr_arr)
        total_Y_ts = torch.LongTensor(total_Y_ts_arr)
        dataset = CustomDataset(total_X_tr_arr, total_Y_tr)



    else:
        total_Y = torch.LongTensor(total_Y_arr)
        dataset = CustomDataset(X_arr, total_Y)

    
    """
    xaxis = np.linspace(0, 100, 100)
    y = trainset[0][0]
    plt.figure()
    plt.scatter(xaxis, y)
    plt.show()
    """          

    k_folds = 5


    #fold metrics
    fold_metrics = {}
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    shapfigs = []
    shap_folds_list = []

    y_true_folds = []
    y_pred_folds = []

    if cross_val == True:

        #Define the K-fold Cross Validator - stratified so same percentage of classes in each split
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)


        for fold, (trainval_idx, test_idx) in enumerate(kfold.split(dataset, dataset.Y)):
                

            #Initialise model for every fold (otherwise will have learned)
            #if model == ()
            
            if model == "3DCNN": 
                my_model = C3D_Model()

            elif model in ["CNN_LSTM","LSTM"]:
                #my_model = GRU_Model()
                my_model = LSTM_Model()
                #batch_size = 16
                batch_size = 64

            elif model == "1DCNN": 
                #my_model = C1D_Model()
                my_model = fcsnet()
                batch_size = 5000

            else:
                my_model = CNN_Model()
                #batch_size = 32
                batch_size = 32
            


            if optimizer == "Adam":
                my_optimizer = torch.optim.Adam(my_model.parameters(), lr=A_lr)

            elif optimizer == "SGD":
                my_optimizer = torch.optim.SGD(my_model.parameters(), lr=SGD_lr, weight_decay = weight_decay, momentum = momentum)  
            
            
            trainloader, validloader, testloader = get_dataloaders(trainval_idx, test_idx, batch_size, dataset)

            directory1 = "D:/" + model + "_res/" + antibiotic +  "/best_model.pth"

            #get model architecture summary
            if fold == 0:
                get_summary(my_model, dataset, batch_size)


            print(f"Fold {fold + 1}")
            print("-------")

            #####################################################
            #train
            #####################################################
            trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc = train_val(my_model, trainloader, validloader, max_epochs, loss_fn, my_optimizer, early_stop_thresh, directory1)
            

            #####################################################
            #test on untrained model
            #####################################################

            if model == "3DCNN": 
                untrained_model = C3D_Model()

            elif model in ["CNN_LSTM","LSTM"]:
                #my_model = GRU_Model()
                untrained_model = LSTM_Model()
                #batch_size = 16
                batch_size = 64

            elif model == "1DCNN": 
                #my_model = C1D_Model()
                untrained_model = fcsnet()
                batch_size = 5000

            else:
                untrained_model = CNN_Model()
                #batch_size = 32
                batch_size = 32
            
        
            
            for X, Y in testloader:
                y_pred = untrained_model(X)
                acc = (torch.argmax(y_pred, 1) == Y).float().mean()

            print("Untrained test accuracy: %.2f%%" % ((acc.item())*100))

            
            #####################################################
            #test on trained model
            #####################################################
            
            if model == "3DCNN": 
                trained_model = C3D_Model()

            elif model in ["CNN_LSTM","LSTM"]:
                #my_model = GRU_Model()
                trained_model = LSTM_Model()
                #batch_size = 16
                batch_size = 64

            elif model == "1DCNN": 
                #my_model = C1D_Model()
                trained_model = fcsnet()
                batch_size = 5000

            else:
                trained_model = CNN_Model()
                #batch_size = 32
                batch_size = 32

            #testing
            test_acc, sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, shapplot = test(trained_model, directory1, testloader, loss_fn, mean_fpr, batch_size)
        
            y_pred_folds.extend(y_pred_list)
            y_true_folds.extend(y_true_list)

            tprs.append(interp_tpr)
            fold_metrics[fold] = {"test accuracy": test_acc.item(), "sens": sens, "spec": spec.item(), "gmean": g_mean, "auc": roc_auc, "f1": f1}

            #placeholders as no featuremaps and no shap (yet) for LSTMs
            directory3 = "D:/" + model + "_res/" + antibiotic + "/featuremaps"
            

            if len(nn_params) == 1 and model == "CNN":
                fig = shap_plot(shapplot)
                shapfigs.append(fig)
                shap_folds_list.append(shapplot)

            else: 
                #placeholder as can't do SHAP with more than one parameter 
                shap_folds_list = []
                shapfigs = 0

            

        if get_featuremaps == True:

                #only gets featuremaps for first fold
                if run == 0:
                    plot_featuremaps()
        
        fold_res(k_folds, fold_metrics, tprs, y_true_folds, y_pred_folds,  mean_fpr, trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc,  shap_folds_list, shapfigs, directory1, directory3)
    
    else: 

        for run in range(k_folds):

            if model == "3DCNN": 
                my_model = C3D_Model()

            elif model in ["CNN_LSTM","LSTM"]:
                #my_model = GRU_Model()
                my_model = LSTM_Model()
                #batch_size = 16
                batch_size = 64

            elif model == "1DCNN": 
                #my_model = C1D_Model()
                my_model = fcsnet()
                batch_size = 5000

            else:
                my_model = CNN_Model()
                #batch_size = 32
                batch_size = 32   
                   


            if optimizer == "Adam":
                my_optimizer = torch.optim.Adam(my_model.parameters(), lr=A_lr)

            elif optimizer == "SGD":
                my_optimizer = torch.optim.SGD(my_model.parameters(), lr=SGD_lr, weight_decay = weight_decay, momentum = momentum)  
            

            if antibiotic == "Gentamycin": 
                X_trainval = total_X_tr_arr
                Y_trainval = total_Y_tr

                X_test = total_X_ts_arr
                Y_test = total_Y_ts
            else: 
                X_trainval = X_arr[:int(len(total_Y)*2/3)]
                Y_trainval = total_Y[:int(len(total_Y)*2/3)]

                X_test = X_arr[int(len(total_Y)*2/3): int(len(total_Y))]
                Y_test = total_Y[int(len(total_Y)*2/3):int(len(total_Y))]

            trainloader, validloader, testloader = get_non_sampling_loaders(X_trainval, Y_trainval, X_test, Y_test, batch_size)

            directory1 = "D:/" + model + "_res/" + antibiotic +  "/best_model.pth"
            
            #get model architecture summary
            if run == 0:
                    get_summary(my_model, dataset, batch_size)


            print(f"Run {run + 1}")
            print("-------")

            #training
            trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc = train_val(my_model, trainloader, validloader, max_epochs, loss_fn, my_optimizer, early_stop_thresh, directory1)
            
            
            #####################################################
            #test on untrained model
            #####################################################
            if model == "3DCNN": 
                untrained_model = C3D_Model()

            elif model in ["CNN_LSTM","LSTM"]:
                #my_model = GRU_Model()
                untrained_model = LSTM_Model()
                #batch_size = 16
                batch_size = 64

            elif model == "1DCNN": 
                #my_model = C1D_Model()
                untrained_model = fcsnet()
                batch_size = 5000

            else:
                untrained_model = CNN_Model()
                #batch_size = 32
                batch_size = 32
            
            """
            for X, Y in testloader:
                y_pred = untrained_model(X)
                
                if model in ["CNN_LSTM", "LSTM", "3DCNN"]:
                    acc = (torch.round(y_pred) == Y).float().mean()

                else: 
                    acc = (torch.argmax(y_pred, 1) == Y).float().mean()

            print("Untrained test accuracy: %.2f%%" % ((acc.item())*100))
            """

            #####################################################
            #test on trained model
            #####################################################
            
            if model == "3DCNN": 
                trained_model = C3D_Model()

            elif model in ["CNN_LSTM","LSTM"]:
                #my_model = GRU_Model()
                trained_model = LSTM_Model()
                #batch_size = 16
                batch_size = 64

            elif model == "1DCNN": 
                #my_model = C1D_Model()
                trained_model = fcsnet()
                batch_size = 5000

            else:
                trained_model = CNN_Model()
                #batch_size = 32
                batch_size = 32

            #testing
            test_acc, sens, spec, g_mean, f1, roc_auc, interp_tpr, y_pred_list, y_true_list, shapplot = test(trained_model, directory1, testloader, loss_fn, mean_fpr, batch_size)
        
            y_pred_folds.extend(y_pred_list)
            y_true_folds.extend(y_true_list)

            tprs.append(interp_tpr)
            fold_metrics[run] = {"test accuracy": test_acc.item(), "sens": sens, "spec": spec.item(), "gmean": g_mean, "auc": roc_auc, "f1": f1}

            #####################################################
            # Results for all runs
            #####################################################

            #placeholders as no featuremaps and no shap (yet) for LSTMs
            directory3 = "D:/" + model + "_res/" + antibiotic + "/featuremaps"
            

            if len(nn_params) == 1 and model == "CNN":
                fig = shap_plot(shapplot)
                shapfigs.append(fig)
                shap_folds_list.append(shapplot)

            else: 
                #placeholder as can't do SHAP with more than one parameter 
                shap_folds_list = []
                shapfigs = 0

        if get_featuremaps == True:

                #only gets featuremaps for first fold
                if run == 0:
                    plot_featuremaps()
        

        fold_res(k_folds, fold_metrics, tprs, y_true_folds, y_pred_folds,  mean_fpr, trainingEpoch_loss, trainingEpoch_acc, valEpoch_loss, valEpoch_acc,  shap_folds_list, shapfigs, directory1, directory3)


def print_input_summary(nn_params, ROI, fit): 

    print("NN classes: 0-susceptible and 1-resistant")
    print("Antibiotic(s) considered: ", antibiotic)
    print("Method chosen: ", method)
    print("Params to be processed and inputted into NN: ", nn_params)
    print("ROI size:", ROI)
    print("Fit used: ", fit)

def input_params(fit):
    print("Note: g0 and intensities don't vary across fits, the options are just given for each anyway. ")
    while True: 
        if fit == "single": 
            nn_params = input('Choose one or more of the following, separated by a comma w/0 space: "_m_mono_", "t_mono", "R", "_g0_", "intensities"').split(",")

        elif fit == "double":
            nn_params = input('Choose one or more of the following, spearated by a comma w/o space: "_m1_", "_m2_", "t1", "t2", "_g0_", "intensities"').split(",")

        elif fit in ["2DTIRF", "3DTIRF", "2Dcutsigma"]:
            nn_params = input('Choose one or more of the following, separated by a comma w/o space:  "N", "kappa", "tD", "_g0_", "intensities"').split(",")

        else: 
            if method == "pf-LSTM": 

                if fit == "single": 
                    nn_params = input('Choose one of the following: "_m_mono_", "t_mono", "R", "_g0_", "intensities"').split(",")

                elif fit == "double":
                    nn_params = input('Choose one of the following: "_m1_", "_m2_", "t1", "t_2", "_g0_", "intensities"').split(",")

                elif fit in ["2DTIRF", "3DTIRF", "2Dcutsigma"]:
                    nn_params = input('Choose one of the following: "N", "kappa", "tD", "_g0_", "intensities"').split(",")


            elif method in ["acf-1DCNN", "acf-LSTM"]:
                nn_params = ["ACF"]
            
            elif method in ["v-CNN", "i-CNNLSTM","i-3DCNN"]:
                nn_params = ["timeseries"]
        
        for el in nn_params: 
            if el not in ["_m_mono_", "t_mono", "_m1_", "_m_2_", "t1", "t_2", "N", "kappa", "tD", "_g0_", "R", "timeseries", "ACF"]:
                print("Choose a valid input parameter!")
        
        else: 
            break

    return nn_params

def filenames(ROI, fit):
    """
    #Names of "folders" to be inputted in pre-processing function. 
    #-----------------------------------------------------------------------
    #Here: 01-10: Gentamycin, Kanamycin, S3-S5: Trimethoprim

    #Kanamycin and Trimethoprim actual folder names are without the "_r" and "_s" - those were just put there since
    #their folders contains both resistant and susceptible, and function will need to distinguish between them

    #different fit choices have different folder names. Different fits were only saved for 20x20 ROIs 
    """

    if ROI == "20x20":
        if fit in ["single", "double", "3DTIRF", "none"]:
            res_sus_datanames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "fig4 kanamycin1_r", "fig4 kanamycin1_s", "fig4 kanamycin2_r","fig4 kanamycin2_s",  "fig4 kanamycin3_r", "fig4 kanamycin3_s", "S3_r", "S3_s", "S4_r", "S4_s", "S5_r", "S5_s"]
        
        elif fit == "2DTIRF":
            res_sus_datanames = ["01_2d_Ginf_Ne20", "02_2d_Ginf_Ne20", "03_2d_Ginf_Ne20", "04_2d_Ginf_Ne20", "05_2d_Ginf_Ne20", "06_2d_Ginf_Ne20", "07_2d_Ginf_Ne20", "08_2d_Ginf_Ne20", "09_2d_Ginf_Ne20", "10_2d_Ginf_Ne20", "fig4 kanamycin1_2d_Ginf_Ne20_r", "fig4 kanamycin1_2d_Ginf_Ne20_s", "fig4 kanamycin2_2d_Ginf_Ne20_r","fig4 kanamycin2_2d_Ginf_Ne20_s",  "fig4 kanamycin3_2d_Ginf_Ne20_r", "fig4 kanamycin3_2d_Ginf_Ne20_s", "S3_2d_Ginf_Ne20_r", "S3_2d_Ginf_Ne20_s", "S4_2d_Ginf_Ne20_r", "S4_2d_Ginf_Ne20_s", "S5_2d_Ginf_Ne20_r", "S5_2d_Ginf_Ne20_s"]

        elif fit == "2Dcutsigma":
            res_sus_datanames = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "fig4 kanamycin1_2d_Ginf_0cut0sigma_Ne10_r", "fig4 kanamycin1_2d_Ginf_0cut0sigma_Ne10_s", "fig4 kanamycin2_2d_Ginf_0cut0sigma_Ne10_r","fig4 kanamycin2_2d_Ginf_0cut0sigma_Ne10_s",  "fig4 kanamycin3_2d_Ginf_0cut0sigma_Ne10_r", "fig4 kanamycin3_2d_Ginf_0cut0sigma_Ne10_s", "S3_2d_Ginf_0cut0sigma_Ne20_r", "S3_2d_Ginf_0cut0sigma_Ne20_s", "S4_2d_Ginf_0cut0sigma_Ne20_r", "S4_2d_Ginf_0cut0sigma_Ne20_s", "S5_2d_Ginf_0cut0sigma_Ne20_r", "S5_2d_Ginf_0cut0sigma_Ne20_s"]
    
    if ROI == "300x4":
        res_sus_datanames = ["01_2d_Ginf_300x4", "02_2d_Ginf_300x4", "03_2d_Ginf_300x4", "04_2d_Ginf_300x4", "05_2d_Ginf_300x4", "06_2d_Ginf_300x4", "07_2d_Ginf_300x4", "08_2d_Ginf_300x4", "09_2d_Ginf_300x4", "10_2d_Ginf_300x4", "fig4 kanamycin1_2d_Ginf_300x4_r", "fig4 kanamycin1_2d_Ginf_300x4_s", "fig4 kanamycin2_2d_Ginf_300x4_r","fig4 kanamycin2_2d_Ginf_300x4_s",  "fig4 kanamycin3_2d_Ginf_300x4_r", "fig4 kanamycin3_2d_Ginf_300x4_s", "S3_2d_Ginf_300x4_r", "S3_2d_Ginf_300x4_s", "S4_2d_Ginf_300x4_r", "S4_2d_Ginf_300x4_s", "S5_2d_Ginf_300x4_r", "S5_2d_Ginf_300x4_s"]

    return res_sus_datanames

def fit_choice():
    fits = ["single", "double", "2DTIRF", "3DTIRF", "2Dcutsigma"]
    while True: 
        fit = input('Choose one of the following fits: "single", "double", "2DTIRF", "3DTIRF", "2Dcutsigma": ')

        if fit not in fits: 
            print("Choose a valid fit!")
        else: 
            break
    return fit

########################################################################
#Output classes
#-----------------------------------------------------------------------
#0 - susceptible, 1 - resistant

#Gentamycin 01-10 correspond to these MIC values (in this order): 1, 1, 4, 16, 32, 2, 8, 64, 4, 8 (µg/ml)
#whereby susceptible < MIC 16 <= resistant

y_vals = [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0] 


########################################################################
#ROI options
#-----------------------------------------------------------------------
#"20x20"
#"300x4"

########################################################################
#nnparams: Parameter input choice
#-----------------------------------------------------------------------
#"_m_mono_", "t_mono" belong to the single exponential fit. 
#"_m1_", "_m_2_", "t1", "t_2" belong to the double exponential fit. 
#"N", "kappa", "tD" belong to the TIRF fit (3D or 2D)
#"_g0_", "intensities": fit-independent
#"R": Rsquared
#"timeseries": intensity timeseries
#"ACF": Autocorrelation function


########################################################################
#Fit choice 
#-----------------------------------------------------------------------
#"single": single exponential fit
#"double": double exponential fit
#"2DTIRF": 2D TIRF (with Ginf)
#"3DTIRF": 3D TIRF (no Ginf)
#"2Dcutsigma": 2D TIRF whereby first point of ACFs were cut off and fit was forced through the second point (first point in new ACF)
#"none": when either ACFs or timeseries wants to be inputted. Mostly because these were only saved for one fit.

########################################################################
#Antibiotic choice
#-----------------------------------------------------------------------
equal_Genta = True
cross_val = False

while True:
    s = input("Type S for standard, C for choices: ")
    if s not in ["S", "C"]: 
        print("Not a valid option!")
    else: 
        break

if s == "S": 
    antibiotic = "Gentamycin"
    method = "p-CNN"
    fit = "2DTIRF"
    ROI = "20x20"
    nn_params = [  "_g0_"]
    res_sus_datanames = filenames(ROI, fit)
    print_input_summary(nn_params, ROI, fit)

    if cross_val == False and antibiotic == "Gentamycin":
        total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(cross_val)
    else: 
        total_X_arr, total_Y_arr = param_data_processing(cross_val)
    
    model = "CNN"

elif s == "C":

    antibiotic_options = ["Gentamycin", "Kanamycin", "Trimethoprim", "all"]
    while True: 
        antibiotic = input('Choose one of the following antibiotics: "Gentamycin", "Kanamycin", "Trimethoprim", "all": ')

        if antibiotic not in antibiotic_options: 
            print("Choose a valid antibiotic!")

        else: 
            break
    ########################################################################
    #Method choice
    #p-CNN: input is fit params
    #v-CNN: input is individual video frames - should not be used, except if for later cnn
    #acf-1DCNN: input: individual ACF curves (no vid distinction)
    #acf-LSTM: input: flattened ACFs for each vid
    #i-CNNLSTM: input is intensity timeseries. Requires having trained a CNN on the antibiotics individual frames beforehand, to then load in its state dict for feature extraction
    #i-3DCNN: input is intensity timeseries. DOES NOT WORK
    #pf-LSTM: flattened params for each vid. DOES NOT WORK


    methods = ["p-CNN", "p-CNN-300x4", "pf-LSTM", "acf-1DCNN", 
            "acf-LSTM", "v-CNN", "i-CNNLSTM"]

    while True: 
        method = input('Choose one of the following methods: "p-CNN", "p-CNN-300x4", "pf-LSTM", "acf-1DCNN", "acf-LSTM", "v-CNN", "i-CNNLSTM":  ' )

        if method not in methods: 
            print("Choose a valid method!")
        else: 
            break


    ########################################################################


    if method == "p-CNN":
        fit = fit_choice()
        ROI = "20x20"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)
        print(res_sus_datanames)
        
        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(cross_val = cross_val)
        model = "CNN"

    if method == "p-CNN-300x4":

        fit = fit_choice()
        ROI = "300x4"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)

        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(cross_val = cross_val)
        model = "CNN"
    
    elif method == "pf-LSTM":
        #flattened fit params 
        
        fit = fit_choice()
        ROI = "20x20"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)
        
        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(flatten = True, cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(flatten = True, cross_val = cross_val)
        model = "LSTM"

    elif method == "acf-LSTM":
        #flattened ACFs 
        
        fit = "none"
        ROI = "20x20"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)
        
        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(flatten = True, cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(flatten = True, cross_val = cross_val)
        model = "LSTM"

    elif method == "acf-1DCNN":
        fit = "none"
        ROI = "20x20"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)

        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(split_ACF=True, cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(split_ACF = True, cross_val = cross_val)
        model = "1DCNN"

    elif method == "v-CNN":
        #just to show that one SHOULD NOT use this method (see report)
        fit = "none"
        ROI = "20x20"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)

        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(split_timeseries = True,cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(split_timeseries = True, cross_val = cross_val)
        model = "CNN"

    elif method == "i-CNNLSTM":
        fit = "none"
        ROI = "20x20"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)

        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(split_timeseries = False, cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(split_timeseries = False, cross_val = cross_val)
        model = "CNN_LSTM"

    elif method == "i-3DCNN":
        #"3D block" of intensities fed in
        #could try 3D block of ACFS... but this method has not worked so far
        #NOT WORKING
        fit = "none"
        ROI = "20x20"
        nn_params = input_params(fit)
        res_sus_datanames = filenames(ROI, fit)
        print_input_summary(nn_params, ROI, fit)

        if cross_val == False and antibiotic == "Gentamycin":
            total_X_tr_arr, total_X_ts_arr, total_Y_tr_arr, total_Y_ts_arr = param_data_processing(split_timeseries = False,cross_val = cross_val)
        else: 
            total_X_arr, total_Y_arr = param_data_processing(split_timeseries = False, cross_val = cross_val)
        model = "3DCNN"

#to determine the sizes of things
if cross_val == False and antibiotic == "Gentamycin":
    arr = total_X_tr_arr
else: 
    arr = total_X_arr

if method == "acf-1DCNN":
    nr_params = 1
    seq_length = 100


elif "timeseries" in nn_params:
  
    vid_seq_length = arr.shape[2]
    nr_params = 1
    seq_length = 128

elif method in ["pf-LSTM" , "acf-LSTM"]:
    nr_params = 1
    seq_length = arr.shape[2]

else: 
    nr_params = arr.shape[1]


nr_classes = len(set(y_vals))       


run_CNN_LSTM_models(cross_val = cross_val)

#modeltoconfig()
pass


