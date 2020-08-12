''' 
The file consists of the database class, and read and write function needed for loading the dataset.
Because of the RAM limitations, every time the data is needed, it is separately loaded and convertet to spectrogram.
'''

import csv 
import numpy as np 
import os
import torch
import librosa                    
import soundfile as sf 


#SYSTEM PARAMETERS
#allowed_genres = ["blues","classical","country","disco","hiphop","metal","pop","reggae","rock"]
allowed_genres = ["classical", "pop", "rock"]
label_row = 59 #TODO: revisit and generalize this part
name_row = 0


def readAudio(path):
    '''
    Read audio file on location path and creates mel spectrogram on logaritmic scale.
    Spectrogram is normalized and corresponding mean and standard deviation are returned.
    '''
    audio, sample_rate = librosa.load(path)
    melSpect = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=220)
    melSpect_dB = librosa.power_to_db(melSpect)
    
    mean_spect = melSpect_dB.mean()
    var_spect = melSpect_dB.var()
    
    spect = (melSpect_dB - mean_spect)/np.sqrt(var_spect)

    return spect, mean_spect, np.sqrt(var_spect), sample_rate
    

def writeAudio(spectrogram, sample_rate, mean, sigma, output_file_name):
    '''
    For given normalized mel spectrogram, sample_rate, mean and standard deviation of
    the original recording, write .wav audiofile with the name output_file_name + '.wav'
    '''
    spectrogram = spectrogram*sigma + mean
    spectrogram = librosa.db_to_power(spectrogram)    
    
    file_name = output_file_name + '.wav'

    audio_signal = librosa.feature.inverse.mel_to_audio(spectrogram, sr=sample_rate)
    sf.write(file_name, audio_signal, sample_rate)
    

class MusicDataset(torch.utils.data.Dataset):

    #initialize by reading all file names and labels
    def __init__(self, path, csv_path):
    
        self.path_label = []    #tuples - (file location, data label)
        self.classes = []       #all classes (possible labels)
        self.labels_onehot = [] #all possible labels one hot encoded
            
        with open(csv_path) as csvfile:
            csv_reader = csv.reader(csvfile,delimiter=',')
            next(csv_reader) #skip first row
            
            for row in csv_reader:
                if (row[label_row] in allowed_genres): 
                
                    #remember audiofile paths and corresponding labels
                    audio_path = os.path.join(path,row[label_row],row[name_row])
                    self.path_label.append((audio_path, row[label_row])) 
                    
                    #remember all possible classes
                    if (row[label_row] not in self.classes):
                        self.classes.append(row[label_row])
                        

        #create onehot encoding for data labels
        i = 0
        for label in self.classes:
            label_onehot = torch.zeros(len(self.classes)) 
            label_onehot[i] = 1
            self.labels_onehot.append(label_onehot)
         

    #getitem reads audiofile - note that it is time consuming
    def __getitem__(self,index):
    
        #read audiofile - spectrogram, mean, standard deviation and sample rate
        spect, mean, sigma, sr = readAudio(self.path_label[index][0])
        
        #reshape spectrogram to represent 1 channel
        spect = spect[:,:1280] #standard size
        spect = torch.Tensor(spect).reshape(1,spect.shape[0],spect.shape[1])
        
        #get label encoded as int
        label = self.path_label[index][1]
        label_int = self.classToInt(label)
        
        return spect, mean, sigma, label_int

    def __len__(self):
        return len(self.path_label)
        
        
    #Conversions for different label encodings
    def classToOnehot(self,label):
        return self.labels_onehot[self.classes.index(label)]
        
    def onehotToClass(self,label_onehot):
        return self.classes[self.labels_onehot.index(label_onehot)]
        
    def classToInt(self,label):
        return self.classes.index(label)
    
    def intToClass(eelf,index):
        return self.classes.get(index)
        
    def onehotToInt(self,label_onehot):
        return self.labels_onehot.index(label_onehot)
        
    def intToOnehot(self,index):
        return self.labels_onehot.get(index)