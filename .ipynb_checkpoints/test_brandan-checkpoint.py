import torch
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchsummary import summary

import gzip
import tqdm
import os
#from models.cdcn import cdcn

MODEL_PATH = "/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/weights/best_model_retrained.pth"
from attempt2 import resnet50_1d  # Directly import from attempt2.py
num_classes = 8  # Change this if your model was trained with a different number of classes
model = resnet50_1d(num_classes=num_classes).to(DEVICE)
# Load the state dictionary into the model
model.load_state_dict(state_dict)
model.eval()

class IQDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        sample = torch.from_numpy(sample).float()
        # Normalize data
        magnitude = torch.sqrt(torch.sum(sample**2, dim=1, keepdim=True))
        sample = sample / magnitude

        label_tensors = torch.tensor(label, dtype=torch.long)

        return sample, label_tensors


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using {} device'.format(device))

folder_path = '/data2/brandan/RFSPOOF/May16'
for filename in os.listdir(folder_path):
    test_data_tensors = []
    test_label_tensors = []
    file_path = os.path.join(folder_path, filename)
    print(f"File: {file_path}")
    file_name = file_path
    file_name = f"/data2/brandan/RFSPOOF/May16/indexing.iq"
    target = 1
    data = np.fromfile(file_name, dtype="float32")
    size_data = len(data)
    #print("Size of data: ", size_data)
    size_half = int(size_data / 2)
    real_part = data[0::2]
    #print("Real Length: ", len(real_part))
    imag_part = data[1::2]
    #print("Imag Length: ", len(imag_part))
    print("Mean Magnitude: ", np.mean(np.abs(real_part + imag_part * 1j)))
    #print(real_part[5000])
    #print(real_part[0])
    #print(imag_part[5000])
    #print(imag_part[0])

    for x in range(4800, 6000):
        # Implement a 10% hop
        combined_data = np.vstack((real_part[(x+1)*500:(x+1)*500+10000], imag_part[(x+1)*500:(x+1)*500+10000]))
        array = np.zeros(combined_data.shape[1])
        array.fill(target)
        test_data_tensors.append(combined_data)
        test_label_tensors.append(target)

    '''
    #for i in range(8):
    for j in range(2, 3):
        for k in range(1, 2):
            file_name = f"./processed_data_justin/GANdata_{j}m_run{k}.npy.gz"
            print(file_name)
            f = gzip.GzipFile(file_name, "r")
            data = np.load(f)
            size_data = len(data)
            size_half = int(size_data / 2)
            real_part = data[0:size_half]
            imag_part = data[size_half:size_data]
            for x in range(0, 6000):
                # Implement a 10% hop
                combined_data = np.vstack((real_part[(x+1)*500:(x+1)*500+10000], imag_part[(x+1)*500:(x+1)*500+10000]))
                array = np.zeros(combined_data.shape[1])
                array.fill(10)
                test_data_tensors.append(combined_data)
                test_label_tensors.append(10)
    '''


    # Format the data
    test_data_tensors = np.stack(test_data_tensors, axis=0)
    test_label_tensors = np.array(test_label_tensors)

    # Randomize the data
    indices = np.random.permutation(len(test_data_tensors))
    test_data = test_data_tensors[indices]
    test_labels = test_label_tensors[indices]
    #print(test_data.shape)
    #print(test_labels.shape)

    # Create the dataset
    test_dataset = IQDataset(test_data, test_labels)

    batch_size = 16

    # Load the data
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(7)
    np.random.seed(7)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    #print(device)
    net = resnet50_1d(num_classes=8)
    net.to(device)
    net.train()
    #summary(net, input_size = (1, 2), batch_size = 8)


    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    PATH = F"./weights/best_model_retrained.pth"
    #net = cdcn(num_classes=8)
    net = resnet50_1d(num_classes=8)
    #net.load_state_dict(torch.load(PATH))
    net.load_state_dict(torch.load(PATH, map_location=torch.device("cpu")))
    net.to(device)
    net.eval()

    correct = 0
    total = 0
    conf_threshold = 0.85
    fooled = 0
    confidence = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            
            #for x in range(0, len(predicted)):
                #print(predicted)
                #print(labels)
            #    if ~(predicted[x] == labels[x]):
            #        print("Predicted: ", predicted[x])
            #        print("Label: ", labels[x])
            correct += (predicted == labels).sum().item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, classes = torch.max(probs, 1)
           
            print(predicted)
            # For GAN
            counter = 0
            for conf_i in conf:
                # For testing normal mode
                if conf_i > conf_threshold and classes[counter] == target:
                # For generative attack
                #if (conf_i > conf_threshold):
                    fooled = fooled + 1
                    confidence = confidence + conf_i
                counter = counter + 1
            
            # For EOT
            '''
            for x in range(0, len(predicted)):
                print(predicted)
                if (predicted[x] == 3):
                    fooled = fooled + 1
                    confidence = confidence + conf[x]
            '''

    ### For fingerprinter ###
    acc  = correct / total * 100## stores the accuracy computed in the above loop
    print('Accuracy of the network on the 10000 test images: %d %%' % (acc))

    ### For Attack ###
    fooled_acc = fooled / total * 100
    if (fooled == 0):
        print("No fools")
    else:
        fooled_conf = confidence / fooled * 100
        print("Average confidence: ", fooled_conf)
    print('Percentage of successful fools: %d %%' % (fooled_acc), " , Number fooled: ", fooled, " , Total: ", total)
    #print("Number fooled: ", fooled)
    #print("Number total: ", total)
