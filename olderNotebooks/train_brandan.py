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
import sys
sys.path.append("/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing")

from models.attempt2 import resnet50_1d
import gzip
import tqdm
import os

save_dir = "./weights"
PATH = os.path.join(save_dir, "weight0to7.pth")

# Check if directory exists, if not, create it
if not os.path.exists(save_dir):
    print(f"Directory {save_dir} does not exist. Creating it now...")
    os.makedirs(save_dir)

# Check if file path is writable
try:
    with open(PATH, "wb") as f:
        pass  # Just attempt to open and close the file
    print(f"Checkpoint path is valid. Model will be saved at: {PATH}")
except IOError as e:
    print(f"ERROR: Cannot write to {PATH}. Please check permissions or available space.")
    exit(1)  # Exit to avoid wasting training time

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

train_data_tensors = []
train_label_tensors = []
test_data_tensors = []
test_label_tensors = []
for i in range(7):
    for j in range(1, 3):
        for k in range(2):
            # TO DO
            ###updated this line -------------------------------------------
            file_name = f"/home/jfeng/Desktop/jfeng/rf_spoofing/spoofing/processed_data_justin/data{i}_{j}m_run{k}.npy.gz"
            print(file_name)
            f = gzip.GzipFile(file_name, "r")
            data = np.load(f)
            size_data = len(data)
            size_half = int(size_data / 2)
            real_part = data[0:size_half]
            imag_part = data[size_half:size_data]
            for x in range(0, 4800):
                #combined_data = np.vstack((real_part[(x+1)*5000:(x+1)*5000+5000], imag_part[(x+1)*5000:(x+1)*5000+5000]))
                # Implement a 10% hop
                combined_data = np.vstack((real_part[(x+1)*500:(x+1)*500+10000], imag_part[(x+1)*500:(x+1)*500+10000]))
                array = np.zeros(combined_data.shape[1])
                array.fill(i)
                train_data_tensors.append(combined_data)
                train_label_tensors.append(i)
            for x in range(4800, 6000):
                # Implement a 10% hop
                combined_data = np.vstack((real_part[(x+1)*500:(x+1)*500+10000], imag_part[(x+1)*500:(x+1)*500+10000]))
                array = np.zeros(combined_data.shape[1])
                array.fill(i)
                test_data_tensors.append(combined_data)
                test_label_tensors.append(i)

# Format the data
train_data_tensors = np.stack(train_data_tensors, axis=0)
train_label_tensors = np.array(train_label_tensors)
print(train_data_tensors.shape)
print(train_label_tensors.shape)
test_data_tensors = np.stack(test_data_tensors, axis=0)
test_label_tensors = np.array(test_label_tensors)

# Randomize the data
indices = np.random.permutation(len(train_data_tensors))
train_data = train_data_tensors[indices]
train_labels = train_label_tensors[indices]
indices = np.random.permutation(len(test_data_tensors))
test_data = test_data_tensors[indices]
test_labels = test_label_tensors[indices]
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

# Create the dataset
train_dataset = IQDataset(train_data, train_labels)
test_dataset = IQDataset(test_data, test_labels)

batch_size = 16

# Load the data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
dataiter = iter(train_loader)


torch.manual_seed(7)
np.random.seed(7)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
net = resnet50_1d(num_classes=7)
net.to(device)
net.train()
#summary(net, input_size = (1, 2), batch_size = 8)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001) #LEARNING RATE & OPTIMIZER
num_epochs = 10 #NUMBER OF EPOCHS
running_loss_list = []
for epoch in (range(num_epochs)):
    running_loss = 0.0
    for i, data in tqdm.tqdm(enumerate(train_loader, 0)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #Send input and label to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 200 == 199:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
            running_loss_list.append(running_loss)
            running_loss = 0.0

print('Training Complete')
#PATH = F"./weights/a.pth"
# TO DO
#PATH = F["./weights/weight0to7.pth"]

torch.save(net.state_dict(), PATH)

#def plot_loss_curve(running_loss_list):
#  plt.plot(running_loss_list)
#plot_loss_curve(running_loss_list)


dataiter = iter(test_loader)
images, labels = next(dataiter)

#PATH = F"./weights/a.pth"
# TO DO
#PATH = F[INSERT PATH FOR WEIGHTS]

net = resnet50_1d(num_classes=7)
net.load_state_dict(torch.load(PATH))
net.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        #for x in range(0, len(predicted)):
        #    if ~(predicted[x] == labels[x]):
                #print("Predicted: ", predicted[x])
                #print("Label: ", labels[x])
        correct += (predicted == labels).sum().item()
acc  = correct / total * 100## stores the accuracy computed in the above loop
print('Accuracy of the network on the 10000 test images: %d %%' % (acc))


