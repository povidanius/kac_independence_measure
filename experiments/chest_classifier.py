import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchvision.models import inception_v3, resnet18
from torch import optim
import sys
import os
from collections import OrderedDict


data_path='/home/tank/Downloads/chest_xray/chest_xray'

sys.path.insert(0, "../")
from kac_independence_measure import KacIndependenceMeasure
from torch.nn.functional import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


#os.listdir('./chest-xray-pneumonia/')
#image = './chest-xray-pneumonia/train/normal/IM-0115-0001.jpeg'
#img = plt.imread(image)
#plt.imshow(img, cmap='gray')
REGULARIZER = 0
LOSS = 1

kim = KacIndependenceMeasure(512, 2, lr=0.0007, input_projection_dim = 32, weight_decay=0.01,device=device) #0.007


train_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
training_dataset = ImageFolder(data_path + '/train', transform=train_transform)
validation_dataset = ImageFolder(data_path + '/val', transform=train_transform)
testing_dataset = ImageFolder(data_path + '/test', transform=test_transform )
len_train= len(training_dataset.samples)
len_test= len(testing_dataset.samples)
len_val= len(validation_dataset.samples)
print(len_train)
print(len_test)
print(len_val)

train_loader = DataLoader(dataset=training_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=validation_dataset, shuffle= True)
test_loader = DataLoader(dataset= testing_dataset, shuffle=False)


model = resnet18(pretrained=True) #, aux_logits=False)

# intermediate activations
"""
model.layer1[0].register_forward_hook(get_activation('layer1_0'))
model.layer1[1].register_forward_hook(get_activation('layer1_1'))
model.layer2[0].register_forward_hook(get_activation('layer2_0'))
model.layer2[1].register_forward_hook(get_activation('layer2_1'))
model.layer3[0].register_forward_hook(get_activation('layer3_0'))
model.layer3[1].register_forward_hook(get_activation('layer3_1'))
model.layer4[0].register_forward_hook(get_activation('layer4_0'))
model.layer4[1].register_forward_hook(get_activation('layer4_1'))
"""
model.avgpool.register_forward_hook(get_activation('avgpool'))


model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr = 0.0002, weight_decay=0.00001)
number_of_epoch = 7

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

#num_kac_iters = 200
optimize_kac_every_iters = 15

dep_history = []

reg_alpha = 0.1

use_regularization = False
mode = LOSS

for epoch in range(number_of_epoch):
    
    train_correct = 0
    test_correct = 0
    train_iter_loss = 0.0
    test_iter_loss = 0.0
    train_iteration = 0
    test_iteration = 0
    
    model.train()
    iteration = 0
    for data,label in train_loader:
        
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        
        pred = model(data)     
        bottleneck = activation['avgpool'].squeeze()
        y = torch.nn.functional.one_hot(label).float()


        if use_regularization:
            if (iteration % optimize_kac_every_iters  == 0) and mode == LOSS:
                mode = REGULARIZER
            elif (iteration % optimize_kac_every_iters  == 0) and mode == REGULARIZER:
                mode = LOSS

        if mode == REGULARIZER:
            reg = kim.forward(bottleneck, y, update=True)

            print("Mode: REGULARIZER")
            print(iteration % optimize_kac_every_iters)
            print("reg {}".format(reg))
            #print("bottleneck: {}, y {}".format(bottleneck.shape, y.shape))
            dep_history.append(reg.detach().cpu().numpy())
            #
            #plt.show(block=False)
            #breakpoint()
            #else:
            #plt.plot(dep_history)
            #plt.show()
            #print("Loss")
        elif mode == LOSS:

            # idea: use Agglomerative information bottleneck?

            print("Mode: LOSS")

            loss = loss_fn(pred, label) 
            if use_regularization:
                reg = kim.forward(bottleneck, y, update=False)
                print("reg {}".format(reg))
                loss = loss - reg_alpha * reg # loss -> min.., dep -> max

            loss.backward()
            optimizer.step()


        
            train_iter_loss += loss.item()
            train_iteration += 1
        
            _, predicted = torch.max(pred, 1)
            train_correct += (predicted == label).sum()
        iteration = iteration + 1

            
        
    train_loss.append(train_iter_loss/train_iteration)
    train_accuracy.append(100*float(train_correct)/len_train)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
        }, format("chest_checkpoint_{}.pt".format(epoch)))

    model.eval()
    with torch.no_grad():
        for data,label in val_loader:
        
            data = data.to(device)
            label = label.to(device)
        
            pred = model(data)
            loss = loss_fn(pred, label)
        
            test_iter_loss += loss.item()
            test_iteration += 1
        
            _, predicted = torch.max(pred, 1)
            test_correct += (predicted == label).sum()
        
    test_loss.append(test_iter_loss/test_iteration)
    test_accuracy.append(100*float(test_correct)/len_val)
    
    print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'
           .format(epoch+1, number_of_epoch, train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]))


corrected = 0

model.eval()
for data, label in test_loader:
    data = data.to(device)
    label = label.to(device)
    
    pred = model(data)
    _, predicted = torch.max(pred, 1)
    
    corrected += (predicted == label).sum()
    
accuracy = 100 * float(corrected)/ len_test

print(f'Test accuracy is {accuracy :.3f}')
