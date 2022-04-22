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
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


#torch.manual_seed(31337)

#data_path='/home/tank/Downloads/chest_xray/chest_xray/chest_xray/merged'
data_path='/home/tank/Downloads/melanoma/joined'

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

kim = KacIndependenceMeasure(512, 2, lr=0.001, input_projection_dim = 0, weight_decay=0.01, device=device) #0.007

"""
train_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                      transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10), #
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                      transforms.ToTensor(),
                                      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), 
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
"""

train_transform = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(10), #
                                      transforms.ColorJitter(brightness=1, contrast=1, saturation=1),
                                      transforms.ToTensor(),
                                      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

test_transform = transforms.Compose([transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

full_dataset = ImageFolder(data_path, transform=train_transform)

#training_dataset = ImageFolder(data_path + '/train', transform=train_transform)
#testing_dataset = ImageFolder(data_path + '/test', transform=test_transform )
#validation_dataset = ImageFolder(data_path + '/val', transform=test_transform )



#len_val= len(validation_dataset.samples)
#print(len_train)
#print(len_test)
#print(len_val)

print(len(full_dataset.samples))
#training_dataset, testing_dataset = torch.utils.data.random_split(full_dataset, [5000, 856])
training_dataset, testing_dataset = torch.utils.data.random_split(full_dataset, [9000, 1605])
len_train= len(training_dataset)
len_test= len(testing_dataset)
#breakpoint()

print("Train: {}, test: {}".format(len_train, len_test))
batch_size = 128 # 128

train_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
#val_loader = DataLoader(dataset=validation_dataset, shuffle= True)
test_loader = DataLoader(dataset= testing_dataset, shuffle=False)


model = resnet18(pretrained=True) #, aux_logits=False)
#model.fc = nn.Linear(512, 2)
model.fc = nn.Sequential(
    #nn.Dropout(0.5),
    nn.Linear(512, 32),
    nn.ReLU(),
    nn.BatchNorm1d(32),
    nn.Linear(32, 2)
)

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
model.avgpool.register_forward_hook(get_activation('bottleneck'))


model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(params=model.parameters(), lr = 0.0002, weight_decay=0.00001)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

num_kac_iters = 300
#optimize_kac_every_iters = 20
#switch_to_loss_every_iters = 100

dep_history = []

reg_alpha = 9.0 #0.1

if len(sys.argv) < 2:
    print(sys.argv)
    breakpoint()
    print("Usage {} 1/0".format(sys.argv[0]))
    sys.exit(0)

if sys.argv[1] == "0":
    use_regularization = False
else:
    use_regularization = True


mode = LOSS

number_of_epoch = 3
if use_regularization:
    number_of_epoch = 2*number_of_epoch
    

global_iteration = 0
for epoch in range(number_of_epoch):
    
    train_correct = 0
    test_correct = 0
    train_iter_loss = 0.0
    test_iter_loss = 0.0
    train_iteration = 0
    test_iteration = 0
    num_train = 0

    model.train()
    iteration = 0
    for data,label in train_loader:
        
        global_iteration += 1

        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)        
        pred = model(data)     

        #breakpoint()
        bottleneck = activation['bottleneck'].squeeze()
        y = torch.nn.functional.one_hot(label).float()
        #print(label)


        loss = loss_fn(pred, label) 
             
                
        reg0 = kim.forward(bottleneck.clone().detach().to(device), y.clone().detach().to(device), update=True)
        print("Dep iteration: epoch {}, iteration {},  reg_estim {} ".format(epoch, iteration,  reg0))
        dep_history.append(reg0.detach().cpu().numpy())
        writer.add_scalar("Dep/train", reg_alpha * reg0, global_iteration)
        iteration = iteration + 1
        if epoch % 2 != 0 and use_regularization:
           continue

        if epoch % 2 == 0 and use_regularization:
            reg = kim.forward(bottleneck, y, update=False)
            dep_history.append(reg.detach().cpu().numpy())
            writer.add_scalar("Dep_loss/train", reg_alpha * reg, global_iteration)
            writer.add_scalar("Loss/train", loss, global_iteration)

            print("Loss iteration: epoch {}, iteration {}, loss {}, reg {} ".format(epoch, iteration, loss, reg))
            loss =  loss - reg_alpha * reg # loss -> min.., dep -> max
            writer.add_scalar("LossReg/train", loss, global_iteration)

            #reg0 = kim.forward(bottleneck.clone().detach().to(device), y.clone().detach().to(device), update=True)

        loss.backward()
        optimizer.step()


        
        train_iter_loss += loss.item()
        train_iteration += 1
        
        _, predicted = torch.max(pred, 1)
        train_correct += (predicted == label).sum()
        num_train += batch_size
        iteration = iteration + 1

     
    if train_iteration > 0:    
        train_loss.append(train_iter_loss/train_iteration)
        train_accuracy.append(100*float(train_correct)/num_train)
    else:
        continue
    #train_accuracy.append(100*float(train_correct)/len_train)
    
    if False:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, format("chest_checkpoint_{}.pt".format(epoch)))

    print ('Epoch {}/{}, Training Loss: {:.3f}, Training Accuracy: {:.3f}'.format(epoch+1, number_of_epoch, train_loss[-1], train_accuracy[-1]))

#plt.plot(dep_history)
#plt.show()
#timestr = time.strftime("%Y%m%d-%H%M%S")
#plt.savefig('./chest_{}.png'.format(timestr))

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
    print("Regularization: {}".format(use_regularization))
    writer.add_scalar("Acc/test", accuracy, global_iteration)

writer.close()

with open("./19result_chest_{}_{}.txt".format(use_regularization, reg_alpha),"a") as f:
#with open("./13result_chest_{}_{}.txt".format(use_regularization, reg_alpha),"a") as f:
    #f.write("{} {} \n".format(accuracy, test_accuracy[-1]))
    f.write("{}\n".format(accuracy))
    

