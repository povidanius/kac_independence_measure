import hub
import torch
from torchvision import transforms, models

ds = hub.load('hub://activeloop/cifar10-train')

tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

hub_loader = ds.pytorch(num_workers=0, batch_size=256, transform={
                        'images': tform, 'labels': None}, shuffle=True)

net = models.resnet18(pretrained=False)
net.fc = torch.nn.Linear(net.fc.in_features, len(ds.labels.info.class_names))
    
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(hub_loader):
        images, labels = data['images'], data['labels']
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels.reshape(-1))
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
