import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import os
import datasets
from datasets import build_dataset

# Adapt the pathes 
class_num = 8
epoch_num = 30
log_file = 
batch_size = 8
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_dir = 
data_dir = 

#Define model
model = models.resnet50(pretrained=True)
os.makedirs(output_dir, exist_ok=True)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dataset_train, _ = build_dataset(is_train=True, nb_classes=class_num,dataset_root=data_dir,transform=data_transforms['train'])
dataset_val, _ = build_dataset(is_train=False, nb_classes=class_num,dataset_root=data_dir,transform=data_transforms['val'])
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
dataloaders = {'train': dataloader_train, 'val': dataloader_val}
dataset_sizes = {'train':len(dataset_train), 'val':len(dataset_val) }

#Change classifier accordingly for different models
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, class_num)
model = model.to(device)

criterion = nn.functional.cross_entropy
criterion_extra = nn.KLDivLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_model(model, criterion, optimizer, num_epochs=epoch_num):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for batch_num,(inputs,label1, label2) in enumerate(dataloaders[phase]):
                optimizer.zero_grad()
                inputs = inputs.to(device)
                label1 = label1.to(device)
                label2 = label2.to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # Choose one loss 
                    # loss = criterion(torch.nn.functional.log_softmax(outputs), label2) # Naive

                    loss = criterion(outputs, label1) + 0.1 * criterion_extra(torch.nn.functional.log_softmax(outputs), label2) # Ours

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        acc=torch.sum(preds == label1.data)/batch_size
                        print(f'batch {batch_num}/{len(dataloaders[phase])}, loss: {loss.item()}', f'acc: {acc}', end='\r')
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == label1.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            # save weights
            if phase == 'val':
                torch.save(model.state_dict(), os.path.join(output_dir, f'res50_{epoch}.pth'))
                with open(os.path.join(output_dir, log_file), 'a') as f:
                    f.write(f'epoch: {epoch}, loss: {epoch_loss}, acc: {epoch_acc}\n')

train_model(model, criterion, optimizer, num_epochs=epoch_num)