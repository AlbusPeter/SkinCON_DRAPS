import torch
import os
from torchvision import models,transforms
from datasets import build_dataset
from torch.utils.data import DataLoader
import tqdm
import numpy as np

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        topk = [1,5]
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
chosen_model = "m_resnet50"
data_dir = 

weight_pth ={}
weight_pth["m_resnet18"] = 
weight_pth["m_resnet50"] = 
weight_pth["ori_resnet18"] = 
weight_pth["ori_resnet50"] = 
weight_pth["m_resnet101"] = 
weight_pth["m_resnet152"] = 
weight_pth["ori_resnet101"] = 
weight_pth["ori_resnet152"] = 
weight_pth["m_resnext101"] = 
weight_pth["ori_resnext101"] = 
weight_pth["m_vgg16"] = 
weight_pth["ori_vgg16"] = 
weight_pth["m_shufflenet"] = 
weight_pth["ori_shufflenet"] = 
weight_pth["m_densenet"]= 
weight_pth["ori_densenet"]= 

model={}
model["m_resnet18"] = models.resnet18(pretrained=False)
model["ori_resnet18"] = models.resnet18(pretrained=False)
model["m_resnet50"] = models.resnet50(pretrained=False)
model["ori_resnet50"] = models.resnet50(pretrained=False)
model["m_resnet101"] = models.resnet101(pretrained=False)
model["ori_resnet101"] = models.resnet101(pretrained=False)
model["m_resnet152"] = models.resnet152(pretrained=False)
model["ori_resnet152"] = models.resnet152(pretrained=False)
model["m_vgg16"] = models.vgg16(pretrained=False)
model["ori_vgg16"] = models.vgg16(pretrained=False)
model["m_shufflenet"] = models.shufflenet_v2_x1_0(pretrained=False)
model["ori_shufflenet"] = models.shufflenet_v2_x1_0(pretrained=False)
model["m_densenet"] = models.densenet161(pretrained=False)
model["ori_densenet"] = models.densenet161(pretrained=False)
model["m_resnext101"] = models.resnext101_64x4d(pretrained=False)
model["ori_resnext101"] = models.resnext101_64x4d(pretrained=False)

transform_resnet=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
dataset_val_resnet, _ = build_dataset(is_train=False, nb_classes=8,dataset_root=data_dir,transform=transform_resnet)
dataloader_val_resnet = DataLoader(dataset_val_resnet, batch_size=8, shuffle=True)

# resnet
num_ftrs = model[chosen_model].fc.in_features
model[chosen_model].fc = torch.nn.Linear(num_ftrs, 8)

# # vgg16
# num_ftrs = model[chosen_model].classifier[6].in_features
# model[chosen_model].classifier[6] = torch.nn.Linear(num_ftrs, 8)

# # shufflenet
# num_ftrs = model[chosen_model].classifier.in_features
# model[chosen_model].classifier = torch.nn.Linear(num_ftrs, 8)

model[chosen_model].load_state_dict(torch.load(weight_pth[chosen_model]))
model[chosen_model] = model[chosen_model].to(device)
model[chosen_model].eval()


dataloader_val = dataloader_val_resnet
dataset_train, _ = build_dataset(is_train=True, nb_classes=8,dataset_root=data_dir,transform=transform_resnet)
dataloader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
pred_scores = []
for batch_num,(inputs,label1, label2) in tqdm.tqdm(enumerate(dataloader_train),total=len(dataloader_train)):
    inputs = inputs.to(device)
    outputs = model[chosen_model](inputs)
    outputs = outputs.cpu().detach().numpy()
    for i in range(len(outputs)):
        pred = outputs[i][label1[i]]
        pred_scores.append(pred)

alpha = 0.1
sorted_list=sorted(pred_scores)
threshold = sorted_list[int(np.floor(len(sorted_list)*alpha) - 1)]

num_after_threshold = 0
acc_after_threshold = 0
total = 0
acc_all_after_threshold = 0
for batch_num,(inputs,label1, label2) in tqdm.tqdm(enumerate(dataloader_val),total=len(dataloader_val)):
    inputs = inputs.to(device)
    outputs = model[chosen_model](inputs)
    label1 = label1.to(device)
    for i in range(len(outputs)):
        temp_sum = torch.sum(outputs[i]>=threshold)
        list_after_threshold = []
        tmp_acc_all_after_threshold = 0
        if(temp_sum>1):
            list_after_threshold = torch.argsort(outputs[i],descending=True).cpu().detach().tolist()[:temp_sum]
        else:
            temp_sum = 1
            list_after_threshold.append(torch.argmax(outputs[i]).cpu().detach().tolist())
        num_after_threshold+= temp_sum
        argsort_label2 = torch.argsort(label2[i],descending=True)[:temp_sum].cpu().detach().tolist()

        for j in range(len(list_after_threshold)):
            if(list_after_threshold[j] in argsort_label2):
                tmp_acc_all_after_threshold += 1
        tmp_acc_all_after_threshold /= temp_sum
        acc_all_after_threshold += tmp_acc_all_after_threshold
        pred = outputs[i].cpu().detach().numpy()
        if(pred[label1[i]]>=threshold or np.argmax(pred) == label1[i]):
            acc_after_threshold+=1
    total+=len(label1)

ave_acc_all_after_threshold = acc_all_after_threshold/total
ave_num_after_threshold = num_after_threshold/total
ave_acc_after_threshold = acc_after_threshold/total
print(chosen_model,ave_num_after_threshold,ave_acc_after_threshold)