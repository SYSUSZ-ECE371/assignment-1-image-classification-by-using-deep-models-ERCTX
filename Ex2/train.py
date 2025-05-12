import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
import os
import time
import copy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set data directory
data_dir = 'flower_dataset'

# Data augmentation and normalization
data_transforms = transforms.Compose([
    # GRADED FUNCTION: Add five data augmentation methods, Normalizating and Tranform to tensor
    ### START SOLUTION HERE ###
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ### END SOLUTION HERE ###
])

# Load dataset
full_dataset = datasets.ImageFolder(data_dir, data_transforms)

# Split into train and validation
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

dataloaders = {'train': train_loader, 'val': val_loader}
dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
class_names = full_dataset.classes

# Load pre-trained model and modify
model = models.resnet18(pretrained=True)

# GRADED FUNCTION: Modify the last fully connected layer of model
### START SOLUTION HERE ###
model.fc = nn.Linear(model.fc.in_features, len(class_names))
### END SOLUTION HERE ###

# GRADED FUNCTION: Define the loss function
### START SOLUTION HERE ###
criterion = nn.CrossEntropyLoss()
### END SOLUTION HERE ###

# GRADED FUNCTION: Define the optimizer
### START SOLUTION HERE ###
optimizer = optim.Adam(model.parameters())
### END SOLUTION HERE ###

# Learning rate scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


# Train function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # 记录loss和acc
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        # GRADED FUNCTION: Backward pass and optimization
                        ### START SOLUTION HERE ###
                        loss.backward()
                        optimizer.step()
                        ### END SOLUTION HERE ###

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存历史
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            # 保存最好的模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                save_dir = 'ckpt'
                os.makedirs(save_dir, exist_ok=True)
                
                # GRADED FUNCTION: Save the best model
                ### START SOLUTION HERE ###
                torch.save(best_model_wts, os.path.join(save_dir, 'best_model.pth'))
                ### END SOLUTION HERE ###

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)

    # 返回模型以及训练历史
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# Train the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model, train_loss, val_loss, train_acc, val_acc = train_model(model, criterion, optimizer, scheduler, num_epochs=400)

# 绘制Loss和Acc曲线
epochs = range(1, len(train_loss)+1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

# ==================
# 测试集推理并绘制混淆矩阵
# ==================

# 加载最好的模型
model.load_state_dict(torch.load('ckpt/best_model.pth'))
model.eval()

# 推理测试集
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10,10))
disp.plot(ax=ax, cmap='Blues')
plt.title('Confusion Matrix on Validation Set')
plt.savefig('confusion_matrix.png')
plt.show()
