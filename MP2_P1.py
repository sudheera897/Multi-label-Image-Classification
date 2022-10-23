#!/usr/bin/env python
# coding: utf-8

# # Assignment 3 : Multi-label Image Classification

# In[1]:


import os
import numpy as np
import torch
import torch.nn as nn
import torchvision

from torchvision import transforms
from sklearn.metrics import average_precision_score
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from kaggle_submission import output_submission_csv
from classifier import SimpleClassifier, Classifier#, AlexNet
from voc_dataloader import VocDataset, VOC_CLASSES

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In this assignment, you train a classifier to do multi-label classificaton on the PASCAL VOC 2007 dataset. The dataset has 20 different class which can appear in any given image. Your classifier will predict whether each class appears in an image. This task is slightly different from exclusive multiclass classification like the ImageNet competition where only a single most appropriate class is predicted for an image.

# ## Reading Pascal Data

# ### Loading Training Data

# In the following cell we will load the training data and also apply some transforms to the data. Feel free to apply more [transforms](https://pytorch.org/docs/stable/torchvision/transforms.html) for data augmentation which can lead to better performance. 

# In[2]:


# Transforms applied to the training data
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std= [0.229, 0.224, 0.225])

train_transform = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            normalize
        ])


# In[3]:


ds_train = VocDataset('/Users/sudheeramaddila/Documents/My Courses/CS747 Deep Learning/Assignment 2/Assignment2_p1/dataset/VOCdevkit_train/VOC2007','train',train_transform)


# ### Loading Validation Data

# We will load the test data for the PASCAL VOC 2007 dataset. Do __NOT__ add data augmentation transforms to validation data.

# In[4]:


# Transforms applied to the testing data
test_transform = transforms.Compose([
            transforms.Resize(227),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            normalize,
        ])


# In[5]:


ds_val = VocDataset('/Users/sudheeramaddila/Documents/My Courses/CS747 Deep Learning/Assignment 2/Assignment2_p1/dataset/VOCdevkit_train/VOC2007','val',test_transform)


# ### Visualizing the Data
# 
# PASCAL VOC has bounding box annotations in addition to class labels. Use the following code to visualize some random examples and corresponding annotations from the train set. 

# In[6]:


for i in range(5):
    idx = np.random.randint(0, len(ds_train.names)+1)
    _imgpath = os.path.join('/Users/sudheeramaddila/Documents/My Courses/CS747 Deep Learning/Assignment 2/Assignment2_p1/dataset/VOCdevkit_train/VOC2007', 'JPEGImages', ds_train.names[idx]+'.jpg')
    
    img = Image.open(_imgpath).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for j in range(len(ds_train.box_indices[idx])):
        obj = ds_train.box_indices[idx][j]
        draw.rectangle(list(obj), outline=(255,0,0))
        draw.text(list(obj[0:2]), ds_train.classes[ds_train.label_order[idx][j]], fill=(0,255,0))
    
    plt.figure(figsize = (10,10))
    plt.imshow(np.array(img))
    


# # Classification

# In[7]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[8]:


train_loader = torch.utils.data.DataLoader(dataset=ds_train,
                                               batch_size=50, 
                                               shuffle=True,
                                               num_workers=1)


# In[9]:


val_loader = torch.utils.data.DataLoader(dataset=ds_val,
                                               batch_size=50, 
                                               shuffle=True,
                                               num_workers=1)


# In[10]:


def train_classifier(train_loader, classifier, criterion, optimizer):
    
    classifier.train()
    loss_ = 0.0
    losses = []
    
    for i, (images, labels, _) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = classifier(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        
    return torch.stack(losses).mean().item()


# In[11]:


def test_classifier(test_loader, classifier, criterion, print_ind_classes=True, print_total=True):
    
    classifier.eval()
    losses = []
    
    with torch.no_grad():
        y_true = np.zeros((0,21))
        y_score = np.zeros((0,21))
        
        for i, (images, labels, _) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            logits = classifier(images)
            
            y_true = np.concatenate((y_true, labels.cpu().numpy()), axis=0)
            y_score = np.concatenate((y_score, logits.cpu().numpy()), axis=0)
            
            loss = criterion(logits, labels)
            losses.append(loss.item())
        
        aps = []
       
        # ignore first class which is background
        for i in range(1, y_true.shape[1]):
            ap = average_precision_score(y_true[:, i], y_score[:, i])
            
            if print_ind_classes:
                print('-------  Class: {:<12}     AP: {:>8.4f}  -------'.format(VOC_CLASSES[i], ap))
            aps.append(ap)
        
        mAP = np.mean(aps)
        test_loss = np.mean(losses)
        
        if print_total:
            print('mAP: {0:.4f}'.format(mAP))
            print('Avg loss: {}'.format(test_loss))
        
    return mAP, test_loss, aps


# In[12]:


# plot functions

def plot_losses(train, val, test_frequency, num_epochs):
    plt.plot(train, label="train")
    indices = [i for i in range(num_epochs) if ((i+1)%test_frequency == 0 or i ==0)]
    plt.plot(indices, val, label="val")
    plt.title("Loss Plot")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    
def plot_mAP(train, val, test_frequency, num_epochs):
    indices = [i for i in range(num_epochs) if ((i+1)%test_frequency == 0 or i ==0)]
    plt.plot(indices, train, label="train")
    plt.plot(indices, val, label="val")
    plt.title("mAP Plot")
    plt.ylabel("mAP")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()
    


# ## Modifying the network 
# 
# The network you are given as is will allow you to reach around 0.15-0.2 mAP. To meet the benchmark for this assignment you will need to improve the network. There are a variety of different approaches you should try:
# 
# * Network architecture changes
#     * Number of layers: try adding layers to make your network deeper
#     * Batch normalization: adding batch norm between layers will likely give you a significant performance increase
#     * Residual connections: as you increase the depth of your network, you will find that having residual connections like those in ResNet architectures will be helpful
# * Optimizer: Instead of plain SGD, you may want to add a learning rate schedule, add momentum, or use one of the other optimizers you have learned about like Adam. Check the `torch.optim` package for other optimizers
# * Data augmentation: You should use the `torchvision.transforms` module to try adding random resized crops and horizontal flips of the input data. Check `transforms.RandomResizedCrop` and `transforms.RandomHorizontalFlip` for this
# * Epochs: Once you have found a generally good hyperparameter setting try training for more epochs
# * Loss function: You might want to add weighting to the `MultiLabelSoftMarginLoss` for classes that are less well represented or experiment with a different loss function
# 
# 

# In[13]:


def train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency=5):
    
    train_losses = []
    train_mAPs = []
    val_losses = []
    val_mAPs = []

    for epoch in range(1,num_epochs+1):
        print("Starting epoch number " + str(epoch))
        
        train_loss = train_classifier(train_loader, classifier, criterion, optimizer)
        train_losses.append(train_loss)
        print("Loss for Training on Epoch " +str(epoch) + " is "+ str(train_loss))
        
        if(epoch%test_frequency==0 or epoch==1):
            mAP_train, _, _ = test_classifier(train_loader, classifier, criterion, False, False)
            train_mAPs.append(mAP_train)
            mAP_val, val_loss, _ = test_classifier(val_loader, classifier, criterion)
            
            print('Evaluating classifier')
            print("Mean Precision Score for Testing on Epoch " +str(epoch) + " is "+ str(mAP_val))
            
            val_losses.append(val_loss)
            val_mAPs.append(mAP_val)
    
    return classifier, train_losses, val_losses, train_mAPs, val_mAPs


# In[14]:


# classifier = Classifier().to(device)
# You can can use this function to reload a network you have already saved previously
#classifier.load_state_dict(torch.load('voc_classifier.pth'))


classifier = SimpleClassifier().to(device)
# You can can use this function to reload a network you have already saved previously
#classifier.load_state_dict(torch.load('voc_classifier.pth'))


# In[15]:


criterion = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)


# In[16]:


# Training the Classifier
num_epochs = 25
test_frequency = 5

classifier, train_losses, val_losses, train_mAPs, val_mAPs = train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency)


# In[17]:


# Compare train and validation metrics
plot_losses(train_losses, val_losses, test_frequency, num_epochs)

plot_mAP(train_mAPs, val_mAPs, test_frequency, num_epochs)


# In[18]:


# Save the clssifier network
# Suggestion: you can save checkpoints of your network during training and reload them later

torch.save(classifier.state_dict(), './voc_classifier.pth')


# # Evaluate on test set
# 
# 

# In[19]:


ds_test = VocDataset('/Users/sudheeramaddila/Documents/My Courses/CS747 Deep Learning/Assignment 2/Assignment2_p1/dataset/VOCdevkit_test/VOC2007','test', test_transform)

test_loader = torch.utils.data.DataLoader(dataset=ds_test,
                                               batch_size=50, 
                                               shuffle=False,
                                               num_workers=1)

mAP_test, test_loss, test_aps = test_classifier(test_loader, classifier, criterion)


# In[20]:


output_submission_csv('my_solution.csv', test_aps)


# In[ ]:





# <h1>AlexNet</h1>

# <h3>Running the classification with AlexNet as the baseline</h3>

# In[21]:


num_epochs = 25
test_frequency = 5

# Change classifier to AlexNet
classifier = torchvision.models.alexnet(pretrained=False)
classifier.classifier._modules['6'] = nn.Linear(4096, 21)   
classifier = classifier.to(device)

criterion = nn.MultiLabelSoftMarginLoss()

optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)


# In[22]:


classifier, train_losses, val_losses, train_mAPs, val_mAPs = train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency)


# In[23]:


plot_losses(train_losses, val_losses, test_frequency, num_epochs)

plot_mAP(train_mAPs, val_mAPs, test_frequency, num_epochs)


# <h1>Pretrained AlexNet</h1>

# In[25]:


num_epochs = 25
test_frequency = 5

# Load Pretrained AlexNet
classifier = torchvision.models.alexnet(pretrained=True)
classifier.classifier._modules['6'] = nn.Linear(4096, 21)   
classifier = classifier.to(device)

optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)


# In[30]:


classifier, train_losses, val_losses, train_mAPs, val_mAPs = train(classifier, num_epochs, train_loader, val_loader, criterion, optimizer, test_frequency)


# In[31]:


plot_losses(train_losses, val_losses, test_frequency, num_epochs)

plot_mAP(train_mAPs, val_mAPs, test_frequency, num_epochs)


# In[32]:


mAP_test, test_loss, test_aps = test_classifier(test_loader, classifier, criterion)

print("Test mAP: ", mAP_test)


# In[ ]:




