
#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# Imports
import torch
import random
from IPython.display import clear_output
from glob import glob
import pandas as pd
cuda = torch.cuda.is_available()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
import os
import pydicom, numpy as np
from skimage.transform import resize
from imgaug import augmenters as iaa
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

sys.path.append("../semi-supervised-pytorch-master/semi-supervised") # path to models
det_class_path = '../Kaggle/all/stage_2_detailed_class_info.csv' # class info
bbox_path = '../Kaggle/all/stage_2_train_labels.csv' # labels
dicom_dir = '../Kaggle/all/stage_2_train_images/' # train images

det_class_df = pd.read_csv(det_class_path)
print(det_class_df.shape[0], 'class infos loaded')
print(det_class_df['patientId'].value_counts().shape[0], 'patient cases')

# Some useful functions
def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def batch(iterable, n=1):
    """Return a batch from the iterable"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# create training dataset with labelled and unlabelled images
image_df = pd.DataFrame({'path': glob(os.path.join(dicom_dir, '*.dcm'))})
image_df['patientId'] = image_df['path'].map(lambda x: os.path.splitext(os.path.basename(x))[0])


from models import AuxiliaryDeepGenerativeModel
image_resize = 225
y_dim = 2
z_dim = 128
a_dim = 128
h_dim = [2048, 1024, 512, 256]

model = AuxiliaryDeepGenerativeModel([image_resize*image_resize, y_dim, z_dim, a_dim, h_dim])
model
if cuda: model = model.cuda()

# training/validation slit
validation = 0.1 
# image resize (the bigger the better, except for computational power ...)
image_resize = 225 
# number of unlabelled images in the training dataset
labelled_images = 1000
# number of labelled ones (the rest of the training dataset)
unlabelled_images = int(image_df.shape[0]*(1-validation)-labelled_images) 
# number of validation images
validation_images = int(validation*image_df.shape[0])
# Some list for the dataset probably I should use something faster ...
labelled = []
label = []
unlabelled = []
# We don't need the same subject repated when multiple bounding boxes occur
det_class_df.drop_duplicates()
allLabel = pd.get_dummies(pd.Series(list(det_class_df['class']))).values
label0Count = 0
label1Count = 0
labelIndex = []
finishLabelling = False

# Prepare training dataset
i = 0
done = 0
while(not finishLabelling):
    if (allLabel[i][0] == 1 or allLabel[i][1] == 1) and label0Count < labelled_images/2:
        done += 1
        label0Count += 1
        labelIndex.append(i)
        k = np.where(image_df['patientId'] == det_class_df['patientId'][i])
        labelled.append(resize(pydicom.read_file(image_df['path'].values[k[0]][0]).pixel_array/255, 
                            (image_resize,image_resize), anti_aliasing=True, mode='constant'))
        label.append([1, 0])
    elif allLabel[i][2] == 1 and label1Count < labelled_images/2:
        done += 1
        label1Count += 1
        labelIndex.append(i)
        k = np.where(image_df['patientId'] == det_class_df['patientId'][i])
        labelled.append(resize(pydicom.read_file(image_df['path'].values[k[0]][0]).pixel_array/255, 
                            (image_resize,image_resize), anti_aliasing=True, mode='constant'))
        label.append([0, 1])
    if label0Count == labelled_images/2 and label1Count == labelled_images/2:
        finishLabelling = True
    i += 1
    if done % 1000 == 0:
        print(str(done) + ' labelled images out of ' + str(labelled_images) + ' done')

print(str(labelled_images) + ' training images labelled loaded')

done = 0
for i in range(labelled_images + unlabelled_images):
    if i not in labelIndex:
        done += 1
        labelIndex.append(i)
        k = np.where(image_df['patientId'] == det_class_df['patientId'][i])
        unlabelled.append(resize(pydicom.read_file(image_df['path'].values[k[0]][0]).pixel_array/255,
                                (image_resize,image_resize), anti_aliasing=True, mode='constant'))
        if done % 1000 == 0 and done != 0:
            print(str(done) + ' unlabelled images out of ' + str(unlabelled_images) + ' done')

print(str(unlabelled_images) + ' training images unlabelled loaded')

# Prepare validation dataset
labelled_val = []
label_val = []
done = 0
for i in range(image_df.shape[0]):
    if i not in labelIndex:
        done += 1
        if allLabel[i][0] == 1 or allLabel[i][1] == 1:
            label_val.append([1, 0])
        elif allLabel[i][2] == 1:
            label_val.append([0, 1])
        k = np.where(image_df['patientId'] == det_class_df['patientId'][i])
        labelled_val.append(resize(pydicom.read_file(image_df['path'].values[k[0]][0]).pixel_array/255, 
                        (image_resize,image_resize), anti_aliasing=True, mode='constant').ravel())
        if done % 1000 == 0:
            print(str(done) + ' images out of ' + str(validation_images) + ' done')

print('Validation images loaded')

trainNbr = np.sum(label, axis=0)
valNbr = np.sum(label_val, axis=0)

print('Summary:')

print('Training images: ' + str(labelled_images + unlabelled_images))
print('Labelled: ' + str(labelled_images) + ', Unlabelled: ' + str(unlabelled_images))
print('Labels: NotNormal ' + str(trainNbr[0]) + ', Normal ' + str(trainNbr[1]))

print('Validation images: ' + str(validation_images))
print('Labels: NotNormal ' + str(valNbr[0]) + ', Normal ' + str(valNbr[1]))


from itertools import cycle
from inference import SVI, DeterministicWarmup, log_gaussian

# We will need to use warm-up in order to achieve good performance.
# Over 200 calls to SVI we change the autoencoder from
# deterministic to stochastic.
def log_gauss(x, mu, log_var):
    return -log_gaussian(x, mu, log_var)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
beta = DeterministicWarmup(n=100)
beta_constant = 1
alpha = beta_constant * (len(unlabelled) + len(labelled)) / len(labelled)
elbo = SVI(model, likelihood=log_gauss, beta=beta)

from torch.autograd import Variable
n_epochs = 200
batchSize= 15
# Some variables for plotting losses
accuracyTrain = []
accuracyVal = []
LTrain = []
LVal = []
UTrain = []
UVal = []
classTrain = []
classVal = []
JAlphaTrain = []
JAlphaVal = []

image_augmenter = iaa.SomeOf((0, None),[iaa.Fliplr(0.5),
                                        iaa.Affine(scale=(0.8, 1.2),
                                        translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                                        rotate=(-15, 15))
                                        ],random_order=True,)

for epoch in range(n_epochs):
    model.train()
    total_L_train, total_U_train, total_classification_loss_train, total_loss_train, accuracy_train = (0, 0, 0, 0, 0)
    total_L_val, total_U_val, total_classification_loss_val, total_loss_val, accuracy_val = (0, 0, 0, 0, 0)
    m_train, m_val = (0, 0)
    latent = []
    y_pred = []
    y_true = []
    l = 0
    # Shuffle the data every epoch (labelled and label should keep the same index ordering!)
    z = list(zip(labelled, label))
    random.shuffle(z)
    random.shuffle(unlabelled)
    labelled, label = zip(*z)

    for x, y, u in zip(cycle(batch(labelled, batchSize)), cycle(batch(label, batchSize)), (batch(unlabelled, batchSize))):
        m_train+=1
        x = image_augmenter.augment_images(x)
        u = image_augmenter.augment_images(u)

        # Wrap in variables
        x, y, u = torch.from_numpy(np.asarray(x).reshape(-1, image_resize*image_resize)), torch.Tensor(y), torch.from_numpy(np.asarray(u).reshape(-1, image_resize*image_resize))
        x, y, u = x.type(torch.FloatTensor), y.type(torch.FloatTensor), u.type(torch.FloatTensor)

        if cuda:
            # They need to be on the same device and be synchronized.
            x, y = x.cuda(device=0), y.cuda(device=0)
            u = u.cuda(device=0)

        L, z = elbo(x, y)
        U, _ = elbo(u)
        if l < 2000:
            latent.append(z.cpu().detach().numpy())
            l = l + 1
            y_true.append(y.cpu().detach().numpy())


        # Add auxiliary classification loss q(y|x)
        logits = model.classify(x)

        # Regular cross entropy
        classication_loss = - torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
        J_alpha_train = - L + alpha * classication_loss - U

        J_alpha_train.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_L_train -= L.item()
        total_U_train -= U.item()
        total_classification_loss_train += classication_loss.item()
        total_loss_train += J_alpha_train.item()
        accuracy_train += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())
        
    model.eval()
    for x, y in zip(batch(labelled_val, batchSize), batch(label_val, batchSize)):
        m_val+=1

        x, y = torch.from_numpy(np.asarray(x).reshape(-1, image_resize*image_resize)), torch.Tensor(y)
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)

        if cuda:
            x, y = x.cuda(device=0), y.cuda(device=0)

        L, _ = elbo(x, y)
        U, _ = elbo(x)

        logits = model.classify(x)
        y_pred.append(torch.max(logits, 1)[1].cpu().detach().numpy())
        classication_loss = - torch.sum(y * torch.log(logits + 1e-8), dim=1).mean()
        J_alpha_val = - L + alpha * classication_loss - U
            
        total_L_val -= L.item()
        total_U_val -= U.item()
        total_classification_loss_val += classication_loss.item()
        total_loss_val += J_alpha_val.item()
        accuracy_val += torch.mean((torch.max(logits, 1)[1].data == torch.max(y, 1)[1].data).float())

    print("Epoch: {}".format(epoch+1))
    print("[Train]\t\t L: {:.2f}, U: {:.2f}, class: {:.2f}, J_a: {:.2f}, accuracy: {:.2f}".format(total_L_train / m_train, total_U_train / m_train, total_classification_loss_train / m_train, total_loss_train / m_train, accuracy_train / m_train))
    print("[Validation]\t L: {:.2f}, U: {:.2f}, class: {:.2f}, J_a: {:.2f}, accuracy: {:.2f}".format(total_L_val / m_val, total_U_val / m_val, total_classification_loss_val / m_val, total_loss_val / m_val, accuracy_val / m_val))

    accuracyTrain.append(accuracy_train / m_train)
    accuracyVal.append(accuracy_val / m_val)
    LTrain.append(total_L_train / m_train)
    LVal.append(total_L_val / m_val)
    UTrain.append(total_U_train / m_train)
    UVal.append(total_U_val / m_val)
    classTrain.append(total_classification_loss_train / m_train)
    classVal.append(total_classification_loss_val / m_val)
    JAlphaTrain.append(total_loss_train / m_train)
    JAlphaVal.append(total_loss_val / m_val)

classes = np.argmax(label_val, axis=1)
y_pred = np.concatenate( y_pred, axis=0 )
conf_matrix = confusion_matrix(classes, np.vstack(y_pred[0:classes.shape[0]]))

PATHMODEL = '../Models/beta2'
PATHFIGURE = '../Figure/training2.npz'
torch.save(model.state_dict(), PATHMODEL)
np.savez(PATHFIGURE,accuracyTrain=accuracyTrain,
accuracyVal=accuracyVal,
classTrain=classTrain,
classVal=classVal,
LTrain=LTrain,
LVal=LVal,
UTrain=UTrain,
UVal=UVal,
JAlphaTrain=JAlphaTrain,
JAlphaVal=JAlphaVal,
conf_matrix=conf_matrix,
latent=latent,
classes=classes,
y_true=y_true)
