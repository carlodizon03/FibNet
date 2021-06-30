import matplotlib.pyplot as plt
import numpy as np
import PIL 
import torchvision
import torch
def Inference(img,gt_mask,pred_mask):
    img = img.to('cpu')
    pred_mask = pred_mask.to('cpu')
    gt_mask = gt_mask.to('cpu')
    img = torchvision.utils.make_grid(img)
    mask = torchvision.utils.make_grid(gt_mask)
    pred_mask = torchvision.utils.make_grid(pred_mask)

    img = img.detach().numpy().transpose((1,2,0))
    mask = mask.detach().numpy().transpose((1,2,0))
    pred_mask = pred_mask.detach().numpy().transpose((1,2,0))
    z = np.zeros([224,224,1])
    pred_mask = np.concatenate((z,pred_mask), -1)

    fig = plt.figure(figsize = (15,8))
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312)
    ax3 = plt.subplot(313)

    ax1.set_title('Image')
    ax2.set_title('Mask')
    ax3.set_title('Segmentation Mask')

    ax1.imshow(img)
    ax2.imshow(mask)
    ax3.imshow(pred_mask)

    plt.pause(0.001)
    plt.show()

def Pair(img, mask):
    
    img = torchvision.utils.make_grid(img)
    mask = torchvision.utils.make_grid(mask)
    img = img.to('cpu')
    mask = mask.to('cpu')
    print(mask.shape)
    img = img.detach().numpy().transpose((1,2,0))
    mask = mask.detach().numpy().transpose((1,2,0)) 
    print(mask.shape)
    fig = plt.figure(figsize = (15,8))
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.set_title('Image')
    ax2.set_title('Mask')
    ax1.imshow(img)
    ax2.imshow(mask)
    plt.pause(0.001)
    plt.show()


def Image(img, label = None):
    img = img.to('cpu')
    img = torch.transpose(img,1,0)
    img = torchvision.utils.make_grid(img[:10])
    print(img.shape)

    img = img.detach().numpy()
    img = img.transpose((1,2,0))
    print(img.shape)

    if label != None:
        plt.xlabel(label)
    plt.imshow(img)
    plt.show()