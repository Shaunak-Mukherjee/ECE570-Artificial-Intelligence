import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np
import random

def saliency_to_traj(saliency, criterion = 'MoRF'):
    if criterion == 'MoRF':
        return np.array(list(reversed(saliency.flatten().argsort()).detach().cpu().numpy()))
    elif criterion == 'LeRF':
        return np.array(list(saliency.flatten().argsort().detach().cpu().numpy()))

def denorm_trace(x, device = None):
    
    if x.shape[-3] == 3:
        if device == None:
            device = x.device
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # xx = torch.zeros(x.shape).to(device)
        if len(x.shape) == 4:
            xx = x.clone().detach().to(device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(device)
        xx[:, 0, :, :] = xx[:, 0, :, :] * std[0] + mean[0]
        xx[:, 1, :, :] = xx[:, 1, :, :] * std[1] + mean[1]
        xx[:, 2, :, :] = xx[:, 2, :, :] * std[2] + mean[2]
    else:
        if device == None:
            device = x.device
        mean = 0.1307
        std = 0.3081
        # xx = torch.zeros(x.shape).to(device)
        if len(x.shape) == 4:
            xx = x.clone().detach().to(device)
        elif len(x.shape) == 3:
            xx = x[None].clone().detach().to(device)
        xx = xx * std + mean
        
    return xx

def plot_tensor_image(image, save=None):
    toshow = image.detach().clone().cpu()
    toshow = toshow.squeeze()
    if toshow.max() > 1:
        toshow = denorm_trace(toshow).squeeze()
    
    if len(toshow.shape) == 4:
        print("Plotting the entire batch")
        toshow = torchvision.utils.make_grid(toshow,nrow=10)
        figsize = (20,20)
    else:
        print("Plotting the single image")
        figsize = (5,5)
    
    plt.figure(figsize=figsize)
    plt.imshow(toshow.numpy().transpose((1,2,0)))
    plt.axis('off')
    if save is not None:
        print(f"save at {save}")
        plt.savefig(save,bbox_inches='tight')
    plt.show()
    
def show_traj(image, traj, reference = "zero", edge = 7, figsize = (15, 15), save=None):
    channel = image.shape[-3]
    masked = image.detach().clone().view(channel,224,224)
    toshow = [masked[None]]
    for ids in traj:
        masked = masking(masked, reference, ids, edge = edge)
        toshow.append(masked[None])
    toshow = torch.cat(toshow)
    plot_tensor_image(toshow,save=save)

def masking(image, reference, idx = 0, edge = 7):
    '''
    image (torch.Tensor): 3 x 224 x 224
    reference (str): the reference types
    '''
    channel = image.shape[-3]
    grid = edge**2
    patch_size = 224//edge
    
    assert idx >= 0 and idx <grid, 'idx out of range'
    x = idx //edge
    y = idx % edge
    masked = image.detach().clone()
    if reference == 'zero':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = 0
    elif reference == 'random':
        reference_values = normalize(torch.rand(channel,patch_size, patch_size).to(image.device)).squeeze()
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = reference_values
    elif reference == 'mean':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] = image.mean()
    elif reference == 'patch_mean':
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] =\
        image[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size].mean()
    elif reference == 'blur':
        blurrer = transforms.GaussianBlur(kernel_size=(31, 31), sigma=(56, 56))
        blurred_img = blurrer(image)
        masked[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size] =\
            blurred_img[:,x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size]
        
    return masked
    
def deletion_attribution(model, image, label, saliency, options, mode):
    saliency_superpatch = F.interpolate(
        torch.FloatTensor(saliency).view(1,1,saliency.shape[-1],saliency.shape[-1]),
        size = (options.edge, options.edge), mode = 'area')
    traj = saliency_to_traj(saliency_superpatch, mode)
    return traj_masking(model, image, label, traj, options.reference, options.edge)[0]

def traj_masking(model, image, label, traj, reference = 'zero', edge = 7):
    model.eval()
    masked = image.detach().clone()
    with torch.no_grad():
        inputs = [masked[None]]
        for ids in traj:
            masked = masking(masked, reference, ids, edge = edge)
            inputs.append(masked[None])
        inputs = torch.cat(inputs).to(image.device)
        prob = torch.softmax(model(inputs), dim = 1)
        return prob[:, label].detach().cpu().numpy(), inputs



def sample_permutations(d, N):
    samples = []
    for _ in range(N):
        # Create a list of numbers from 1 to d
        perm = list(range(1, d + 1))
        # Shuffle the list in-place
        random.shuffle(perm)
        samples.append(perm)
    return samples

def denorm(tensor: torch.Tensor) -> torch.Tensor:
    
    tensor = tensor.detach().clone()
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def plot_image(image: torch.Tensor) -> None:
    
    fig, ax = plt.subplots(1,1,figsize = (5,5), dpi = 100)

    ax.imshow(denorm(image).permute(1,2,0))
    ax.axis("off")
    plt.show()
    
    
def plot_image_id(idx: int, valset: torch.utils.data.Dataset) -> None:
    
    img, label = valset[idx]
    fig, ax = plt.subplots(1,1,figsize = (5,5), dpi = 100)

    ax.imshow(denorm(img).permute(1,2,0))
    ax.axis("off")
    plt.title(f"Class: {valset.class_list[label.item()]}")
    plt.show()
    
    
def plot_batch_image(images: torch.Tensor) -> None:
    
    bs = len(images)

    grid = torchvision.utils.make_grid(images, nrow = int(np.sqrt(bs)))

    plt.figure(figsize=(6,6),dpi=100)
    plt.imshow(denorm(grid).permute(1,2,0))
    plt.axis('off')
    plt.show()
    


def plot_image_attr(image: torch.Tensor, attr: torch.Tensor, overlap = False) -> None:
    
    if overlap:
        fig, ax = plt.subplots(1,1, figsize = (6,6),dpi = 100)

        ax.imshow(denorm(image[0]).permute(1,2,0))
        ax.imshow(abs(attr[0]).mean(0), cmap = "jet", alpha = 0.3)

        ax.axis("off")
        plt.tight_layout()
        plt.show()
        
    else:
        fig, ax = plt.subplots(1,2, figsize = (12,6),dpi = 100)

        ax[0].imshow(denorm(image[0]).permute(1,2,0))
        ax[1].imshow(abs(attr[0]).mean(0), cmap = "gray")

        ax[0].axis("off")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()
        
    
def plot_batch_image_attr(images: torch.Tensor, attr: torch.Tensor, overlap = False) -> None:
    
    bs = len(attr)
    grid_attr = torchvision.utils.make_grid(attr, nrow = int(np.sqrt(bs)))
    grid_attr = torch.abs(grid_attr).mean(0)

    grid_img = torchvision.utils.make_grid(images, nrow = int(np.sqrt(bs)))

    if overlap:
        fig, ax = plt.subplots(1,1, figsize = (6,6),dpi = 100)

        ax.imshow(denorm(grid_img).permute(1,2,0))
        ax.imshow(grid_attr, cmap = "jet", alpha = 0.5)

        ax.axis("off")
        plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(1,2, figsize = (12,6),dpi = 100)

        ax[0].imshow(denorm(grid_img).permute(1,2,0))
        ax[1].imshow(grid_attr, cmap = "gray")

        ax[0].axis("off")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()
