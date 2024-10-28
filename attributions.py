import torch
from tqdm import tqdm
from skimage.transform import resize
from torch import nn
import numpy as np
from utils import *
import itertools


class GuidedBackprop():

    def __init__(self, model):
        self.model = model
        self.handles = []  # Store handles for the registered hooks
        self.gradients = None  # Will store the computed gradients
        self._register_hooks()  # Register the forward and backward hooks

    def _register_hooks(self):
        
        def forward_hook(module, input, output):
            if isinstance(module, nn.ReLU):
                return torch.clamp(output, min=0)  # Standard ReLU behavior

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0),)  # Only positive gradients

        # Register hooks for all ReLU layers in the model
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # Register both forward and backward hooks
                self.handles.append(module.register_forward_hook(forward_hook))
                self.handles.append(module.register_backward_hook(backward_hook))

    def generate_gradients(self, images, labels):
        
        # Enable gradient computation for the input image
        input_image = images.detach().clone().requires_grad_(True)
        
        # Clear any existing gradients in the model
        self.model.zero_grad()
        
        # Forward pass through the model
        target = self.model(input_image)[range(len(labels)), labels].sum()
        target.backward()
        
        return input_image.grad.data

    def __del__(self):
        for handle in self.handles:
            handle.remove()

def input_gradient(
    model: torch.nn.Module, 
    images: torch.Tensor, 
    labels: torch.Tensor) -> torch.Tensor:
    
    model.zero_grad()
    images = images.detach().clone().requires_grad_(True)

    pred = model(images)
    target = pred[range(len(labels)), labels].sum()

    grad = abs(torch.autograd.grad(
        inputs = images,
        outputs = target,
    )[0].detach().clone())
    
    return grad


def input_x_gradient(
    model: torch.nn.Module, 
    images: torch.Tensor, 
    labels: torch.Tensor) -> torch.Tensor:
    
    
    model.zero_grad()
    images = images.detach().clone().requires_grad_(True)

    pred = model(images)
    target = pred[range(len(labels)), labels].sum()

    grad = torch.autograd.grad(
        inputs = images,
        outputs = target,
    )[0].detach().clone()

    input_x_grad = images*grad
    
    return input_x_grad


def integrated_gradients(
    model: torch.nn.Module, 
    images: torch.Tensor, 
    labels: torch.Tensor,
    M: int = 50,
) -> torch.Tensor:

    model.zero_grad()
    reference = torch.zeros_like(images)

    ig = torch.zeros_like(images)
    for m in tqdm(range(1,M+1)):
        model.zero_grad()
        line = (reference.detach().clone() + (images.detach().clone() - reference.detach().clone()) * m/M).requires_grad_(True)

        pred = model(line)
        target = pred[range(len(labels)), labels].sum()

        grad = torch.autograd.grad(
            inputs = line,
            outputs = target,
        )[0].detach().clone()

        ig += grad/M

    ig = ig*(images - reference)
    ig = ig.detach().clone()
    
    return ig



def grad_cam(
    model: torch.nn.Module, 
    images: torch.Tensor, 
    labels: torch.Tensor,
    M: int = 50,
) -> torch.Tensor:
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output#.detach()
        return hook
    # handle = model.layer3[1].conv2.register_forward_hook(get_activation('feature'))
    handle = model.layer4[1].bn1.register_forward_hook(get_activation('feature'))
    target = model(images.to(device))[range(len(labels)), labels].sum()
    handle.remove()

    grad = torch.autograd.grad(
        inputs = activation['feature'],
        outputs = target,
        create_graph = True
    )[0]

    gradcam = (activation['feature'] * grad.mean(dim = [2,3],keepdims=True)).sum(1).detach().cpu()

    upsampler = torch.nn.Upsample(scale_factor = 32, mode = "bilinear")
    gradcam = upsampler(gradcam.unsqueeze(1))
    return gradcam



class GuidedBackprop:

    def __init__(self, model):
        self.model = model
        self.handles = []  # Store handles for the registered hooks
        self.gradients = None  # Will store the computed gradients
        self._register_hooks()  # Register the forward and backward hooks

    def _register_hooks(self):
        
        def forward_hook(module, input, output):
            if isinstance(module, nn.ReLU):
                return torch.clamp(output, min=0)  # Standard ReLU behavior

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0),)  # Only positive gradients

        # Register hooks for all ReLU layers in the model
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                # Register both forward and backward hooks
                self.handles.append(module.register_forward_hook(forward_hook))
                self.handles.append(module.register_backward_hook(backward_hook))

    def generate_gradients(self, input_image, target_class):
        
        # Enable gradient computation for the input image
        input_image = input_image.detach().clone().requires_grad_(True)
        
        # Clear any existing gradients in the model
        self.model.zero_grad()
        
        # Forward pass through the model
        output = self.model(input_image)
        
        # Create a one-hot encoded tensor for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        
        # Backward pass
        # This computes gradients of the target class output with respect to the input image
        output.backward(gradient=one_hot)
        
        # Return the gradients of the input image
        return input_image.grad.data

    def __del__(self):
        for handle in self.handles:
            handle.remove()
            
            
class RISE(nn.Module):
    def __init__(self, model, input_size, device, gpu_batch=100):
        super(RISE, self).__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch
        self.device = device

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.to(self.device)
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().to(self.device)
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal
    
    
class RISEBatch(RISE):
    def forward(self, x):
        # Apply array of filters to the image
        N = self.N
        B, C, H, W = x.size()
        stack = torch.mul(self.masks.view(N, 1, H, W), x.data.view(B * C, H, W))
        stack = stack.view(B * N, C, H, W)
        stack = stack

        #p = nn.Softmax(dim=1)(model(stack)) in batches
        p = []
        for i in range(0, N*B, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N*B)]))
        p = torch.cat(p)
        CL = p.size(1)
        p = p.view(N, B, CL)
        sal = torch.matmul(p.permute(1, 2, 0), self.masks.view(N, H * W))
        sal = sal.view(B, CL, H, W)
        return sal
    
    
    
def GradCAM(model, images, labels):
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output#.detach()
        return hook
    # handle = model.layer3[1].conv2.register_forward_hook(get_activation('feature'))
    handle = model.layer4[1].bn1.register_forward_hook(get_activation('feature'))
    target = model(images)[range(len(labels)), labels].sum()
    handle.remove()

    grad_for_gradcam = torch.autograd.grad(
        inputs = activation['feature'],
        outputs = target,
        create_graph = True
    )[0]

    gradcam = (activation['feature'] * grad_for_gradcam.mean(dim = [2,3],keepdims=True)).sum(1).detach().cpu()

    upsampler = torch.nn.Upsample(scale_factor = 32, mode = "bilinear")
    gradcam = upsampler(gradcam.unsqueeze(1))
    
    return gradcam
    
    

def SHAP(model, images, labels):
    
    samples = sample_permutations(10,100)
    shap = []

    with torch.no_grad():
        for sample in samples:
            shap.append(torch.zeros_like(input))
            path = torch.zeros_like(input).to(device)
            for idx in sample:

                path_next = path.detach().clone()
                path_next[0,idx-1] = input[0,idx-1]

                shap[-1][0,idx-1] += (mlp(path_next) - mlp(path))[0,0]
                path = path_next.detach().clone()

    shap = torch.cat(shap)
    
    
def func(model, image, label, traj, mode = "LeRF", edge = 7, reference = "zero"):    
    if mode == 'LeRF-MoRF':
        return traj_masking(model, image, label, traj, edge = edge, reference = reference)[0].sum() -\
    traj_masking(model, image, label, reversed(traj), edge = edge, reference = reference)[0].sum()
    
    elif mode == 'MoRF':
        return traj_masking(model, image, label, traj, edge = edge, reference = reference)[0].sum()
    elif mode == 'LeRF':
        return traj_masking(model, image, label, traj, edge = edge, reference = reference)[0].sum()
    else:
        print('WRONG MODE!')
        

def greedy(model, image, label, reference = "zero", edge = 7, mode = 'LeRF'):
    '''
    Parameters:
    - model (torch.nn.Module): the model to be explained
    - image (torch.Tensor): 3 x 224 x 224
    - label (int): the label of the image
    - mode  (str): MoRF (Most Relevant First) or LeRF (Least Relevant First) 
    
    Returns:
    - traj: (list): a list of the deletion trajectoy
    - Prob: (list): the predicted probability of the corresponding class w.r.t. the deletion process
    '''  
    
    channel = image.shape[-3]
    if len(image.shape) == 4:
        if image.shape[0] > 1:
            raise ValueError("only 1 image at a time")

            
    grid = edge**2
    patch_size = 224//edge
    
    all_idx = set(range(grid))
    traj = []
    Prob = []
    model.eval()
    with torch.no_grad():
        new_image = image.detach().clone()
        pred = model(new_image[None])[:, label].item()
        prob = torch.softmax(model(new_image[None]), dim = 1)[:, label].item()
        Prob.append(prob)
        while len(all_idx) > 1: 
            masked = []
            ids = []
            for i in all_idx:
                masked.append(masking(new_image, reference, i, edge = edge))
                ids.append(i)
            masked = torch.stack(masked)
            prob = torch.softmax(model(masked), dim = 1)[:, label]
            
            if mode == 'LeRF':
                argmax = prob.argmax().item()
                traj.append(ids[argmax])
                all_idx.remove(traj[-1])
                Prob.append(prob.max().item())
                new_image = masked[prob.argmax().item()].detach().clone()
            elif mode == 'MoRF':
                traj.append(ids[prob.argmin().item()])
                all_idx.remove(traj[-1])
                Prob.append(prob.min().item())
                new_image = masked[prob.argmin().item()].detach().clone()
            else:
                raise ValueError("mode has to be either 'MoRF' or 'LeRF'")  
             
                
#         masked = torch.zeros(1,channel,224,224).to(image.device)
        if reference == 'zero':
            masked = torch.zeros(1,channel,224,224).to(image.device)
        elif reference == 'mean':
            masked = torch.ones(1,channel,224,224).to(image.device)*image.mean()
        elif reference == 'patch_mean':
            masked = F.interpolate(
                F.interpolate(image.view(1,3,224,224), size=(7,7), mode='area'),
                size=(224,224), mode='area')
        elif reference == 'blur':
            blurrer = transforms.GaussianBlur(kernel_size=(31, 31), sigma=(56, 56))
            masked = blurrer(image)[None]
            
        
        prob = torch.softmax(model(masked), dim = 1)[:, label].item()
        Prob.append(prob)
        traj.append(all_idx.pop())
        
    return np.array(traj), np.array(Prob)



def simulated_annealing(
    model,
    image,
    label,
    candidates,
    reference = "zero",
    mode = "LeRF",
    edge = 7,
    T = 0.05, 
    eta = 0.998, 
    threshold = 0.,
    n_iteration = 1000,
    verbose = False,
    random_init = False,
):
    
    if random_init:
        traj_init = np.random.permutation(edge**2)
    else:
        traj_init, _ = greedy(
            model, 
            image, 
            label, 
            reference = reference, 
            edge = edge, 
            mode = "LeRF")
    
    # Initialization
    sol_0 = np.array(traj_init).copy()
    ct_array = []
    best = sol_0.copy()

    score_best = -func(model, image, label, best, mode, edge = edge)
    score_0 = -func(model, image, label, sol_0, mode, edge = edge)

    print('start: ', score_best)
    for t in range(1, n_iteration + 1):
        temp = sol_0.copy()
        sol_1 = nC2_swap(sol_0, candidates)

        # energy
        score_1 = -func(model, image, label, sol_1, mode, edge = edge)
        delta_t = score_1 - score_0
        
        if delta_t < 0:
            sol_0 = sol_1.copy()
            score_0 = score_1
            ct_array.append(1)
        else:
            p = np.exp(-delta_t/T)
            if np.random.uniform(0, 1) < p:
                sol_0 = sol_1.copy()
                score_0 = score_1
                ct_array.append(1)
            else:
                ct_array.append(0)

        if score_best > score_0:
            best = sol_0.copy()
            score_best = score_0
            print('t: {}, score: {:.4f}, T: {:.4f} kendall onestep: {:.4f}, kendall overall: {:.4f}'.format(
                t, score_best, T, kendall_distance(temp, sol_1),
                kendall_distance(sol_0.argsort(), np.array(traj_init).argsort())
            ))
        elif ct_array[-1] == 1 and verbose:
            print('t: {}, score: {:.4f}/{:.4f}, T: {:.4f} kendall onestep: {:.4f}, kendall overall: {:.4f}'.format(
                t, score_0, score_best, T, kendall_distance(temp, sol_1),
                kendall_distance(sol_0.argsort(), np.array(traj_init).argsort())
            ))

        if T > threshold:
            T *= eta

    return best


def nC2_swap(sol, candidates):
    traj = sol.copy()
    positions = list(candidates[np.random.randint(0, len(candidates))])
    traj[positions] = np.flip(traj[positions])
    return traj

def kendall_distance(t1, t2):
    n = len(t1)
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    ndisordered = np.logical_or(np.logical_and(t1[i] < t1[j], t2[i] > t2[j]), 
                                np.logical_and(t1[i] > t1[j], t2[i] < t2[j])).sum()
    return ndisordered/2