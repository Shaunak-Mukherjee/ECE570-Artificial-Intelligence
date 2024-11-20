import os
import torch
from PIL import Image
from torchvision import transforms

def get_batch_data(valset, ids = None):
    if ids is None:
        ids = [i for i in range(16)]
        
    images, labels = [], []
    for i in ids:
        image, label = valset[i]
        images.append(image)
        labels.append(label)
    return torch.stack(images), torch.stack(labels)

class ILSVRC2012_val(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, class_path):
        self.data_path = data_path
        self.label_path = label_path
        self.class_path = class_path
        
        
        with open("../data/ILSVRC_classes.txt", "r") as f:
            self.class_list = eval(f.read())
        self.class_list = {int(key): value for key, value in self.class_list.items()}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229, 0.224, 0.225])
        ])
        with open(self.label_path, "r") as labels:
            self.labels = labels.readlines()
        for i in range(len(self.labels)):
            self.labels[i] = int(self.labels[i][29:-1])
    
    def __getitem__(self, index):
        
        name = 'ILSVRC2012_val_000' + str(index+1).zfill(5) + '.JPEG'
        img_name = os.path.join(self.data_path, name)
        image = Image.open(img_name).convert('RGB')
        
        label = torch.tensor(self.labels[index])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def __len__(self):
        return 50000
