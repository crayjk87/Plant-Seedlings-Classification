import pandas as pd
import os

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from utils.save_plot import save_bar
from sklearn.model_selection import train_test_split


root_train = './train/'
root_test = './test/'
batch_size = 32     # Because the use of batch 64 will cause 'out of memery', so use 32

i_to_c = {
    0 : "Black-grass",
    1 : "Charlock",
    2 : "Cleavers",
    3 : "Common Chickweed",
    4 : "Common wheat",
    5 : "Fat Hen",
    6 : "Loose Silky-bent",
    7 : "Maize",
    8 : "Scentless Mayweed",
    9 : "Shepherds Purse",
    10 : "Small-flowered Cranesbill",
    11 : "Sugar beet"
}

class seed_train_dataset(Dataset):  
    def __init__(self, dataframe, root, transforms=None):
        self.df = dataframe
        self.root = root
        self.transforms = transforms
    
    def __getitem__(self, index):
        image_path = Image.open(os.path.join(self.root, self.df.iloc[index][1]+ '/' + self.df.iloc[index][2]))
        image = image_path.convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, self.df.iloc[index][0]

    def __len__(self):
        return len(self.df)

class seed_test_dataset(Dataset):
    def __init__(self, root, transforms):
        self.img = os.listdir(root)
        self.root = root
        self.transforms = transforms
    
    def __getitem__(self, index):
        image_path = Image.open(os.path.join(self.root, self.img[index]))
        image = image_path.convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        return image, self.img[index]

    def __len__(self):
        return len(self.img)


def list_dir(path):     # Create the dataframe for dataloader
    if path == './train/':
        data = []
        for i, label in i_to_c.items():
            for img in os.listdir(path + '/' + label):
                data.append([i, label, img])
        df = pd.DataFrame(data, columns=["Index_of_class", "Catagory", "Img"])
        #print(os.path.join(root, df.iloc[0][1]+ '/' + df.iloc[0][2]))
        print(df)
    elif path == './test/':
        df = pd.DataFrame(os.listdir('./test'), columns=['file'])
        df = df.reindex(columns = df.columns.tolist() + ['species'])
        print(df)

    return df

def load_train_data():
    train_data_trans = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.CenterCrop(200),
            transforms.RandomRotation(90), 
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomInvert(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])

    val_data_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    df = list_dir(root_train)
    save_bar(df)

    train_split, val_split = train_test_split(df, random_state=100, stratify=df['Index_of_class'], train_size=0.8)
    train_set = seed_train_dataset(train_split, root_train, train_data_trans)
    val_set = seed_train_dataset(val_split, root_train, val_data_trans)

    # Dataloader of the data
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory = True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, pin_memory = True)

    return train_loader, val_loader

def load_test_data():
    val_data_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    df = list_dir(root_test)
    test_set = seed_test_dataset(root_test, val_data_trans)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

    return test_loader, df

# To transfrom the data index to catagory string
def submission_trans(submission):
    submission = submission.replace({'species':i_to_c})
    return submission


