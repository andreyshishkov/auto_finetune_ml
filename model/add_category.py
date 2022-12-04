import torch
from train import update_model
import torchvision
import os

train_dir = 'train'
val_dir = 'val'
class_names = sorted(os.listdir('../data/images'))
os.chdir('../')


def add_category():
    model = torch.load(
        'model/models/model',
        map_location='cuda:0' if torch.cuda.is_available() else 'cpu',
    )

    num_in = model.fc.out_features
    update_model(model, num_in + 1)
