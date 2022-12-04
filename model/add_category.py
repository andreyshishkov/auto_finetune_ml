import torch
from model.train import update_model
import os


def add_category():
    model = torch.load(
        'model/models/model',
        map_location='cuda:0' if torch.cuda.is_available() else 'cpu',
    )

    num_in = model.fc.out_features
    update_model(model, num_in + 1)


if __name__ == '__main__':
    os.chdir('../')