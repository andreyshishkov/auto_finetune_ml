import torch
from PIL import Image
import os
from transforms import transform_for_gray_scale, val_transforms
import csv

model = torch.load(
    '../model/models/model',
        map_location='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
weights = torch.load(
    '../model/weights/weights',
    map_location='cuda:0' if torch.cuda.is_available() else 'cpu',
    )
model.load_state_dict(weights)
class_names = sorted(os.listdir('../data/images'))

model.eval()
preds = []
samples = os.listdir('Хакатон')
for sample in samples:
    file_path = f'../testing/Хакатон/{sample}'
    idx = sample.remove('.jpg')

    img = Image.open(file_path)
    img = img.convert('RGB')
    img = val_transforms(img)
    img = img.unsqueeze(0)
    prediction = model(img)

    prediction = prediction.argmax()
    prediction = class_names[prediction]

    print(idx, prediction)
    preds.append((idx, prediction))

with open('result.csv', 'w') as file:
    preds.sort(key=lambda x: x[0])
    writer = csv.writer(file, delimiter=';')
    writer.writerows(preds)
