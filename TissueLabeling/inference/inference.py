
import numpy as np
import torch
import csv

from utils import init_fabric, load_brains, crop
from models.segformer import Segformer

NR_OF_CLASSES = 6

fabric = init_fabric()
state = fabric.load("../models/checkpoint.ckpt")

model = Segformer(NR_OF_CLASSES)
model = torch.compile(model)
model.load_state_dict(state["model"])

image_file = 'pac_36_orig.nii.gz'
mask_file = 'pac_36_aseg.nii.gz'
file_path = '/om2/user/matth406/nobrainer_data/data/SharedData/segmentation/freesurfer_asegs/'
brain, mask, _ = load_brains(image_file, mask_file, file_path)

# map classification masks
original_classes, new_classes = [], []
class_mapping = {}
with open('/home/matth406/unsupervised_brain/data/class-mapping.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    # skip header
    next(spamreader, None)
    for row in spamreader:
        original_classes.append(row[1])
        new_classes.append(row[2])
        class_mapping[int(row[1])] = int(row[2])
u, inv = np.unique(mask,return_inverse = True)
mask = np.array([class_mapping[x] if x in class_mapping else 0 for x in u])[inv].reshape(mask.shape)

brain_slice = brain[:,:,125]
mask_slice = mask[:,:,125]
# crop image
brain_slice = crop(brain_slice, 162, 194)
mask_slice = crop(mask_slice, 162, 194)

# normalize
normalization_constants = np.load("/om2/user/matth406/nobrainer_data_norm/data_prepared_medium/normalization_constants.npy")
brain_slice = (brain_slice - normalization_constants[0]) / normalization_constants[1]

brain_slice = torch.from_numpy(brain_slice).to(torch.float32)
brain_slice = brain_slice[None, None]
mask_slice = torch.tensor(mask_slice)[None, None].long()

model.eval()
with torch.no_grad():
    probs = model(brain_slice)
preds = probs.argmax(1)

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm

# color map to get always the same colors for classes
colors = plt.cm.plasma(np.linspace(0, 1, NR_OF_CLASSES))
# new plt cmap
cmap = ListedColormap(colors)
# new plt norm
bounds=np.arange(0,NR_OF_CLASSES+1)
norm = BoundaryNorm(bounds, cmap.N)

# save mask
fig, ax = plt.subplots()
ax.imshow(mask_slice.squeeze(), cmap=cmap, norm=norm)
ax.axis('off')
fig.canvas.draw()
fig.savefig("../images/inference_mask.png")
plt.close()

# save prediction
fig, ax = plt.subplots()
ax.imshow(preds.squeeze(), cmap=cmap, norm=norm)
ax.axis('off')
fig.canvas.draw()
fig.savefig("../images/inference_prediction.png")
plt.close()

