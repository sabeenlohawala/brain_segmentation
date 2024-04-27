import glob
import os
import json

slice_dir = '/om2/scratch/Mon/sabeen/kwyk_slice_split_250'
with open(f'{slice_dir}/percent_backgrounds.json') as f:
    percent_backgrounds = json.load(f)

under_80 = [key for key,val in percent_backgrounds.items() if val < 0.8]
train_80 = [item for item in under_80 if 'train' in item]
val_80 = [item for item in under_80 if 'validation' in item]
test_80 = [item for item in under_80 if 'test' in item]

train_feature_80 = [item.split('/')[-1] for item in train_80 if 'features' in item]
train_label_80 = [item.split('/')[-1] for item in train_80 if 'labels' in item]
val_feature_80 = [item.split('/')[-1] for item in val_80 if 'features' in item]
val_label_80 = [item.split('/')[-1] for item in val_80 if 'labels' in item]
test_feature_80 = [item.split('/')[-1] for item in test_80 if 'features' in item]
test_label_80 = [item.split('/')[-1] for item in test_80 if 'labels' in item]

with open(f'{slice_dir}/train/features/files_to_copy.txt','w') as f:
    f.write('\n'.join(train_feature_80))
with open(f'{slice_dir}/train/labels/files_to_copy.txt','w') as f:
    f.write('\n'.join(train_label_80))
with open(f'{slice_dir}/validation/features/files_to_copy.txt','w') as f:
    f.write('\n'.join(val_feature_80))
with open(f'{slice_dir}/validation/labels/files_to_copy.txt','w') as f:
    f.write('\n'.join(val_label_80))
with open(f'{slice_dir}/test/features/files_to_copy.txt','w') as f:
    f.write('\n'.join(test_feature_80))
with open(f'{slice_dir}/test/labels/files_to_copy.txt','w') as f:
    f.write('\n'.join(test_label_80))
