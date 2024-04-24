import os
import glob
import json
from collections import ChainMap

DATA_DIR = '/om2/scratch/Mon/sabeen/kwyk_slice_split_250'

percent_background_files = sorted(glob.glob(os.path.join(DATA_DIR,'percent_backgrounds_*_*.json')))

all_percent_backgrounds = dict()
for filename in percent_background_files:
    print(filename)
    with open(filename) as f:
        percent_backgrounds = json.load(f)
    all_percent_backgrounds.update(dict(percent_backgrounds))

print('Saving all_percent_backgrounds...')
with open(os.path.join(DATA_DIR, "percent_backgrounds.json"), "w") as f:
    json.dump(all_percent_backgrounds, f)
print('Done!')