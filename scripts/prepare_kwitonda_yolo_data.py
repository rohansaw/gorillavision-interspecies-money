from pathlib import Path
from glob import glob
import random

random.seed(0)
train_file = '/data/kwitonda_manually_labeled/train.txt'

with open(train_file, 'r')  as f:
    lines = f.readlines()

lines = [line.strip() for line in lines]

lines = [line.replace('data/obj_train_data', '/data/kwitonda_manually_labeled/obj_train_data').replace(".jpg", ".JPG") for line in lines]

# randomize
random.shuffle(lines)

# split

split = 0.8
split_idx = int(len(lines) * split)

train_lines = lines[:split_idx]

train_file = '/data/kwitonda_manually_labeled/train_split.txt'
val_file = '/data/kwitonda_manually_labeled/val_split.txt'


with open(train_file, 'w') as f:
    for line in train_lines:
        f.write(line + '\n')

with open(val_file, 'w') as f:
    for line in lines[split_idx:]:
        f.write(line + '\n')


