__author__ = 'Tanuj'

__author__ = 'Tanuj'

import skimage.io as io
from os import walk
import re

FDDB_path = "./data/FDDB-folds/"

data = []

f = []
for (dirpath, dirnames, filenames) in walk(FDDB_path):
    f.extend(filenames)
    break

regex = re.compile(".*(ellipseList).*")
f = [m.group(0) for l in f for m in [regex.search(l)] if m]

file = open(FDDB_path + f[0], 'rb')

n = -1
mode = None

for line in file:
    if(regex.search(line) != None):
        mode = "image"
    regex = re.compile(".*(img).*")

