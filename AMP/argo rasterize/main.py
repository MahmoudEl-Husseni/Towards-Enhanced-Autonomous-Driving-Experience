# ------------------------------------------
# This file is part of argo rasterize.
# File: main.py

# Autor: Mahmoud ElHusseni
# Created on 2024/02/01.
# Github: https://github.com/MahmoudEl-Husseni
# Email: mahmoud.a.elhusseni@gmail.com
# ------------------------------------------

import os 
from render import render
from config import *

if __name__=='__main__' : 
    typs = ['train', 'val', 'test']
    
    for typ in typs: 
        print(f'Processing {typ} scenes ...')
        render(os.path.join(RAW_DIR, typ), os.path.join(OUT_DIR, typ))

    print('Done!')
