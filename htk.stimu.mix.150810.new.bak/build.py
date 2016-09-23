#! /usr/bin/env python3

import sys
import os
import re

cfg = None
if len(sys.argv) > 1:
    cfg = sys.argv[1]

if cfg == 'dbg':
    os.system('make clean;rm HTKTools.64bit/HNTrainSGD;make;ls HTKTools.64bit/HNTrainSGD')
else:
    os.system('make clean;rm HTKTools.64bit/HNTrainSGD;make -j 30;ls HTKTools.64bit/HNTrainSGD')
