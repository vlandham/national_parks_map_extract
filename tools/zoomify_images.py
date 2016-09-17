#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import deepzoom

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = '../data/npmaps_jpg'
OUTPUT_DIR = '../data/zooms'


input_filenames = glob.glob(os.path.join(CUR_DIR, INPUT_DIR, '*.jpg'))

creator = deepzoom.ImageCreator(tile_size=256, tile_overlap=2, tile_format="jpg",
                                image_quality=0.7, resize_filter="bicubic")

for filename in input_filenames:
    print(filename)
    basename = os.path.basename(filename).split('.')[0]

    output_file = os.path.join(CUR_DIR, OUTPUT_DIR, basename + ".dzi")
    creator.create(filename, output_file)

# Specify your source image
#SOURCE = "helloworld.jpg"

# Create Deep Zoom Image creator with weird parameters

# Create Deep Zoom image pyramid from source
#creator.create(SOURCE, "helloworld.dzi")
