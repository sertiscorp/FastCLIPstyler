import os
from tqdm import tqdm
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from styleaug.text_embedder import TextEmbedder
from FastCLIPstyler import FastCLIPStyler

mapping = {
    "Thawan Duchanee style": "red swirly fire and scary", 
    "Chakrabhand Posayakrit style": "style of mild colorful pastels",
    "Chalermchai Kositpipat style": "blue color canvas painting",
    "Vasan Sitthiket style": "fauvism style",
    "Fua Haripitak style": "crayon style painting",
    "Mit Jai Inn style": "colorful pointillism style"
}

class params:
    img_width=512
    img_height=512
    num_crops=16
    text_encoder='edgeclipstyler' # can be either fastclipstyler or edgeclipstyler

class TrainStylePredictor():
    def __init__(self):
        self.trainer = FastCLIPStyler(params)

        # pass

    def test(self, content_img, style_desc):

        # trainer = FastCLIPStyler(params)

        self.trainer.content_image = content_img
        if style_desc in mapping.keys():
            style_desc = mapping[style_desc]
        params.text = style_desc

        return self.trainer.test()

