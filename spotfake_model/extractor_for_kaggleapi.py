import os
import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS
import pprint
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Get the list of files in the dataset

dataset = "shmalex/instagram-images"

files = api.dataset_list_files(dataset)

kaggle_metadata = []

for file in files.files:
    kaggle_metadata.append({
        "filename": file.name,
        "file_size": file.size
    })

print(kaggle_metadata[0])