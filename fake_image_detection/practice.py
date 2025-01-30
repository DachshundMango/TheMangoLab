from pathlib import Path
from PIL import Image
import os
import pandas as pd

path1 = Path("./image_metadata.csv")

image_path = Path("./dataset/training_fake/mid_480_1111.jpg")

print(image_path.name)
print(image_path.parts)
print(image_path.rglob)

with Image.open(image_path) as img:
  print(os.path.getsize(image_path) / (1024 * 1024))
  

image_metadata_csv = pd.read_csv("./image_metadata.csv")

print(image_metadata_csv[["filename", "label"]].head())
