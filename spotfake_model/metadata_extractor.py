import os
from pathlib import Path
import pandas as pd
from PIL import Image, ExifTags
import piexif

IMAGE_DIR = "./image"

metadata = []

def exif_data_for_img(file_path):
    try:
      
        if not file_path or file_path == b"":
            return "unknown", "unknown"

        
        if isinstance(file_path, bytes):
            file_path = file_path.decode("utf-8").strip()

        if not os.path.exists(file_path):
            return "unknown", "unknown"      
      
        with Image.open(file_path) as img:
            exif_data = img._getexif()
            if exif_data is None:
                exif_data = piexif.load(img.info.get("exif", b"")).get("0th", {})
            
            if not exif_data:
                return "unknown", "unknown"

            exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}   
            
            capture_device = exif.get("Model", "unknown")
            compression = exif.get("Compression", "unknown")
            
            return capture_device, compression
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return "unknown", "unknown"

def split_img(image_path):
    filepath = os.path.dirname(image_path) 
    if "train" in filepath:
        split = "train"
    elif "test" in filepath:
        split = "test" 
    elif "val" in filepath:
        split = "val" 
    else:
        split = "unknown"
    return split

def ground_truth_img(image_path):
    filepath = os.path.dirname(image_path)
    if "real" in filepath:
        ground_truth = 1
    elif "fake" in filepath:
        ground_truth = 0
    else:
        ground_truth = -1
    return ground_truth

def is_valid_image(image_path):
    valid_extensions = [".png", ".jpg", ".jpeg"]
    return image_path.suffix.lower() in valid_extensions

def get_image_metadata(image_path):

    try:
        with Image.open(image_path) as img:
            filename = os.path.basename(image_path) # file name
            filepath = os.path.dirname(image_path) # file path
            format = img.format # JPEG, PNG, etc
            resolution = f"{img.width}x{img.height}" # resolution
            source = img.info.get("source", "")
            colour_mode = img.mode # RGB, CMYK, etc.
            channels = len(img.getbands()) # number of channels
            capture_device, compression = "unknown", "unknown"
            file_size = os.path.getsize(image_path) # file size in bytes
            split = split_img(image_path)
            ground_truth = ground_truth_img(image_path)
            ground_truth_metadata = "1 is real, 0 is fake, -1 is unknown"
            original_ground_truth = img.info.get("label", "unknown")

            return {
                "filename": filename,
                "filepath": filepath,
                "format": format,
                "resolution": resolution,
                "source": source,
                "colour_mode": colour_mode,
                "channels": channels,
                "capture_device": capture_device,
                "compression": compression,
                "file_size": file_size,
                "split": split,
                "ground_truth": ground_truth,
                "ground_truth_metadata": ground_truth_metadata,
                "original_ground_truth": original_ground_truth
            }
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


directory_path = Path(IMAGE_DIR)

image_files = [f for f in directory_path.rglob("*.*") if is_valid_image(f)]
metadata_list = []

for image_path in image_files:
    metadata = get_image_metadata(image_path)
    if metadata:
        metadata_list.append(metadata)

df = pd.DataFrame(metadata_list)
df.to_csv("local_img_metadata.csv", index=False)
