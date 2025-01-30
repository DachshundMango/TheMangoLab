from pathlib import Path
from PIL import Image
import os
import pandas as pd

def is_valid_image(file_path):
  valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"]
  return file_path.suffix.lower() in valid_extensions

def extract_image_metadata(image_path):
  """
  이미지 메타데이터를 추출합니다.
  Args:
      image_path (str) : 이미지 파일 경로
  Returns:
      dict: 이미지 메타데이터    
  """
  try:
    # 레이블 설정: "real" 또는 "fake" 기반으로 설정
    if "real" in str(image_path.parts).lower():
      label = 1
    elif "fake" in str(image_path.parts).lower():
      label = 0
    else:
      print(f"Unknown label for {image_path}")
      return None # 유효하지 않은 데이터는 제외
    with Image.open(image_path) as img:
      filename = image_path.name
      filepath = str(image_path)
      format = img.format
      resolution = f"{img.width} x {img.height}"
      source = "dataset"
      colour_mode = img.mode
      #compression
      channels = len(img.getbands())
      #capture_divice
      file_size = os.path.getsize(image_path)

      metadata = {
        "filename": filename,
        "filepath": filepath,
        "format": format,
        "resolution": resolution,
        "source": source,
        "colour_mode": colour_mode,
        "compression": "unknown",
        "channels": channels,
        "capture_device": "unknown",
        "file_size": file_size,
        "label": label
      }

      return metadata
    
  except Exception as e:
    print(f"Error extracting metadata for {image_path}: {e}")
    return None
  
def process_directory_for_metadata(directory_path, output_csv):
  """
  디렉토리 내 모든 이미지의 메타데이터를 추출하고 CSV를 저장합니다.
  Args:
      directory_path (str): 이미지 파일이 저장된 디렉토리 경로
      output_csv (str): 저장할 CSV 파일 경로
  """
  directory_path = Path(directory_path)
  image_files = [f for f in directory_path.rglob("*.*") if is_valid_image(f)]
  metadata_list = []

  for image_file in image_files:
    metadata = extract_image_metadata(image_file)
    if metadata:
      metadata_list.append(metadata)

  if metadata_list:
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_csv, index = False)
    print(f"Metadata saved to {output_csv}")
  else:
    print("No valid metadata extracted. Check the directory of image fies.")