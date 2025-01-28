from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms
from a_metadataExtractor import process_directory_for_metadata

def preprocess_image_from_metadata(input_dir, metadata_csv, output_dir):
  """
  메타데이터 CSV 를 기반으로 이미지를 전처리하고, 전처리 결과를 업데이트된 메타데이터로 저장합니다.
  Args:
      input_dir (str): 원본 이미지가 있는 디렉토리
      metadata_csv (str): 메타데이터 CSV 파일 경로
      output_dir (str): 전처리된 이미지가 저장될 디렉토리
      
  Returns:
      pd.DataFrame: 전처리 결과가 반영된 메타데이터
  """
  # 메타데이터 CSV 파일 신규 생성
  if not Path(metadata_csv).exists():
    print(f"Metadata file not fount at {metadata_csv}. Generating metadata...")
    process_directory_for_metadata(input_dir, metadata_csv)

  # 메타데이터 로드
  metadata = pd.read_csv(metadata_csv)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True) # 출력 디렉토리 생성
  processed_metadata = []
  
  # Transform 정의 (Normalization 포함)
  transform = transforms.Compose([
    transforms.Resize((224, 224)), # 크기 조정
    transforms.ToTensor(), # 텐서 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  for index, row in metadata.iterrows():
    img_path = Path(row['filepath'])
    label = row['label']
    
    try:
      # 원본 이미지 경로
      image = Image.open(img_path).convert("RGB") # 이미지를 RGB로 변환
      transformed_image = transform(image) # Transform 적용

      # 전처리된 이미지를 저장
      processed_path = output_dir / img_path.name
      transformed_image_pil = transforms.ToPILImage()(transformed_image)
      transformed_image_pil.save(processed_path)
      
      # 처리된 메타데이터 저장
      processed_metadata.append({
        'filepath': str(processed_path),
        'label': label
      })        
  
    except Exception as e:
      print(f"Error processing {img_path}: {e}")
      continue

  # 처리된 메타데이터 CSV 저장
  processed_metadata_df = pd.DataFrame(processed_metadata)
  processed_metadata_df.to_csv(output_dir / 'processed_metadata.csv', index=False)
  print(f"Updated metadata saved to {output_dir}")
  return processed_metadata_df

if __name__ == "__main__":
  # 메타데이터 경로와 출력 디렉토리설정
  input_dir = "./dataset"
  metadata_csv = "./image_metadata.csv"
  output_dir = "./processed_images"
  # 전처리 싱행
  print("Starting image preprocessing...")
  preprocessed_metadata = preprocess_image_from_metadata(input_dir, metadata_csv, output_dir)
  print("Image preprocessing completed")
