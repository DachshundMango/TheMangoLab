from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pandas as pd
from PIL import Image
from pathlib import Path


# 사용자 정의 Dataset 클래스
class CustomDatasetFromMetadata(Dataset):  
  """
  전처리 완료된 메타데이터 CSV 를 기반으로 PyTorch Dataset 을 생성합니다
  """
  
  def __init__(self, processed_metadata_csv, transform=None):
    """
    Args:
        processed_metadata_csv (str): 메타데이터 CSV 파일 경로
        transform (callable, optional): 데이터 증강 및 정규화 수행 함수
    """ 
    self.df = pd.read_csv(processed_metadata_csv)
    self.df = self.df[self.df['filepath'].notnull()]
    self.transform = transform if transform else self._default_transforms()
  def __len__(self):
    """
    데이터셋 크기 변환 (이미지 갯수)
    """
    return len(self.df)
  
  def __getitem__(self, index):
    """
    데이터셋에서 특정 인덱스의 데이터를 반환합니다.
    """
    row = self.df.iloc[index]
    img_path = Path(row['filepath'])
    label = row['label']
    
    try:
      img = Image.open(img_path).convert('RGB')

      if self.transform:
        img = self.transform(img)
      
      return img, label
    
    except Exception as e:
      print(f"Error loading image at {img_path}: {e}")
      return None, None
    
  def _default_transforms(self):
    """
    기본 데이터 증강 및 정규화 트랜스폼 정의
    """
    return transforms.Compose([
      transforms.Resize((224, 224)), # 크기 조정
      transforms.RandomHorizontalFlip(), # 좌우 반전
      transforms.RandomRotation(15), # 회전
      transforms.ColorJitter(brightness=0.2), #밝기 조정
      transforms.ToTensor(), # 텐서 변환
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 정규화
    ])