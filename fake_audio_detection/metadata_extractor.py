from datasets import load_dataset
import pandas as pd
import torchaudio
import os

def process_audio_metadata(dataset_name, split, output_csv, source_name):
  """
  허깅페이스 오디오 데이터셋에서 메타데이터를 추출하여 CSV 파일로 저장합니다.
  Args:
      dataset_name (str): 허깅페이스 데이터셋 이름
      split (str): 데이터셋 분할 ('train', 'set' 등)
      output_csv (str): 저장할 CSV 파일 경로
      source_name (str): 데이터셋 출처의 사용자 정의 이름 ('dataset1' 등) 
  """
  # 허깅페이스 데이터셋 로드 (스트리밍 방식)
  dataset = load_dataset(dataset_name, split=split, streaming=True)

  # 메타데이터 추출
  metadata = []
  for sample in dataset:
    audio = sample["audio"] # 오디오 정보 딕셔너리
    audio_path = audio["path"] # 오디오 파일 경로
    
    filename = os.path.basename(audio_path)
    filepath = os.path.dirname(audio_path) + "/"
    format = filename.split(".")[-1]
    
    sample_rate = audio["sampling_rate"]
    duration = len(audio["array"]) / sample_rate

    label = "bonafide" if sample['label'] == 0 else "spoof"
    
    bit_depth = audio.get("bit_depth", "16-bit") # default 
    channels = audio.get("channels", "mono") # default

    metadata.append({
      "filename": filename,
      "filepath": filepath,
      "format": format,
      "sample_rate": sample_rate,
      "source": source_name,
      "duration": duration,
      "label": label,
      "split": split,
      "bit_depth": bit_depth,
      "channels": channels
    })
      
