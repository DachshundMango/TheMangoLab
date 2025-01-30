from datasets import load_dataset
import os
import pandas as pd
import torchaudio
import torch
import pprint

def mp3_bit_depth(file_path):

  try:
    
    waveform, sample_rate = torchaudio.load(file_path)
    dtype = waveform.dtype
    bit_depth_mapping = {
      torch.int8: 8,
      torch.int16: 16,
      torch.int32: 32,
      torch.float32: 32
    }
    
    return bit_depth_mapping.get(dtype, "Unknown")
  
  except Exception as e:
    print(f"Unexpected error occurs: {e}")
    return None

def process_audio_metadata(dataset_name, split, output_csv, source_name):
  
  dataset = load_dataset("laion/LAION-Audio-300M", split="train", streaming=True)

  metadata = []

  for sample in dataset:
    
    audio_info = sample["audio.mp3"]
    extra_info = sample["metadata.json"]
    
    filename = extra_info['segment_filename']
    filepath = audio_info['path']
    format = filename.split(".")[-1]
    sample_rate = audio_info['sampling_rate']
    duration = extra_info['duration_ms'] / 1000
    label = "bonafide"
    bit_depth = mp3_bit_depth(filepath)
    