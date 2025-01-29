from datasets import load_dataset

# 스트리밍 모드로 데이터 로드
dataset = load_dataset("ArissBandoss/fake_or_real_dataset_for_rerecorded", split="train")

print(dataset[0]['audio'])