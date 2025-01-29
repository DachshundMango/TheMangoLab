import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from fake_image_detectation.a_metadata_extractor import process_directory_for_metadata
from b_preprocessing import preprocess_image_from_metadata
from c_dataset import CustomDatasetFromMetadata

# CNN Model
class ImageFakeRealClassifier(torch.nn.Module):

  def __init__(self):

    super(ImageFakeRealClassifier, self).__init__()

    self.conv = torch.nn.Sequential(
      
      torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2),

      torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      torch.nn.ReLU(),
      torch.nn.MaxPool2d(kernel_size=2, stride=2)

    )

    self.fc = torch.nn.Sequential(
      
      torch.nn.Flatten(),
      torch.nn.Linear(32 * 56 * 56, 128),
      torch.nn.ReLU(),
      torch.nn.Linear(128, 2)

    )

  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x
  

if __name__ == "__main__":

  dataset_dir = "./dataset"
  metadata_csv = "./image_metadata.csv"
  processed_dir = "./processed_images"

  # 1단계: 메타데이터 생성
  print("Step 1: Generating metadata...")
  process_directory_for_metadata(dataset_dir, metadata_csv)
  print(f"Metadata saved to {metadata_csv}")

  # 2단계: 메타데이터 기반 전처리 실행
  print("Step 2: Starting image preprocessing...")
  preprocess_image_from_metadata(dataset_dir, metadata_csv, processed_dir)
  print(f"Preprocessing completed. Updated metadata at {metadata_csv}")

  # 3단계: 학습용 테스트용 transform 생성
  print("Step 3: Generating trasforms for train & test...")
  train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
  ])
  test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  
  # 4단계: Dataset 및 DataLoader 생성
  print("Step 4: Loading dataset...")
  train_dataset = CustomDatasetFromMetadata(metadata_csv, transform=train_transform)
  test_dataset = CustomDatasetFromMetadata(metadata_csv, transform=test_transform)
  print(f"Train dataset size: {len(train_dataset)}")
  print(f"Test dataset size: {len(test_dataset)}")
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
  
  # 5단계: 모델 초기화
  print("Step 5: Initializing model...")
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = ImageFakeRealClassifier().to(device)

  # 6단계: 손실 함수와 옵티마이저 정의
  print("Step 6: Defining loss function and optimizer...")
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

  # 7단계: 학습 루프
  print("Step 7: Starting training...")
  epochs = 5
  for epoch in range(epochs):
    
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
      images, labels = images.to(device), labels.to(device)
    
      # Forward
      outputs = model(images)
      loss = criterion(outputs, labels)

      # Backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    # 8단계: 검증 단계
    print(f"{epoch+1} Validation processing...")
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    val_loss /= len(test_loader)
    val_accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, "
          f"Val Accuracy: {val_accuracy:.2f}%")


  print("Training completed")
  
  # 9단계: 모델 저장
  print("Step 9: Saving model...")
  model_path = "./image_fake_real_classifier.pth"
  torch.save(model.state_dict(), model_path)
  print(f"Model saved to {model_path}")
  