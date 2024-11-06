COCO 2017 데이터셋(임시)을 사용하여 **Time Series Transformer (TST)** 모델을 학습하고, 이를 통해 이미지 시퀀스에서 객체를 예측

---


**1. 데이터 준비 및 전처리**

- **COCO 데이터셋**: `instances_train2017.json` 어노테이션 파일과 `train2017` 이미지를 사용하여 각 이미지에 대한 객체 클래스 정보를 확보.
- 이미지를 ResNet18을 이용해 **임베딩 벡터로 변환**하여 모델의 입력으로 사용.
- 기존 노션에 놀라온 **슬라이딩 윈도우**를 적용하여 시퀀스 데이터를 생성하고, 각 시퀀스에 대한 라벨(객체 클래스)을 멀티레이블로 인코딩.
- 전처리된 데이터(`tst_data.pt`, `tst_labels.pt`)를 저장하여 TST 모델 학습에 사용.

**2. 모델 정의 및 학습**

- pytorch 공식문서에서 **Time Series Transformer (TST)** 모델을 정의하여 시계열 이미지 데이터를 처리.
- `nn.TransformerEncoder`와 `PositionalEncoding`을 사용하여 시퀀스 데이터를 처리하는 구조 설계.
- 저장된 전처리 데이터를 사용하여 모델을 학습.
    - 손실 함수: `BCEWithLogitsLoss` (멀티레이블 분류용).
    - 옵티마이저: Adam, 학습률 `1e-4`.
    - 학습 완료 후 가중치를 `tst_model.pth`로 저장.

**3. 학습된 모델을 사용한 예측**

- 저장된 모델 가중치(`tst_model.pth`)를 로드하여 새로운 데이터에 대한 예측 수행.
- 새로운 이미지 시퀀스를 전처리(임베딩)하여 모델 입력 형태로 변환.
- 모델은 COCO 데이터셋의 80개 클래스 중 해당하는 클래햐스를 예측하고, 예측 결과를 출력.

---

- COCO 데이터셋을 기반으로 한 **TST 모델의 학습** 및 **가중치 저장** (`tst_model.pth`).
- 새로운 이미지 시퀀스에 대한 **객체 클래스 예측** 기능 구현.
- 학습된 모델을 통해 **이미지 분석** 및 **객체 감지**가 가능.

https://github.com/fire717/movenet.pytorch
## 모델 구현 코드
```
# 데이터 로드
data_sequences = torch.load('tst_data.pt')  # [num_sequences, window_size, embed_dim]
label_sequences = torch.load('tst_labels.pt')  # [num_sequences, window_size, num_classes]

# 데이터셋 및 데이터 로더 생성
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(data_sequences, label_sequences)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 학습 루프 예시
criterion = nn.BCEWithLogitsLoss()  # 멀티레이블 분류를 위한 손실 함수
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=1e-4)
num_epochs = 10

for epoch in range(num_epochs):
    transformer_model.train()
    for input_tensor, target_tensor in data_loader:
        optimizer.zero_grad()
        output = transformer_model(input_tensor)
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{num_epochs} 완료")

```



## 모델을 이용한 코드
```
 import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import torchvision

# 1. 모델 구조 재정의 (학습 시 사용했던 구조와 동일하게 정의)
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, embed_dim, num_classes, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, src):
        src = src.permute(1, 0, 2)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)
        logits = self.classifier(output)
        return logits

embed_dim = 512  # ResNet18의 임베딩 차원
num_classes = 80  # COCO 데이터셋의 클래스 수

# 2. 모델 인스턴스 생성
transformer_model = TimeSeriesTransformer(embed_dim, num_classes)

# 3. 가중치 로드
transformer_model.load_state_dict(torch.load('tst_model.pth'))
transformer_model.eval()  # 평가 모드로 설정

# 4. 새로운 데이터로 예측 수행
def frame_embedding(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    frame = preprocess(frame).unsqueeze(0)
    embedding_model = torchvision.models.resnet18(pretrained=True)
    embedding_model = nn.Sequential(*list(embedding_model.children())[:-1])
    embedding_model.eval()
    with torch.no_grad():
        embedding = embedding_model(frame).view(1, -1)  # [1, embed_dim]
    return embedding

# 예시로 새로운 이미지 시퀀스에서 예측
new_images = [cv2.imread('/path/to/image1.jpg'), cv2.imread('/path/to/image2.jpg')]  # 새 이미지 경로
frame_queue = []

for frame in new_images:
    embedded_frame = frame_embedding(frame).squeeze(0)
    frame_queue.append(embedded_frame)

# 시퀀스 길이 맞추기
if len(frame_queue) < 16:
    while len(frame_queue) < 16:
        frame_queue.insert(0, frame_queue[0])  # 첫 프레임을 복사하여 시퀀스 채우기

input_tensor = torch.stack(frame_queue).unsqueeze(0)  # [1, window_size, embed_dim]

with torch.no_grad():
    output = transformer_model(input_tensor)
    predictions = (torch.sigmoid(output) > 0.5).int()

# 예측 결과 출력
print("Predictions:", predictions)
```


## 결과값 보기 편하게 만든 코드
```
import os

import cv2

import torch

import torch.nn as nn

from torchvision import transforms

import torchvision

  

# 이미지가 저장된 폴더 경로

image_folder = './image' # 이미지가 저장된 폴더 경로를 지정

  

# 폴더에서 모든 이미지 파일 가져오기

image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

  

# 1. 모델 구조 재정의 (학습 시 사용했던 구조와 동일하게 정의)

class PositionalEncoding(nn.Module):

def __init__(self, embed_dim, dropout=0.1, max_len=5000):

super(PositionalEncoding, self).__init__()

self.dropout = nn.Dropout(p=dropout)

position = torch.arange(0, max_len).unsqueeze(1).float()

div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))

pe = torch.zeros(max_len, embed_dim)

pe[:, 0::2] = torch.sin(position * div_term)

pe[:, 1::2] = torch.cos(position * div_term)

pe = pe.unsqueeze(1)

self.register_buffer('pe', pe)

  

def forward(self, x):

x = x + self.pe[:x.size(0), :]

return self.dropout(x)

  

class TimeSeriesTransformer(nn.Module):

def __init__(self, embed_dim, num_classes, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1):

super(TimeSeriesTransformer, self).__init__()

self.positional_encoding = PositionalEncoding(embed_dim, dropout)

encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward)

self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

self.classifier = nn.Linear(embed_dim, num_classes)

  

def forward(self, src):

src = src.permute(1, 0, 2)

src = self.positional_encoding(src)

output = self.transformer_encoder(src)

output = output.permute(1, 0, 2)

logits = self.classifier(output)

return logits

  

embed_dim = 512 # ResNet18의 임베딩 차원

num_classes = 80 # COCO 데이터셋의 클래스 수

  

# 2. 모델 인스턴스 생성 및 가중치 로드

transformer_model = TimeSeriesTransformer(embed_dim, num_classes)

transformer_model.load_state_dict(torch.load('tst_model.pth', weights_only=True))

transformer_model.eval()

  

# 3. 새로운 데이터 예측

def frame_embedding(frame):

preprocess = transforms.Compose([

transforms.ToPILImage(),

transforms.Resize((128, 128)),

transforms.ToTensor(),

transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])

frame = preprocess(frame).unsqueeze(0)

embedding_model = torchvision.models.resnet18(pretrained=True)

embedding_model = nn.Sequential(*list(embedding_model.children())[:-1])

embedding_model.eval()

with torch.no_grad():

embedding = embedding_model(frame).view(1, -1) # [1, embed_dim]

return embedding

  

# COCO 클래스 인덱스와 이름 매핑

coco_classes = [

'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',

'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',

'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',

'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',

'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',

'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',

'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard',

'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',

'scissors', 'teddy bear', 'hair drier', 'toothbrush'

]

  

frame_queue = []

  

# 폴더 내 이미지에 대한 임베딩 및 시퀀스 구성

for image_path in image_files:

frame = cv2.imread(image_path)

if frame is None:

print(f"Failed to read image {image_path}.")

continue

embedded_frame = frame_embedding(frame).squeeze(0)

frame_queue.append(embedded_frame)

  

# 시퀀스 길이 맞추기 (최소 16개 프레임 필요)

while len(frame_queue) < 16:

frame_queue.insert(0, frame_queue[0]) # 첫 프레임을 복사하여 시퀀스 채우기

  

input_tensor = torch.stack(frame_queue).unsqueeze(0) # [1, window_size, embed_dim]

  

# 예측 수행

with torch.no_grad():

output = transformer_model(input_tensor)

probabilities = torch.sigmoid(output)

predictions = (probabilities > 0.3).int() # 임계값 0.3 적용

  

# 결과 출력 및 시각화

for i, frame_preds in enumerate(predictions[0]):

frame_probabilities = [(idx, prob.item()) for idx, prob in enumerate(probabilities[0][i]) if prob > 0.3]

predicted_classes = [coco_classes[idx] for idx, val in enumerate(frame_preds) if val == 1]

print(f"Frame {i+1} Probabilities: {frame_probabilities}")

print(f"Frame {i+1} Predictions: {predicted_classes}")

  

# 시각적 확인

frame_copy = cv2.imread(image_files[i])

if frame_copy is None:

continue

text_y = 30

for class_idx, prob in frame_probabilities:

text = f"{coco_classes[class_idx]}: {prob:.2f}"

cv2.putText(frame_copy, text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

text_y += 30

cv2.imshow(f"Frame {i+1} Predictions", frame_copy)

# 3000ms (3초) 동안 이미지 표시 후 자동으로 다음 이미지로 넘어가기

if cv2.waitKey(3000) & 0xFF == ord('q'):

break

  

cv2.destroyAllWindows()
```

## 예시
<img width="1281" alt="스크린샷 2024-11-06 오후 3 52 19" src="https://github.com/user-attachments/assets/cbad9783-a8c9-48ad-b40a-8bead514824b">

