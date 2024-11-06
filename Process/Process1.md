g4dn.2xlarge: AWS 서버 사양.

데이터 전처리팀이 "프레임 분할"과 "영상 크롭"을 진행

1. **프레임 분할**: 영상 데이터를 프레임별로 분할하여 각 프레임을 이미지처럼 개별적으로 저장하는 과정입니다. 이는 모델이 영상이 아닌 정지된 개별 이미지(프레임) 단위로 학습할 수 있게 하기 위해 자주 사용됩니다.
    
2. **랜덤 영상 및 비정상 영상 크롭**: 영상에서 관심 있는 부분(예: 사람이나 특정 객체)만 남기고 불필요한 배경을 제외하는 작업을 크롭이라고 합니다. 여기서 "랜덤 영상"은 정상적인 상태의 영상, "비정상 영상"은 폭력, 사고 등의 비정상적인 상태를 나타내는 영상으로 분류하여, 각각의 영상에서 중요한 영역만 잘라낸 것입니다.

즉, 데이터 전처리팀이 각 프레임을 분할한 뒤, 랜덤 영상과 비정상 영상에서 핵심 영역만 남겨 분석에 용이하도록 크롭한 것입니다. 이를 통해 모델 학습 시 불필요한 배경을 제거하고, 주목해야 할 영역에 집중할 수 있게 됩니다.


STSD (Spatial-Temporal Semantic Decomposition)는 일반적으로 스켈레톤 기반 행동 인식을 위해 인간 행동의 공간적 및 시간적 속성을 세분화하여 모델링하는 방법론을 의미합니다. 이는 행동 인식에 필요한 다양한 세부 요소, 즉 관절 간의 관계, 신체 부위의 의미 등을 모델링합니다.

STSD-TR (Spatial-Temporal Semantic Decomposition Transformer)는 이 STSD 개념을 Transformer 구조에 통합한 것입니다. 즉, STSD-TR은 Transformer의 자기 주의 메커니즘을 사용하여 스켈레톤 데이터의 시간적 및 공간적 관계를 모델링합니다. 이 모델은 관절 간의 관계를 캡처하고, 동시에 신체 부위의 의미와 하위 행동 의미를 명시적으로 고려하여 행동 인식 성능을 향상시키는 것을 목표로 합니다.

### MoveNet을 통한 크롭된 영상 학습 방법

MoveNet은 사람의 자세를 추정하기 위해 설계된 딥러닝 모델로, 스켈레톤 데이터를 추출하는 데 사용할 수 있습니다. 다음은 MoveNet을 통해 크롭한 영상을 STSD-TR과 함께 학습하는 방법입니다:

1. **데이터 수집**: MoveNet을 사용하여 비정상 행동이나 랜덤 영상을 실시간으로 분석합니다. 이 과정에서 각 프레임의 관절 위치를 추출하여 스켈레톤 형태로 변환합니다.
    
2. **데이터 전처리**: 크롭한 스켈레톤 데이터와 원본 영상에서 동작 정보를 추출합니다. STSD-TR 모델에 입력될 수 있는 형태로 변환합니다. 예를 들어, 각 동작의 서브-액션으로 분할하거나 신체 부위의 의미를 추가하여 스켈레톤 데이터의 특징을 강화합니다.
    
3. **모델 학습**: STSD-TR에 크롭된 스켈레톤 데이터를 입력으로 사용하여 행동 인식을 위한 모델 학습을 진행합니다. 이 과정에서 모델은 관절 간의 관계 및 하위 동작을 이해하고, 특정 행동을 인식하는 데 필요한 공간적 및 시간적 패턴을 학습합니다.



논문 제목: **STSD: Spatial-Temporal Semantic Decomposition Transformer for Skeleton-Based Action Recognition**

### 요약 및 개요

이 논문은 **STSD-TR**라는 새로운 변환기 기반 모델을 제안합니다. 이 모델은 3D 스켈레톤 데이터를 사용하여 사람의 행동을 인식하는 데 초점을 맞추고 있습니다. 스켈레톤 데이터는 조명, 카메라 각도 및 복잡한 배경 변화에 매우 강력합니다. 이 모델은 관절 간의 관계를 모델링하고, 사람의 신체 부위와 동적 의미 정보를 활용하여 행동 인식 성능을 향상시킵니다.

### 주요 구성 요소

1. **Body Parts Semantic Decomposition Module (BPSD)**:
    
    - 이 모듈은 3D 관절 좌표에서 신체 부위의 의미 정보를 추출합니다. 이를 통해 각 관절이 어떤 신체 부위에 속하는지를 명시적으로 나타냅니다.
2. **Temporal-Local Spatial-Temporal Attention Module (TL-STA)**:
    
    - 이 모듈은 여러 연속 프레임의 관절 간 관계를 캡처하여 지역적인 하위 행동 의미 정보를 이해합니다.
3. **Global Spatial-Temporal Module (GST)**:
    
    - 이 모듈은 지역적 특성을 통합하여 전체 행동 시퀀스에 대한 글로벌 표현을 생성합니다.
4. **BodyParts-Mix 전략**:
    
    - 두 사람의 신체 부위를 독특한 방식으로 혼합하여 성능을 더욱 높입니다. 이 방식은 데이터 다양성을 증가시킵니다.

### 행동 인식의 중요성

행동 인식은 로봇 비전, 비디오 감시, 인간-컴퓨터 상호작용, 스포츠 분석 등 다양한 분야에서 응용됩니다. 스켈레톤 데이터는 RGB 비디오보다 더 적은 데이터로도 높은 인식 정확도를 제공합니다. 이 연구는 다양한 행동 인식 방법을 비교하며 STSD-TR의 성능을 두 개의 대규모 데이터 세트(NTU RGB+D 및 NTU RGB+D-120)에서 평가합니다.

### STSD-TR의 혁신점

- **신체 부위 의미 정보**와 **하위 행동 의미 정보**를 명시적으로 사용하여 행동 인식 성능을 향상시킵니다.
- 변환기 기반의 모델 구조를 통해 수동으로 정의된 그래프 구조가 필요 없으며, 데이터 기반의 접근 방식을 사용합니다.
- 행동을 여러 하위 행동으로 분해하여 보다 정확한 인식이 가능하게 합니다.

### MoveNet을 통한 영상 학습

**MoveNet**은 사람의 자세를 추정하여 스켈레톤 데이터를 생성하는 데 사용될 수 있습니다. 크롭한 영상에서 MoveNet을 적용하여 각 프레임의 관절 위치를 추출하고, 이를 STSD-TR 모델에 입력으로 사용하여 행동 인식을 위한 학습을 진행할 수 있습니다.

https://github.com/Chiaraplizz/ST-TR 



근데 안됨 잘;
그래서 st-gcn 쓰려했는데 환경 다 맞추니 또 없어졌대? 그래서
그것의 신모델인 
https://github.com/open-mmlab/mmskeleton/tree/master
이걸로 넘어감.

근데 김서연과 대화해본 결과
-> mmpose는 전처리 + 모델인데 굳이 쓸 이유가 없음
-> 그래서 tst로 학습 시도.
-> 전처리 코드 노션에서 가져와서 수정함. -> 이후 텐서로 저장 (내가 쓰려고)

##  데이터 전처리 코드
```
import os
import cv2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from pycocotools.coco import COCO

# 필요한 설정
window_size = 16  # 시퀀스 길이
frame_queue = []  # 슬라이딩 윈도우를 유지할 큐
labels_queue = []  # 슬라이딩 윈도우의 라벨을 유지할 큐
embed_dim = 512  # ResNet18 임베딩 차원
num_classes = 80  # COCO 데이터셋의 클래스 수

# COCO 데이터셋 경로 설정
image_dir = 'data/train2017'  # 이미지 경로 (사용자 지정)
annotation_file = 'data/annotations/instances_train2017.json'  # 어노테이션 파일 경로 (사용자 지정)

# COCO API 초기화
coco = COCO(annotation_file)

# 유효한 클래스 ID를 0부터 시작하는 인덱스로 맵핑
valid_coco_classes = {cat['id']: idx for idx, cat in enumerate(coco.loadCats(coco.getCatIds()))}

# 이미지 ID 목록 가져오기
img_ids = coco.getImgIds()
img_infos = coco.loadImgs(img_ids)

# 프레임 임베딩을 위한 사전 학습된 모델 (ResNet18)
embedding_model = torchvision.models.resnet18(pretrained=True)
embedding_model = nn.Sequential(*list(embedding_model.children())[:-1])  # 마지막 FC 레이어 제거
embedding_model.eval()

def frame_embedding(frame):
    # 전처리: 이미지 변환
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    frame = preprocess(frame)  # [C, H, W]
    frame = frame.unsqueeze(0)  # [1, C, H, W]
    with torch.no_grad():
        embedding = embedding_model(frame)  # [1, 512, 1, 1]
    embedding = embedding.view(embedding.size(0), -1)  # [1, embed_dim]
    return embedding  # [1, embed_dim]

# Transformer 모델 초기화 (모델 정의는 이전 코드와 동일)
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
        # x shape: [seq_length, batch_size, embed_dim]
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
        # src shape: [batch_size, seq_length, embed_dim]
        src = src.permute(1, 0, 2)  # [seq_length, batch_size, embed_dim]
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # [batch_size, seq_length, embed_dim]
        logits = self.classifier(output)  # [batch_size, seq_length, num_classes]
        return logits

# 모델 초기화
transformer_model = TimeSeriesTransformer(embed_dim, num_classes)
transformer_model.eval()  # 평가 모드로 설정

# 슬라이딩 윈도우 데이터를 저장할 리스트
data_sequences = []
label_sequences = []

# COCO 이미지 처리 루프
for idx, img_info in enumerate(img_infos):
    # 진행 상황 출력 (선택 사항)
    if (idx + 1) % 1000 == 0:
        print(f"Processing image {idx + 1}/{len(img_infos)}")

    # 이미지 파일 경로
    img_path = os.path.join(image_dir, img_info['file_name'])
    # 이미지 읽기
    frame = cv2.imread(img_path)
    if frame is None:
        continue  # 이미지 읽기에 실패하면 다음으로 넘어감

    # 프레임 전처리 및 슬라이딩 윈도우 큐에 추가
    embedded_frame = frame_embedding(frame)
    embedded_frame = embedded_frame.squeeze(0)  # [embed_dim]
    frame_queue.append(embedded_frame)

    # 이미지에 대한 클래스 라벨 추출
    ann_ids = coco.getAnnIds(imgIds=img_info['id'])
    anns = coco.loadAnns(ann_ids)
    categories = set()
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id in valid_coco_classes:
            categories.add(valid_coco_classes[cat_id])
    label = torch.zeros(num_classes)
    for cat_idx in categories:
        label[cat_idx] = 1  # 멀티레이블 바이너리 인코딩
    labels_queue.append(label)

    # 윈도우 크기 유지
    if len(frame_queue) > window_size:
        frame_queue.pop(0)
        labels_queue.pop(0)

    # 윈도우가 충분히 채워졌을 때 데이터를 저장
    if len(frame_queue) == window_size:
        data_sequences.append(torch.stack(frame_queue))  # [window_size, embed_dim]
        label_sequences.append(torch.stack(labels_queue))  # [window_size, num_classes]

# 데이터를 텐서 파일로 저장
if data_sequences and label_sequences:
    data_sequences = torch.stack(data_sequences)  # [num_sequences, window_size, embed_dim]
    label_sequences = torch.stack(label_sequences)  # [num_sequences, window_size, num_classes]

    torch.save(data_sequences, 'tst_data.pt')
    torch.save(label_sequences, 'tst_labels.pt')

    print("데이터와 라벨이 저장되었습니다.")
else:
    print("저장할 데이터가 없습니다.")
```

