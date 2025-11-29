import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ============================================================
# 1) 예시 데이터 (나중에 실제 데이터로 바꿔도 코드는 그대로 동작)
# ============================================================
X_driver = np.random.rand(1000, 10)
X_env = np.random.rand(1000, 5)
X_time_location = np.random.rand(1000, 3)
y = np.random.randint(0, 2, size=(1000,))

# ============================================================
# 2) 데이터 셋 분할
# ============================================================
X_driver_train, X_driver_test, \
X_env_train, X_env_test, \
X_time_location_train, X_time_location_test, \
y_train, y_test = train_test_split(
    X_driver, X_env, X_time_location, y,
    test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3) 텐서 변환 + GPU 이동
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_driver_train = torch.tensor(X_driver_train, dtype=torch.float32).to(device)
X_env_train = torch.tensor(X_env_train, dtype=torch.float32).to(device)
X_time_location_train = torch.tensor(X_time_location_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)

X_driver_test = torch.tensor(X_driver_test, dtype=torch.float32).to(device)
X_env_test = torch.tensor(X_env_test, dtype=torch.float32).to(device)
X_time_location_test = torch.tensor(X_time_location_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# ============================================================
# 4) DataLoader 만들기
# ============================================================
train_data = TensorDataset(
    X_driver_train, X_env_train, X_time_location_train, y_train
)
test_data = TensorDataset(
    X_driver_test, X_env_test, X_time_location_test, y_test
)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print("Training set size:", len(train_loader.dataset))
print("Test set size:", len(test_loader.dataset))

# ============================================================
# 5) 모델 정의
# ============================================================
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(18, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = SimpleNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# 6) 학습 함수
# ============================================================
def train_model(model, loader, criterion, optimizer, epochs=10):
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for driver, env, time_loc, labels in loop:
            inputs = torch.cat((driver, env, time_loc), dim=1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=running_loss / len(loader))

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")

# ============================================================
# 7) 평가 함수
# ============================================================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for driver, env, time_loc, labels in loader:
            inputs = torch.cat((driver, env, time_loc), dim=1)
            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ============================================================
# 8) 실행
# ============================================================
train_model(model, train_loader, criterion, optimizer, epochs=10)
evaluate(model, test_loader)
