import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

# CNN Modelini Tekrar Tanımla
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool=nn.MaxPool2d(kernel_size=2)
        self.fc1=nn.Linear(64*5*5, 128)
        self.fc2=nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Modeli yüklemek için cihaz ayarla
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli oluştur
model = CNN().to(device)

# Kaydedilen Ağırlıkları Yükle
model.load_state_dict(torch.load("cnn_model.pt", map_location=device, weights_only=True))

model.eval()  # Modeli inference moduna al
# Resmi Yükle ve Ön İşleme
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Siyah-beyaza çevir
        transforms.Resize((28, 28)),  # MNIST boyutuna getir
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(image_path)  # Resmi aç
    image = transform(image)
    print(image.shape)# Dönüştür
    image = image.unsqueeze(0)  # Batch boyutu ekle
    return image.to(device)

# Test Etmek İçin Örnek Bir Resim Kullan
image_path = "C://Users//Eda ÇINAR//Desktop//sekizz.png"  # Buraya test etmek istediğin resmin yolunu yaz
image = transform_image(image_path)

# Tahmin Yap
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)
    print(f"Tahmin Edilen Rakam: {predicted.item()}")
