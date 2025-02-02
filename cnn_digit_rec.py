import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,),(0.5,)),])
train_data=datasets.MNIST(
    root="C://Users//Eda ÇINAR//Desktop//cnn_digit",
    train=True,
    transform=transform,
    download=True
)
test_data=datasets.MNIST(
root="C://Users//Eda ÇINAR//Desktop//cnn_digit",
    train=False,
    transform=transform,
    download=True

)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=False)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

criterion=nn.NLLLoss()
optimizer=optim.Adam(model.parameters(),0.001)
epochs=10
for epoch in range(epochs):
    running_loss=0
    for images,labels in train_loader:
        images ,labels=images.to(device), labels.to(device)
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader)}")
correct=0
total=0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output=model(images)
        _, predicted = torch.max(output, 1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        print(f"Test Set Accuracy: ",100 * correct / total)

torch.save(model.state_dict(), "cnn_model.pt")

