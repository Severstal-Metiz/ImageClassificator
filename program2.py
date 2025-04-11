import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime

# Параметры
IMAGE_SIZE = (64, 100)
BATCH_SIZE = 2
EPOCHS = 250 #118
LEARNING_RATE = 0.00002
MODEL_PATH = "model/checkpoint"
FORWARD_INPUT = "forwardinput"
OUTPUT_GOAL = "outputgoal"
OUTPUT_OTHER = "outputother"
TRAIN_GOAL = "traingoal"
TRAIN_OTHER = "trainother"
TRAIN_FOLDER = "train"
TEST_FOLDER = "test"


def create_dirs():
    os.makedirs(OUTPUT_GOAL, exist_ok=True)
    os.makedirs(OUTPUT_OTHER, exist_ok=True)
    os.makedirs(TEST_FOLDER, exist_ok=True)


def get_data_loader(train=True):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=TRAIN_FOLDER, transform=transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)


def train_model():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataloader = get_data_loader()

    losslist=[]
    plt.xlabel('Эпоха')
    plt.ylabel('Total Loss')
    test()

    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(dataloader):.4f}")
        losslist.append(total_loss)
    torch.save(model.state_dict(), MODEL_PATH)
    plt.plot( losslist, 'r-o')
    plt.savefig(f"{TEST_FOLDER}/Epochs_{EPOCHS}_LR_{LEARNING_RATE}.jpg")
    plt.show()

def classify_images():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    test()
    for filename in os.listdir(FORWARD_INPUT):
        filepath = os.path.join(FORWARD_INPUT, filename)
        image = Image.open(filepath).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image).cpu().numpy().flatten()
        class_index = output.argmax()
        target_folder = OUTPUT_GOAL if class_index == 0 else OUTPUT_OTHER
        shutil.copy(filepath, os.path.join(target_folder, filename))
        print(f"{filename}: {'goal' if class_index == 0 else 'other'} (Confidence goal: {output[0]:.2f}, Confidence other: {output[1]:.2f})")
def test():
    for filename in os.listdir(FORWARD_INPUT):
        filepath = os.path.join(FORWARD_INPUT, filename)
        filepathout = os.path.join(TEST_FOLDER, filename)
        image = Image.open(filepath).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image.save(filepathout)
        #plt.plot([1,2,3,4,5], 'r-o')
        #plt.savefig(f"{TEST_FOLDER}/Epochs_{EPOCHS}_LR_{LEARNING_RATE} Datetime: {datetime.now()}.jpg")
        #print(datetime.strptime(str(datetime.now()),"%d_%m_%Y_%H_%M"))
        #plt.savefig(f"{TEST_FOLDER}/Epochs_{EPOCHS}_LR_{LEARNING_RATE}.jpg")
        #plt.savefig("1.jpg")




if __name__ == "__main__":
    create_dirs()
    action = input("Введите 'tr' для обучения или 'cl' для классификации, test для просмотра миниатюр изображений в папке test:")
    if action == 'tr':
        train_model()
    elif action == 'cl':
        classify_images()
    elif action == 'test':
        test()
