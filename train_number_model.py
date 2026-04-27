import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from cnn_model_plus import NumberNet

class NumberDataset(Dataset):
    def __init__(self, csv_file, img_dir):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((40, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        tens = self.data.iloc[idx, 1]
        units = self.data.iloc[idx, 2]
        return image, tens, units

def train_number_model(epochs=30):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = NumberDataset("dataset/numbers/train/labels.csv", "dataset/numbers/train")
    # For special elite training, we validate on the same set
    test_set = train_set
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    
    model = NumberNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Ikilil Sayi Modeli Egitimi Basliyor ({DEVICE})...", flush=True)
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, tens, units in train_loader:
            imgs, tens, units = imgs.to(DEVICE), tens.to(DEVICE), units.to(DEVICE)
            
            optimizer.zero_grad()
            out_t, out_u = model(imgs)
            
            loss = criterion(out_t, tens) + criterion(out_u, units)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, tens, units in test_loader:
                imgs, tens, units = imgs.to(DEVICE), tens.to(DEVICE), units.to(DEVICE)
                ot, ou = model(imgs)
                _, pt = torch.max(ot, 1)
                _, pu = torch.max(ou, 1)
                
                correct += ((pt == tens) & (pu == units)).sum().item()
                total += imgs.size(0)
                
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: %{acc:.2f}", flush=True)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "number_classifier.pth")
            print(f"   --> Model kaydedildi (%{acc:.2f})")

    print(f"Egitim tamamlandi. En iyi: %{best_acc:.2f}", flush=True)

if __name__ == "__main__":
    train_number_model()
