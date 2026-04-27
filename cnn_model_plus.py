import torch
import torch.nn as nn

class NumberNet(nn.Module):
    """
    0-99 Arası Sayıları Çift Kafalı (Multitask) Tanıyan Model
    Giriş: 1 x 40 x 64
    """
    def __init__(self):
        super(NumberNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 20 x 32
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 10 x 16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 5 x 8
        )
        
        # 2 Kafa: Onlar ve Birler Basamağı
        # 11 Sınıf: 0-9 + 10 (Boş)
        self.fc_tens = nn.Sequential(
            nn.Linear(128 * 5 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 11)
        )
        
        self.fc_units = nn.Sequential(
            nn.Linear(128 * 5 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 11)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_tens(x), self.fc_units(x)

if __name__ == "__main__":
    # Test
    model = NumberNet()
    dummy_input = torch.randn(1, 1, 40, 64)
    output = model(dummy_input)
    print(f"Sayı Modeli Çıkış Boyutu (Beklenen 1x100): {output.shape}")
