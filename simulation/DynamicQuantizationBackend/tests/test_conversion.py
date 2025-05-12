import torch.nn as nn

from backend import convert_model, create_config, QuantMode


class SubNet(nn.Module):
    """Nested subnetwork for testing module hierarchy"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        
    def forward(self, x):
        return self.fc2(self.fc1(x))

class TestNet(nn.Module):
    """Test network with various layer configurations"""
    def __init__(self):
        super().__init__()
        # Sequential block with Conv2d and Linear
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.Conv2d(16, 32, 3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Standalone Conv2d
        self.conv1 = nn.Conv2d(32, 64, 1)
        
        # Nested subnetwork
        self.sub = SubNet()
        
        # Final classifier
        self.classifier = nn.Linear(32, 10)
        
        # Another linear layer
        self.final_fc = nn.Linear(10, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.sub(x)
        x = self.classifier(x)
        return self.final_fc(x)

if __name__ == "__main__":
    # Create test network
    test_net = TestNet()
    
    # Print model structure
    print("Test Network Structure:")
    print(test_net)

    mode=QuantMode.ESTIMATE

    conf = create_config('config.py')
    breakpoint()

    convert_model(test_net, mode, conf, inplace=True)

    print(test_net)
    
    