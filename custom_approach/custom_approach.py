import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from TSViT import TSViT


#from models.patch_embeddings import StandardPatchEmbedding

# Placeholder for your TSViT model import
# from models.ts_vit import TSViT  # Adjust according to your actual model import

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Initialize your dataset (Replace with actual loading of your 24x24, 9-band Sentinel-1 data)
def load_data():
    # Placeholder for loading your dataset
    images = np.random.rand(100, 37,80, 80, 9).astype(np.float32)  # Example with 100 images of 9 bands
    labels = np.random.randint(0, 4, (100, 80, 80))  # For pixel-wise classification
    print(images.shape, labels.shape)

    return images, labels

# Simplified Training Loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            print(inputs.shape, labels.shape, "Labels", "inputs")
            inputs, labels = inputs.to(device), labels.to(device)

            B, T, H, W, C = inputs.shape
            # add channel that contains time steps
            time_points = torch.randint(low=0,high=365,size=(37,))
            print("time points", time_points)
            time_channel = time_points.repeat(B,H,W,1).permute(0,3,1,2) # BxTxHxW

            inputs = torch.cat((inputs, time_channel[:,:,:,:,None]), dim=4) # BxTxHxWxC + BxTxHxWx1 = BxTx(C+1)xHxW
            # last layer should contain only the value of the timestep for fixed T
    
            for t in range(T):
                assert int(np.unique(inputs[:,t,:,:,-1].numpy(), return_counts=True)[0][0]) == time_points.numpy()[t]

            inputs = inputs.permute(0, 1, 4, 2, 3)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    print('Finished Training')
# Load Data
images, labels = load_data()

# Create Dataset and DataLoader
train_dataset = CustomDataset(images, labels)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

patch_size = 2
config = {'patch_size': patch_size, 'patch_size_time': 1, 'patch_time': 4,
                    'dim': 128, 'temporal_depth': 6, 'spatial_depth': 2, 'channel_depth': 4,
                    'heads': 4, 'dim_head': 64, 'dropout': 0., 'emb_dropout': 0.,
                    'scale_dim': 4, 'depth': 4}
# Initialize the TSViT Model (Adjust according to your actual model import)
# For example: model = TSViT(input_channels=9, img_size=24, num_classes=2)
model = TSViT(config,  img_res=80, num_channels=[9], num_classes=4, max_seq_len=37, patch_embedding="Channel Encoding")
print("Done creating model")
print(model(torch.rand(1, 37, 10, 80, 80)))

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Train the Model
train_model(model, train_loader, criterion, optimizer)

# Save the Model
torch.save(model.state_dict(), 'tsvit_model.pth')
