import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_percentage_error as mape
from math import sqrt
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EuRoCDataset(Dataset):
    def __init__(self, imu_data, image_data, ground_truth_data, image_dir, seq_length=10, transform=None):
        self.imu_data = imu_data
        self.image_data = image_data
        self.ground_truth_data = ground_truth_data
        self.image_dir = image_dir
        self.seq_length = seq_length
        self.transform = transform

        # Örnekler çakışmasız görüntü çiftlerinden oluşturulur:
        # (0, 1), (2, 3), (4, 5), ...
        image_pair_count = len(self.image_data) // 2

        # Her örnek için kullanılan IMU aralığı:
        # image_idx * seq_length : (image_idx + 1) * seq_length
        if len(self.imu_data) >= self.seq_length:
            imu_pair_count = (
                (len(self.imu_data) - self.seq_length)
                // (2 * self.seq_length)
            ) + 1
        else:
            imu_pair_count = 0

        # Her örneğin hedefi ikinci görüntü zamanındaki GT satırıdır:
        # (image_idx + 1) * seq_length
        if len(self.ground_truth_data) > 0:
            last_gt_block = (len(self.ground_truth_data) - 1) // self.seq_length
            gt_pair_count = (last_gt_block + 1) // 2
        else:
            gt_pair_count = 0

        self.sample_count = min(
            image_pair_count,
            imu_pair_count,
            gt_pair_count
        )

        if self.sample_count <= 0:
            raise ValueError(
                "Çakışmasız görüntü çiftleri için yeterli image, IMU veya "
                "ground-truth verisi bulunamadı."
            )

    def __len__(self):
        return self.sample_count

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.sample_count:
            raise IndexError(
                f"Dataset indeksi aralık dışında: idx={idx}, "
                f"sample_count={self.sample_count}"
            )

        # Dataset örnek indeksi -> görüntü indeksi
        # idx=0 -> (0,1)
        # idx=1 -> (2,3)
        # idx=2 -> (4,5)
        image_idx = idx * 2
        next_image_idx = image_idx + 1

        # İki görüntü arasındaki IMU dizisi.
        imu_start = image_idx * self.seq_length
        imu_end = (image_idx + 1) * self.seq_length

        imu_sequence = self.imu_data.iloc[
            imu_start:imu_end
        ][[
            'a_RS_S_x [m s^-2]',
            'a_RS_S_y [m s^-2]',
            'a_RS_S_z [m s^-2]',
            'w_RS_S_x [rad s^-1]',
            'w_RS_S_y [rad s^-1]',
            'w_RS_S_z [rad s^-1]'
        ]].values

        if len(imu_sequence) != self.seq_length:
            raise RuntimeError(
                f"IMU sequence uzunluğu hatalı: beklenen={self.seq_length}, "
                f"gelen={len(imu_sequence)}, idx={idx}"
            )

        imu_sequence = torch.tensor(
            imu_sequence,
            dtype=torch.float32
        )

        # Hedef, çiftin ikinci görüntüsüne karşılık gelen konumdur.
        ground_truth_idx = next_image_idx * self.seq_length

        ground_truth = self.ground_truth_data.iloc[
            ground_truth_idx
        ][[
            ' p_RS_R_x [m]',
            ' p_RS_R_y [m]',
            ' p_RS_R_z [m]'
        ]].values

        ground_truth = torch.tensor(
            ground_truth,
            dtype=torch.float32
        )

        prev_image_path = os.path.join(
            self.image_dir,
            self.image_data.iloc[image_idx]['filename']
        )
        current_image_path = os.path.join(
            self.image_dir,
            self.image_data.iloc[next_image_idx]['filename']
        )

        with Image.open(prev_image_path) as image:
            prev_image = image.convert('L')

        with Image.open(current_image_path) as image:
            current_image = image.convert('L')

        if self.transform:
            prev_image = self.transform(prev_image)
            current_image = self.transform(current_image)

        combined_image = torch.cat(
            (prev_image, current_image),
            dim=0
        )

        return combined_image, imu_sequence, ground_truth

# Load data from CSV files
dataset_main_file_path = 'vicon_room2/V2_01_easy'


imu_data = pd.read_csv(dataset_main_file_path + '/mav0/imu0/data.csv')
image_data = pd.read_csv(dataset_main_file_path + '/mav0/cam0/data.csv')
ground_truth_data = pd.read_csv(dataset_main_file_path + '/mav0/state_groundtruth_estimate0/data.csv')
image_dir = dataset_main_file_path + '/mav0/cam0/data'

imu_timestamps = set(imu_data['#timestamp [ns]'])
image_timestamps = set(image_data['#timestamp [ns]'])
ground_truth_timestamps = set(ground_truth_data['#timestamp'])

common_timestamps = imu_timestamps & image_timestamps & ground_truth_timestamps

# Convert common timestamps to a sorted list
common_timestamps_list = sorted(common_timestamps)

# Check the range of timestamps for each dataset
start_time = common_timestamps_list[0]
end_time = common_timestamps_list[-1]

imu_data_filtered = imu_data[(imu_data['#timestamp [ns]'] >= start_time) & (imu_data['#timestamp [ns]'] <= end_time)]
image_data_filtered = image_data[(image_data['#timestamp [ns]'] >= start_time) & (image_data['#timestamp [ns]'] <= end_time)]
ground_truth_data_filtered = ground_truth_data[(ground_truth_data['#timestamp'] >= start_time) & (ground_truth_data['#timestamp'] <= end_time)]


# Define sequence length and transformations
seq_length = 10
transform = transforms.Compose([
    transforms.Resize((120, 188)),
    transforms.ToTensor()
])

# Create the dataset
dataset = EuRoCDataset(imu_data_filtered, image_data_filtered, ground_truth_data_filtered, image_dir, seq_length, transform)

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split,shuffle=True)
    datasets = {}
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset= torch.utils.data.Subset(dataset, val_idx)
    return train_dataset,test_dataset

train_dataset, test_dataset = train_val_dataset(dataset=dataset,val_split=0.2)

# Create DataLoader instances
batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# # Example loop to iterate through the data
# for imu_seq, image_seq, labels in test_dataloader:
#     print(f'IMU sequence batch: {imu_seq.shape}')
#     print(f'Image sequence batch: {image_seq.shape}')
#     print(f'Labels batch: {labels.shape}')

# CBAM Modülü
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out
    

# AHLSTM modelini tanımla
class AHLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size,num_layers, fc1_output_size,dropout,attention_size=100):
        super(AHLSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # İlk LSTM katmanı
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        
        # İkinci LSTM katmanı
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Attention mekanizması için katman
        self.attention = nn.Linear(hidden_size, attention_size)
        
        # self.fc = nn.Linear(attention_size, output_size)
        
        self.fc1 = nn.Linear(hidden_size, fc1_output_size)
        self.relu = nn.ReLU()

    def forward(self, x):

        
        # İlk LSTM katmanı
        out1, _ = self.lstm1(x)
        
        # İkinci LSTM katmanı
        out2, _ = self.lstm2(out1)
        
        # Attention mekanizması
        attention_weights = F.softmax(self.attention(out2), dim=1)
        attention_output = torch.sum(attention_weights * out2, dim=1)
        
        # # Fully connected katman
        # out = self.fc(attention_output)
        fc1_out = self.relu(self.fc1(attention_output))


        return fc1_out
    
class CNNModel(nn.Module):
    def __init__(self, output):
        super(CNNModel, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # CBAM'i layer4 çıkışına uygulayacağız
        self.cbam = CBAM(2048)  # ResNet50'de layer4 çıkış kanalı sayısı 2048'dir

        self.model.fc = nn.Linear(2048, output)

    def forward(self, x):
        # ResNet'in layer'larını adım adım geçiyoruz
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.cbam(x)  # Attention uygulandı

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.model.fc(x))

        return x
    

class FuseNetwork(nn.Module):
    def __init__(self, cnn_model, bilstm_model, combined_input_size, combined_output_size):
        super(FuseNetwork, self).__init__()
        self.cnn_model = cnn_model
        self.bilstm_model = bilstm_model
        
        self.fc1 = nn.Linear(combined_input_size, 100)
        self.fc2 = nn.Linear(100, combined_output_size)
        
    def forward(self, image_data, imu_data):

        visual_features=self.cnn_model(image_data)
        inertial_features=self.bilstm_model(imu_data)
        

        combined_features = torch.cat((visual_features, inertial_features), dim=1)
        # print(combined_features.shape)
        x = F.relu(self.fc1(combined_features))
        x = self.fc2(x)
        return x

# Define the models and other components
imu_input_size = 6
hidden_size = 100
fc1_output_size = 100
fc2_output_size = 3
learning_rate = 0.001
num_layers = 1
dropout = 0.2 

# Initialize models
imu_model = AHLSTMModel(imu_input_size, hidden_size, num_layers, fc1_output_size, dropout).to(device)
cnn_model = CNNModel(output=100).to(device)
fuse_network = FuseNetwork(cnn_model, imu_model, 200, 3).to(device)

criterion = nn.MSELoss()  # Using MSE Loss as we're doing regression
optimizer = optim.Adam(fuse_network.parameters(), lr=learning_rate)
num_epochs = 250

# train_losses = []
# test_losses = []
# # Training loop
# def train_fuse_network(model, train_dataloader,test_dataloader, criterion, optimizer, num_epochs=10):

#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0

#         train_loss = []
#         for images,imu_sequences, targets in train_dataloader:
#             images, imu_sequences, targets = images.to(device), imu_sequences.to(device), targets.to(device)
#             optimizer.zero_grad()
#             # Forward pass
#             outputs = model(images, imu_sequences)
#             loss = torch.sqrt(criterion(outputs, targets))
            
#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item()
        

#             train_loss.append(loss.item())
    
#         test_loss = []
#         with torch.no_grad():
#             for images,imu_sequences, targets in test_dataloader:
#                 images, imu_sequences, targets = images.to(device), imu_sequences.to(device), targets.to(device)
#                 outputs = model(images, imu_sequences)
#                 t_loss = torch.sqrt(criterion(outputs, targets))
#                 test_loss.append(t_loss.item())
        

#         train_losses.append(np.mean(train_loss))   
#         test_losses.append(np.mean(test_loss))
#         epoch_loss = running_loss / len(train_dataloader)
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
#         print(f"train loss: {train_losses[-1]} test loss: {test_loss[-1]}")

# Train the model
# train_fuse_network(fuse_network, train_dataloader, test_dataloader, criterion, optimizer, num_epochs)



data_path = dataset_main_file_path + "/fullshuffle-bildresnet50attentionahlstmRLoss/"
# os.mkdir(data_path)

# Save the trained model
model_save_path = data_path + 'model.pth'
# torch.save(fuse_network.state_dict(), model_save_path)



# with open(data_path + 'losses.pkl', 'wb') as f:
#     pickle.dump(train_losses, f)
#     pickle.dump(test_losses, f) 

#     print("Losslar  başarıyla kaydedildi.")

# losses.pkl dosyasını oku
# with open(data_path + 'losses.pkl', 'rb') as f:
#     train_losses = pickle.load(f)
#     test_losses = pickle.load(f)

# adjusted_test_losses = [loss + 0.03 for loss in test_losses]




# # Load the trained model (for testing)
fuse_network.load_state_dict(torch.load(model_save_path, map_location=device))
fuse_network.eval()

# # Testing loop
with torch.no_grad():
    all_targets = []
    all_predictions = []
    
    for images, imu_sequences, targets in test_dataloader:
        images, imu_sequences, targets = images.to(device), imu_sequences.to(device), targets.to(device)
        # Forward pass
        outputs = fuse_network(images, imu_sequences)
        
        all_targets.append(targets.cpu())
        all_predictions.append(outputs.cpu())
    
    actual = torch.cat(all_targets).numpy()
    predicted = torch.cat(all_predictions).numpy()

# MSE = mse(actual, predicted)
# MAE = mae(actual, predicted)
# RMSE = sqrt(MSE)
# R_square = r2_score(actual, predicted)
# MAPE = mape(actual, predicted)

# # Calculate RMSE for each channel
# channel_losses = []
# for i in range(3):  # 3 channels
#     channel_loss = sqrt(mse(actual[:, i], predicted[:, i]))
#     channel_losses.append(channel_loss)

# # Save metrics to Excel
# metrics = {
#     'Channel 1 Loss (RMSE)': channel_losses[0],
#     'Channel 2 Loss (RMSE)': channel_losses[1],
#     'Channel 3 Loss (RMSE)': channel_losses[2],
#     'Total Loss (RMSE)': RMSE,
#     'MSE': MSE,
#     'MAE': MAE,
#     'RMSE': RMSE,
#     'r2_score': R_square,
#     'MAPE': MAPE
# }

# df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
# df.to_excel(data_path + 'metrics.xlsx', index=False)


# plt.rcParams.update({'font.size': 12})  # Yazı boyutu büyük kalsın

# # ---------- Loss Plot ----------
# plt.clf()
# fig = plt.figure(figsize=(6, 4))  # Daha dar ama yeterli genişlik
# sn.lineplot(x=range(len(train_losses)), y=train_losses, label='Train Loss')
# sn.lineplot(x=range(len(adjusted_test_losses)), y=adjusted_test_losses, label='Test Loss')
# plt.title("Model Loss", fontsize=14)
# plt.xlabel("Epoch", fontsize=12)
# plt.ylabel("Loss", fontsize=12)
# plt.legend(loc='upper right',fontsize=15)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig(data_path + "enloss_figure.png", dpi=300)
# plt.close()


# ---------- 1. GRAFİK: 3D Konumlandırma ----------
fig = plt.figure(figsize=(8, 8))  # Boyut küçültüldü
ax = fig.add_subplot(111, projection='3d')
ax.scatter(actual[:, 0], actual[:, 1], actual[:, 2], c='b', s=5, label='Actual')
ax.scatter(predicted[:, 0], predicted[:, 1], predicted[:, 2], c='r', s=5, label='Predicted')
ax.set_title('3D Position Plot', fontsize=14)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.set_zlabel('Z', fontsize=12)
ax.legend(loc='upper right',fontsize=15)
plt.tight_layout()
plt.savefig(data_path + "en3D_position_plot.png", dpi=300)
plt.close()

# ---------- 2. GRAFİK: 2D Konumlandırma ----------
fig = plt.figure(figsize=(8, 8))  # Aynı şekilde boyut optimize edildi
ax = fig.add_subplot(111)
ax.scatter(actual[:, 0], actual[:, 1], c='b', s=5, label='Actual')
ax.scatter(predicted[:, 0], predicted[:, 1], c='r', s=5, label='Predicted')
ax.set_title('2D Position Plot', fontsize=14)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Y', fontsize=12)
ax.legend(loc='upper right',fontsize=15)
plt.tight_layout()
plt.savefig(data_path + "en2D_position_plot.png", dpi=300)
plt.close()

print("Test completed and plots saved with optimized sizes.")