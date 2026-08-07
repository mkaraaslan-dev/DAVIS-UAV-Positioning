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
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))),
        test_size=val_split,
        shuffle=True,
        random_state=42
    )
    datasets = {}
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset= torch.utils.data.Subset(dataset, val_idx)
    return train_dataset,test_dataset

train_dataset, test_dataset = train_val_dataset(dataset=dataset,val_split=0.2)

# Create DataLoader instances
batch_size = 8

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# # Example loop to iterate through the data
# for imu_seq, image_seq, labels in test_dataloader:
#     print(f'IMU sequence batch: {imu_seq.shape}')
#     print(f'Image sequence batch: {image_seq.shape}')
#     print(f'Labels batch: {labels.shape}')
# =============================================================================
# ABLATION STUDY: A1 - A8
# =============================================================================

import gc
import random


def set_random_seed(seed=42):
    """Tüm ablasyon deneylerinde tekrar üretilebilirliği artırır."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# CBAM
# -----------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        reduced_channels = max(1, in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(
                in_planes,
                reduced_channels,
                kernel_size=1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                reduced_channels,
                in_planes,
                kernel_size=1,
                bias=False
            )
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat((avg_out, max_out), dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# -----------------------------------------------------------------------------
# IMU modelleri
# -----------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """
    AHLSTM ile aynı iki LSTM ve FC yapısını kullanır.
    Fark: attention yerine son zaman adımındaki çıktı kullanılır.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        fc1_output_size,
        dropout
    ):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.fc1 = nn.Linear(hidden_size, fc1_output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)

        # Attention yok: son zaman adımı kullanılır.
        last_output = out2[:, -1, :]
        return self.relu(self.fc1(last_output))


class AHLSTMModel(nn.Module):
    """
    Orijinal AHLSTM yapısı korunmuştur.
    Her hidden feature için zaman ekseni üzerinde attention uygulanır.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        fc1_output_size,
        dropout,
        attention_size=100
    ):
        super(AHLSTMModel, self).__init__()

        if attention_size != hidden_size:
            raise ValueError(
                "Mevcut AHLSTM çarpımı için attention_size ve hidden_size "
                "eşit olmalıdır."
            )

        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            batch_first=True
        )

        self.attention = nn.Linear(hidden_size, attention_size)
        self.fc1 = nn.Linear(hidden_size, fc1_output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1, _ = self.lstm1(x)
        out2, _ = self.lstm2(out1)

        attention_weights = F.softmax(
            self.attention(out2),
            dim=1
        )
        attention_output = torch.sum(
            attention_weights * out2,
            dim=1
        )

        return self.relu(self.fc1(attention_output))


# -----------------------------------------------------------------------------
# Görsel model
# -----------------------------------------------------------------------------

class CNNModel(nn.Module):
    """
    use_cbam=False: ResNet50
    use_cbam=True : ResNet50 + layer4 sonrası CBAM
    """

    def __init__(self, output, use_cbam):
        super(CNNModel, self).__init__()

        self.use_cbam = use_cbam
        self.model = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT
        )

        # Ardışık iki gri görüntü iki kanal olarak verilir.
        self.model.conv1 = nn.Conv2d(
            2,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        if self.use_cbam:
            self.cbam = CBAM(2048)
        else:
            self.cbam = None

        self.model.fc = nn.Linear(2048, output)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        if self.cbam is not None:
            x = self.cbam(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return F.relu(self.model.fc(x))


# -----------------------------------------------------------------------------
# Fusion ve tek-modalite ağları
# -----------------------------------------------------------------------------

class FuseNetwork(nn.Module):
    def __init__(
        self,
        cnn_model,
        imu_model,
        combined_input_size,
        combined_output_size
    ):
        super(FuseNetwork, self).__init__()

        self.cnn_model = cnn_model
        self.imu_model = imu_model

        self.fc1 = nn.Linear(combined_input_size, 100)
        self.fc2 = nn.Linear(100, combined_output_size)

    def forward(self, image_data, imu_data):
        visual_features = self.cnn_model(image_data)
        inertial_features = self.imu_model(imu_data)

        combined_features = torch.cat(
            (visual_features, inertial_features),
            dim=1
        )

        x = F.relu(self.fc1(combined_features))
        return self.fc2(x)


class VisualOnlyNetwork(nn.Module):
    def __init__(self, cnn_model, feature_size=100, output_size=3):
        super(VisualOnlyNetwork, self).__init__()

        self.cnn_model = cnn_model
        self.fc1 = nn.Linear(feature_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, image_data, imu_data):
        del imu_data

        visual_features = self.cnn_model(image_data)
        x = F.relu(self.fc1(visual_features))
        return self.fc2(x)


class IMUOnlyNetwork(nn.Module):
    def __init__(self, imu_model, feature_size=100, output_size=3):
        super(IMUOnlyNetwork, self).__init__()

        self.imu_model = imu_model
        self.fc1 = nn.Linear(feature_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, image_data, imu_data):
        del image_data

        inertial_features = self.imu_model(imu_data)
        x = F.relu(self.fc1(inertial_features))
        return self.fc2(x)


# -----------------------------------------------------------------------------
# Ablasyon tanımları
# -----------------------------------------------------------------------------

ABLATION_EXPERIMENTS = {
    "A1_ResNet50_LSTM_Fusion": {
        "modality": "fusion",
        "use_cbam": False,
        "imu_model": "lstm"
    },
    "A2_ResNet50_CBAM_LSTM_Fusion": {
        "modality": "fusion",
        "use_cbam": True,
        "imu_model": "lstm"
    },
    "A3_ResNet50_AHLSTM_Fusion": {
        "modality": "fusion",
        "use_cbam": False,
        "imu_model": "ahlstm"
    },
    "A4_ResNet50_CBAM_AHLSTM_Fusion": {
        "modality": "fusion",
        "use_cbam": True,
        "imu_model": "ahlstm"
    },
    "A5_VisualOnly_ResNet50": {
        "modality": "visual",
        "use_cbam": False,
        "imu_model": None
    },
    "A6_VisualOnly_ResNet50_CBAM": {
        "modality": "visual",
        "use_cbam": True,
        "imu_model": None
    },
    "A7_IMUOnly_LSTM": {
        "modality": "imu",
        "use_cbam": False,
        "imu_model": "lstm"
    },
    "A8_IMUOnly_AHLSTM": {
        "modality": "imu",
        "use_cbam": False,
        "imu_model": "ahlstm"
    }
}

# Varsayılan olarak A1-A8'in tamamı çalışır.
# Tek deney çalıştırmak için örnek:
# EXPERIMENTS_TO_RUN = ["A4_ResNet50_CBAM_AHLSTM_Fusion"]
EXPERIMENTS_TO_RUN = list(ABLATION_EXPERIMENTS.keys())


# -----------------------------------------------------------------------------
# Hiperparametreler
# -----------------------------------------------------------------------------

imu_input_size = 6
hidden_size = 100
feature_output_size = 100
position_output_size = 3
learning_rate = 0.001
num_layers = 1
dropout = 0.2
num_epochs = 250
random_seed = 42

criterion = nn.MSELoss()

ablation_root = os.path.join(
    dataset_main_file_path,
    "ablation_nonoverlap_A1_A8"
)
os.makedirs(ablation_root, exist_ok=True)


def build_ablation_model(experiment_config):
    """Deney konfigürasyonuna göre modeli kurar."""

    modality = experiment_config["modality"]
    use_cbam = experiment_config["use_cbam"]
    imu_model_type = experiment_config["imu_model"]

    cnn_model = None
    imu_model = None

    if modality in ("fusion", "visual"):
        cnn_model = CNNModel(
            output=feature_output_size,
            use_cbam=use_cbam
        )

    if modality in ("fusion", "imu"):
        if imu_model_type == "lstm":
            imu_model = LSTMModel(
                imu_input_size,
                hidden_size,
                num_layers,
                feature_output_size,
                dropout
            )
        elif imu_model_type == "ahlstm":
            imu_model = AHLSTMModel(
                imu_input_size,
                hidden_size,
                num_layers,
                feature_output_size,
                dropout,
                attention_size=hidden_size
            )
        else:
            raise ValueError(
                f"Geçersiz IMU model tipi: {imu_model_type}"
            )

    if modality == "fusion":
        model = FuseNetwork(
            cnn_model,
            imu_model,
            combined_input_size=feature_output_size * 2,
            combined_output_size=position_output_size
        )
    elif modality == "visual":
        model = VisualOnlyNetwork(
            cnn_model,
            feature_size=feature_output_size,
            output_size=position_output_size
        )
    elif modality == "imu":
        model = IMUOnlyNetwork(
            imu_model,
            feature_size=feature_output_size,
            output_size=position_output_size
        )
    else:
        raise ValueError(f"Geçersiz modality: {modality}")

    return model.to(device)


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# -----------------------------------------------------------------------------
# Eğitim ve değerlendirme
# -----------------------------------------------------------------------------

def train_model(
    model,
    train_loader,
    evaluation_loader,
    criterion,
    optimizer,
    epochs,
    experiment_name
):
    train_losses = []
    evaluation_losses = []

    for epoch in range(epochs):
        model.train()
        train_batch_losses = []

        for images, imu_sequences, targets in train_loader:
            images = images.to(device)
            imu_sequences = imu_sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(images, imu_sequences)
            loss = torch.sqrt(criterion(outputs, targets))

            loss.backward()
            optimizer.step()

            train_batch_losses.append(loss.item())

        model.eval()
        evaluation_batch_losses = []

        with torch.no_grad():
            for images, imu_sequences, targets in evaluation_loader:
                images = images.to(device)
                imu_sequences = imu_sequences.to(device)
                targets = targets.to(device)

                outputs = model(images, imu_sequences)
                evaluation_loss = torch.sqrt(
                    criterion(outputs, targets)
                )
                evaluation_batch_losses.append(
                    evaluation_loss.item()
                )

        mean_train_loss = float(np.mean(train_batch_losses))
        mean_evaluation_loss = float(
            np.mean(evaluation_batch_losses)
        )

        train_losses.append(mean_train_loss)
        evaluation_losses.append(mean_evaluation_loss)

        print(
            f"[{experiment_name}] "
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train RMSE: {mean_train_loss:.6f}, "
            f"Test RMSE: {mean_evaluation_loss:.6f}"
        )

    return train_losses, evaluation_losses


def predict_model(model, data_loader):
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for images, imu_sequences, targets in data_loader:
            images = images.to(device)
            imu_sequences = imu_sequences.to(device)
            targets = targets.to(device)

            outputs = model(images, imu_sequences)

            all_targets.append(targets.cpu())
            all_predictions.append(outputs.cpu())

    actual = torch.cat(all_targets).numpy()
    predicted = torch.cat(all_predictions).numpy()

    return actual, predicted


def calculate_metrics(actual, predicted):
    mse_value = mse(actual, predicted)
    mae_value = mae(actual, predicted)
    rmse_value = sqrt(mse_value)
    r_square_value = r2_score(actual, predicted)
    mape_value = mape(actual, predicted)

    channel_rmse = [
        sqrt(mse(actual[:, channel], predicted[:, channel]))
        for channel in range(3)
    ]

    return {
        "RMSE_X": channel_rmse[0],
        "RMSE_Y": channel_rmse[1],
        "RMSE_Z": channel_rmse[2],
        "MSE": mse_value,
        "MAE": mae_value,
        "RMSE": rmse_value,
        "R2": r_square_value,
        "MAPE": mape_value
    }


def save_experiment_outputs(
    experiment_name,
    experiment_config,
    model,
    train_losses,
    test_losses,
    actual,
    predicted,
    metrics
):
    experiment_dir = os.path.join(
        ablation_root,
        experiment_name
    )
    os.makedirs(experiment_dir, exist_ok=True)

    model_save_path = os.path.join(
        experiment_dir,
        "model.pth"
    )
    torch.save(model.state_dict(), model_save_path)

    with open(
        os.path.join(experiment_dir, "losses.pkl"),
        "wb"
    ) as file:
        pickle.dump(train_losses, file)
        pickle.dump(test_losses, file)

    pd.DataFrame(
        {
            "Epoch": range(1, len(train_losses) + 1),
            "Train_RMSE": train_losses,
            "Test_RMSE": test_losses
        }
    ).to_csv(
        os.path.join(experiment_dir, "losses.csv"),
        index=False
    )

    metric_rows = {
        "Experiment": experiment_name,
        "Modality": experiment_config["modality"],
        "CBAM": experiment_config["use_cbam"],
        "IMU_Model": experiment_config["imu_model"],
        "Trainable_Parameters": count_trainable_parameters(model),
        **metrics
    }

    pd.DataFrame(
        list(metric_rows.items()),
        columns=["Metric", "Value"]
    ).to_excel(
        os.path.join(experiment_dir, "metrics.xlsx"),
        index=False
    )

    np.save(
        os.path.join(experiment_dir, "actual.npy"),
        actual
    )
    np.save(
        os.path.join(experiment_dir, "predicted.npy"),
        predicted
    )

    # Loss grafiği
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(train_losses) + 1),
        train_losses,
        label="Train Loss"
    )
    plt.plot(
        range(1, len(test_losses) + 1),
        test_losses,
        label="Test Loss"
    )
    plt.title(f"{experiment_name} Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "loss_figure.png"),
        dpi=300
    )
    plt.close()

    # 3B konum grafiği
    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(
        actual[:, 0],
        actual[:, 1],
        actual[:, 2],
        s=5,
        label="Actual"
    )
    axis.scatter(
        predicted[:, 0],
        predicted[:, 1],
        predicted[:, 2],
        s=5,
        label="Predicted"
    )
    axis.set_title(f"{experiment_name} - 3D Position")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "3D_position_plot.png"),
        dpi=300
    )
    plt.close()

    # 2B konum grafiği
    plt.figure(figsize=(8, 8))
    plt.scatter(
        actual[:, 0],
        actual[:, 1],
        s=5,
        label="Actual"
    )
    plt.scatter(
        predicted[:, 0],
        predicted[:, 1],
        s=5,
        label="Predicted"
    )
    plt.title(f"{experiment_name} - 2D Position")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(experiment_dir, "2D_position_plot.png"),
        dpi=300
    )
    plt.close()

    return metric_rows


def save_ablation_summary(summary_rows):
    summary_df = pd.DataFrame(summary_rows)

    summary_csv_path = os.path.join(
        ablation_root,
        "ablation_summary.csv"
    )
    summary_excel_path = os.path.join(
        ablation_root,
        "ablation_summary.xlsx"
    )

    summary_df.to_csv(summary_csv_path, index=False)
    summary_df.to_excel(summary_excel_path, index=False)


# -----------------------------------------------------------------------------
# A1-A8 deneylerini sırayla çalıştır
# -----------------------------------------------------------------------------

all_experiment_results = []

print(f"Device: {device}")
print(f"Toplam dataset örneği: {len(dataset)}")
print(f"Train örneği: {len(train_dataset)}")
print(f"Test örneği: {len(test_dataset)}")
print(f"Çalıştırılacak deneyler: {EXPERIMENTS_TO_RUN}")

for experiment_name in EXPERIMENTS_TO_RUN:
    if experiment_name not in ABLATION_EXPERIMENTS:
        raise KeyError(
            f"Tanımsız deney adı: {experiment_name}"
        )

    print("\n" + "=" * 80)
    print(f"ABLATION EXPERIMENT: {experiment_name}")
    print("=" * 80)

    set_random_seed(random_seed)

    experiment_config = ABLATION_EXPERIMENTS[
        experiment_name
    ]

    model = build_ablation_model(experiment_config)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    train_losses, test_losses = train_model(
        model=model,
        train_loader=train_dataloader,
        evaluation_loader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=num_epochs,
        experiment_name=experiment_name
    )

    actual, predicted = predict_model(
        model,
        test_dataloader
    )
    metrics = calculate_metrics(
        actual,
        predicted
    )

    result_row = save_experiment_outputs(
        experiment_name=experiment_name,
        experiment_config=experiment_config,
        model=model,
        train_losses=train_losses,
        test_losses=test_losses,
        actual=actual,
        predicted=predicted,
        metrics=metrics
    )

    all_experiment_results.append(result_row)

    # Her deneyden sonra özet kaydedilir; yarıda kesilirse sonuçlar korunur.
    save_ablation_summary(all_experiment_results)

    print(f"{experiment_name} sonuçları:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")

    del model
    del optimizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\nA1-A8 ablasyon çalışmaları tamamlandı.")
print(
    pd.DataFrame(all_experiment_results)[
        [
            "Experiment",
            "Modality",
            "CBAM",
            "IMU_Model",
            "Trainable_Parameters",
            "RMSE_X",
            "RMSE_Y",
            "RMSE_Z",
            "RMSE",
            "MAE",
            "R2",
            "MAPE"
        ]
    ].to_string(index=False)
)
