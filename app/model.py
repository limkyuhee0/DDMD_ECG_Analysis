import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim    
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.random.set_seed(42)
def CNN_MODEL(tot_df,batch_size, epoch, waveform):
    total_data = tot_df

    # PatientID별로 데이터 나누기
    unique_patient_ids = total_data['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

    train_df = total_data[total_data['PatientID'].isin(train_ids)]
    test_df = total_data[total_data['PatientID'].isin(test_ids)]

    # 데이터 전처리 함수 정의
    def preprocess_data(df,waveform):
        data_list = []
        for i in range(len(df)):
            d = df[waveform].iloc[i]
            if len(d) > 12:
                keys_to_exclude = {'V3R', 'V4R', 'V7'}
                filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
            else:
                filtered_dict = d

            single_data = np.vstack(filtered_dict.values())
            if single_data.shape != (12, 5000):
                array_2500 = single_data
                x_2500 = np.linspace(0, 12, single_data.shape[1])
                x_5000 = np.linspace(0, 12, 5000)
                linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
                array_5000 = linear_interpolator(x_5000)
                data_list.append(array_5000)
            else:
                data_list.append(single_data)
        return np.array(data_list)

    # 데이터 전처리 수행
    X_train = preprocess_data(train_df)
    y_train = np.array(train_df['Label'])
    X_test = preprocess_data(test_df)
    y_test = np.array(test_df['Label'])
    print(X_train.shape)
    print(y_train.shape)

    # 데이터 정규화 (각 채널별로)
    scaler = StandardScaler()

    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.reshape(-1, 12, 5000)

    X_test = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(-1, 12, 5000)

    # Convert numpy arrays to PyTorch tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # PyTorch CNN Model Definition
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(256 * 625, 256)
            self.fc2 = nn.Linear(256, 1)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(self.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 256 * 625)
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x).squeeze(-1)
            return x

    # Initialize the model, define the loss function and the optimizer
    model = CNNModel().to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = epoch
    batch_size = batch_size

    # Create DataLoader
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            # outputs = outputs.squeeze()  # Remove extra dimensions
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    print("Training complete.")

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_outputs = test_outputs.squeeze()  # Remove extra dimensions

    # Convert logits to probabilities and then to binary predictions
    test_probabilities = torch.sigmoid(test_outputs)
    test_predictions = torch.round(test_probabilities)

    # Convert PyTorch tensors to numpy arrays
    test_predictions_np = test_predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Compute accuracy
    test_accuracy = accuracy_score(y_test_np, test_predictions_np)

    # Compute ROC AUC score
    test_roc_auc = roc_auc_score(y_test_np, test_probabilities.cpu().numpy())

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test ROC AUC Score: {test_roc_auc:.4f}')




    # lstm_input_size = 12  # Number of features (channels)
    # hidden_size = 64  # Number of LSTM units
    # lstm_output_size = 32  # Embedding size
    # num_epochs = 20
    
    # batch_size = 8  # Batch size for DataLoader
    # num_epochs_cnn = 100  # Number of epochs for CNN training

def CNN_MODEL_TABULAR_ADDED(tot_df, batch_size, epoch, waveform, tabular_features):
    total_data = tot_df

    # PatientID별로 데이터 나누기
    unique_patient_ids = total_data['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

    train_df = total_data[total_data['PatientID'].isin(train_ids)]
    test_df = total_data[total_data['PatientID'].isin(test_ids)]

    # 데이터 전처리 함수 정의
    def preprocess_data(df):
        data_list = []
        tabular_data_list = []
        for i in range(len(df)):
            d = df[waveform].iloc[i]
            if len(d) > 12:
                keys_to_exclude = {'V3R', 'V4R', 'V7'}
                filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
            else:
                filtered_dict = d

            single_data = np.vstack(filtered_dict.values())
            if single_data.shape != (12, 5000):
                array_2500 = single_data
                x_2500 = np.linspace(0, 12, single_data.shape[1])
                x_5000 = np.linspace(0, 12, 5000)
                linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
                array_5000 = linear_interpolator(x_5000)
                data_list.append(array_5000)
            else:
                data_list.append(single_data)
            tabular_data_list.append(df[tabular_features].iloc[i].values)
        return np.array(data_list), np.array(tabular_data_list)

    # 데이터 전처리 수행
    X_train, X_train_tabular = preprocess_data(train_df)
    y_train = np.array(train_df['Label'])
    X_test, X_test_tabular = preprocess_data(test_df)
    y_test = np.array(test_df['Label'])
    print(X_train.shape)
    print(y_train.shape)

    # 데이터 정규화 (각 채널별로)
    scaler = StandardScaler()

    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.reshape(-1, 12, 5000)

    X_test = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(-1, 12, 5000)

    # Tabular 데이터 정규화
    tabular_scaler = StandardScaler()
    X_train_tabular = tabular_scaler.fit_transform(X_train_tabular)
    X_test_tabular = tabular_scaler.transform(X_test_tabular)

    # Convert numpy arrays to PyTorch tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_train_tabular = torch.tensor(X_train_tabular, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_test_tabular = torch.tensor(X_test_tabular, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # PyTorch CNN Model Definition
    class CNNModel(nn.Module):
        def __init__(self, num_tabular_features):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(256 * 625, 256)
            self.fc2 = nn.Linear(256 + num_tabular_features, 1)  # Combining CNN output and tabular data
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
            
        def forward(self, x, tabular_data):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(self.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, 256 * 625)
            x = self.dropout(self.relu(self.fc1(x)))
            combined = torch.cat((x, tabular_data), dim=1)  # Concatenate CNN output and tabular data
            x = self.fc2(combined).squeeze(-1)
            return x

    # Initialize the model, define the loss function and the optimizer
    model = CNNModel(num_tabular_features=len(tabular_features)).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = epoch
    batch_size = batch_size

    # Create DataLoader
    train_data = torch.utils.data.TensorDataset(X_train, X_train_tabular, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        train_loss = 0.0
        for inputs, tabular_data, labels in train_loader:
            inputs, tabular_data, labels = inputs.to(device), tabular_data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs, tabular_data)
            # outputs = outputs.squeeze()  # Remove extra dimensions
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    print("Training complete.")

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test, X_test_tabular)
        test_outputs = test_outputs.squeeze()  # Remove extra dimensions

    # Convert logits to probabilities and then to binary predictions
    test_probabilities = torch.sigmoid(test_outputs)
    test_predictions = torch.round(test_probabilities)

    # Convert PyTorch tensors to numpy arrays
    test_predictions_np = test_predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # Compute accuracy
    test_accuracy = accuracy_score(y_test_np, test_predictions_np)

    # Compute ROC AUC score
    test_roc_auc = roc_auc_score(y_test_np, test_probabilities.cpu().numpy())

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test ROC AUC Score: {test_roc_auc:.4f}')


# def LSTM_CNN_MODEL(tot_df, waveform, lstm_input_size, hidden_size, lstm_output_size,num_epochs,batch_size,num_epochs_cnn):

#     total_data = tot_df

#     # Split data by PatientID
#     unique_patient_ids = total_data['PatientID'].unique()
#     train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

#     train_df = total_data[total_data['PatientID'].isin(train_ids)]
#     test_df = total_data[total_data['PatientID'].isin(test_ids)]

#     # Data preprocessing function
#     def preprocess_data(df):
#         data_dict = {}
#         labels = {}

#         for patient_id in df['PatientID'].unique():
#             patient_data = df[df['PatientID'] == patient_id]
#             data_list = []

#             for i in range(len(patient_data)):
#                 d = patient_data[waveform].iloc[i]

#                 if len(d) > 12:
#                     keys_to_exclude = {'V3R', 'V4R', 'V7'}
#                     filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
#                 else:
#                     filtered_dict = d

#                 single_data = np.vstack(filtered_dict.values())
#                 if single_data.shape != (12, 5000):
#                     array_2500 = single_data
#                     x_2500 = np.linspace(0, 12, single_data.shape[1])
#                     x_5000 = np.linspace(0, 12, 5000)
#                     linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
#                     array_5000 = linear_interpolator(x_5000)
#                     data_list.append(array_5000.astype(np.float32))  # Ensure data is float32
#                 else:
#                     data_list.append(single_data.astype(np.float32))  # Ensure data is float32

#             data_dict[patient_id] = np.array(data_list, dtype=np.float32)
#             labels[patient_id] = patient_data['Label'].values[0]

#         return data_dict, labels

#     # Preprocess data
#     train_data, train_labels = preprocess_data(train_df)
#     test_data, test_labels = preprocess_data(test_df)

#     # Normalize data (channel-wise)
#     scaler = StandardScaler()

#     for patient_id in train_data.keys():
#         num_ecgs = train_data[patient_id].shape[0]
#         train_data[patient_id] = train_data[patient_id].reshape(-1, train_data[patient_id].shape[-1])
#         train_data[patient_id] = scaler.fit_transform(train_data[patient_id])
#         train_data[patient_id] = train_data[patient_id].reshape(num_ecgs, 5000, 12)

#     for patient_id in test_data.keys():
#         num_ecgs = test_data[patient_id].shape[0]
#         test_data[patient_id] = test_data[patient_id].reshape(-1, test_data[patient_id].shape[-1])
#         test_data[patient_id] = scaler.transform(test_data[patient_id])
#         test_data[patient_id] = test_data[patient_id].reshape(num_ecgs, 5000, 12)

#     # Define a custom dataset for loading ECG data
#     class ECGDataset(Dataset):
#         def __init__(self, data_dict, labels):
#             self.data_dict = data_dict
#             self.labels = labels
#             self.patient_ids = list(data_dict.keys())

#         def __len__(self):
#             return len(self.patient_ids)

#         def __getitem__(self, idx):
#             patient_id = self.patient_ids[idx]
#             ecgs = self.data_dict[patient_id]
#             label = self.labels[patient_id]
#             return torch.tensor(ecgs, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#     # Create DataLoader
#     lstm_batch_size = 1
#     train_dataset = ECGDataset(train_data, train_labels)
#     train_loader = DataLoader(train_dataset, batch_size=lstm_batch_size, shuffle=True)

#     test_dataset = ECGDataset(test_data, test_labels)
#     test_loader = DataLoader(test_dataset, batch_size=lstm_batch_size, shuffle=False)

#     # Define the LSTM model for embedding
#     class LSTMEmbedder(nn.Module):
#         def __init__(self, input_size, hidden_size, output_size):
#             super(LSTMEmbedder, self).__init__()
#             self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#             self.fc = nn.Linear(hidden_size, output_size)

#         def forward(self, x):
#             lstm_out, _ = self.lstm(x)
#             lstm_out = lstm_out[:, -1, :]  # Use the output of the last time step
#             embedding = self.fc(lstm_out)
#             return embedding

#     # Initialize model


#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     lstm_embedder = LSTMEmbedder(lstm_input_size, hidden_size, lstm_output_size).to(device)

#     # Define loss and optimizer
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(lstm_embedder.parameters(), lr=0.001)

#     # Train the LSTM model
    
#     for epoch in range(num_epochs):
#         lstm_embedder.train()

#         for X_batch, y_batch in train_loader:
#             X_batch = X_batch.squeeze(0).to(device)  # Remove batch dimension
#             y_batch = y_batch.to(device)
#             optimizer.zero_grad()
            
#             embeddings = lstm_embedder(X_batch)
#             outputs = embeddings.mean(dim=0)  # Mean of all embeddings for the patient
            
#             loss = criterion(outputs.unsqueeze(0), y_batch)
#             loss.backward()
#             optimizer.step()

#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

#     # Extract embeddings for the training and test data
#     def extract_embeddings(data_loader, model):
#         embeddings = []
#         labels = []
#         model.eval()
#         with torch.no_grad():
#             for X_batch, y_batch in data_loader:
#                 X_batch = X_batch.squeeze(0).to(device)
#                 y_batch = y_batch.to(device)
#                 patient_embeddings = model(X_batch)
#                 patient_embedding = patient_embeddings.mean(dim=0)
#                 embeddings.append(patient_embedding.cpu().numpy())
#                 labels.append(y_batch.cpu().numpy())
#         return np.array(embeddings, dtype=np.float32), np.array(labels)

#     train_embeddings, train_labels_array = extract_embeddings(train_loader, lstm_embedder)
#     test_embeddings, test_labels_array = extract_embeddings(test_loader, lstm_embedder)

#     # Print the shape of the embeddings and labels
#     print(f'Train embeddings shape: {train_embeddings.shape}')
#     print(f'Train labels shape: {train_labels_array.shape}')
#     print(f'Test embeddings shape: {test_embeddings.shape}')
#     print(f'Test labels shape: {test_labels_array.shape}')



#     # 설정 변수
#     input_size = lstm_output_size  # Embedding size from LSTM
#     batch_size = batch_size
#     num_epochs_cnn = num_epochs_cnn
#     # Define the CNN model for prediction
#     class ComplexCNNModel(nn.Module):
#         def __init__(self):
#             super(ComplexCNNModel, self).__init__()
#             self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
#             self.bn1 = nn.BatchNorm1d(64)
#             self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
#             self.bn2 = nn.BatchNorm1d(128)
#             self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#             self.bn3 = nn.BatchNorm1d(256)
#             self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#             self.fc1 = nn.Linear(256 * (input_size // 8), 256)  # Adjust based on the new input size
#             self.fc2 = nn.Linear(256, 1)
#             self.dropout = nn.Dropout(0.5)
#             self.relu = nn.ReLU()
            
#         def forward(self, x):
#             x = self.pool(self.relu(self.bn1(self.conv1(x))))
#             x = self.pool(self.relu(self.bn2(self.conv2(x))))
#             x = self.pool(self.relu(self.bn3(self.conv3(x))))
#             x = x.view(x.size(0), -1)  # Use x.size(0) to ensure the correct batch size
#             x = self.dropout(self.relu(self.fc1(x)))
#             x = self.fc2(x)
#             return x

#     # Initialize the CNN model
#     cnn_model = ComplexCNNModel().to(device)

#     # Define loss and optimizer
#     cnn_criterion = nn.BCEWithLogitsLoss()
#     cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

#     # Convert embeddings and labels to PyTorch tensors and create DataLoader
#     train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
#     train_labels_tensor = torch.tensor(train_labels_array.squeeze(), dtype=torch.float32).to(device)  # Ensure labels are in the correct shape and type
#     test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
#     test_labels_tensor = torch.tensor(test_labels_array.squeeze(), dtype=torch.float32).to(device)  # Ensure labels are in the correct shape and type

#     train_embed_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
#     train_embed_loader = DataLoader(train_embed_dataset, batch_size=batch_size, shuffle=True)

#     test_embed_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
#     test_embed_loader = DataLoader(test_embed_dataset, batch_size=batch_size, shuffle=False)

#     # Train the CNN model
#     for epoch in range(num_epochs_cnn):
#         cnn_model.train()

#         for X_batch, y_batch in train_embed_loader:
#             cnn_optimizer.zero_grad()
            
#             outputs = cnn_model(X_batch)  # Input already has channel dimension
#             loss = cnn_criterion(outputs.squeeze(), y_batch)  # Adjust the shape for BCEWithLogitsLoss
#             loss.backward()
#             cnn_optimizer.step()

#         if (epoch + 1) % 10 == 0:
#             print(f'CNN Epoch {epoch+1}/{num_epochs_cnn}, Loss: {loss.item():.3f}')

#     # Evaluate the CNN model
#     cnn_model.eval()
#     with torch.no_grad():
#         train_preds = []
#         train_labels = []
#         for X_batch, y_batch in train_embed_loader:
#             train_outputs = cnn_model(X_batch)  # Input already has channel dimension
#             train_preds.append(train_outputs)
#             train_labels.append(y_batch)
        
#         train_preds = torch.cat(train_preds)
#         train_labels = torch.cat(train_labels)
#         train_pred = torch.sigmoid(train_preds).cpu().numpy() > 0.5  # Use sigmoid for binary classification
#         train_accuracy = accuracy_score(train_labels.cpu(), train_pred)
#         train_roc_auc = roc_auc_score(train_labels.cpu(), torch.sigmoid(train_preds).cpu())

#         test_preds = []
#         test_labels = []
#         for X_batch, y_batch in test_embed_loader:
#             test_outputs = cnn_model(X_batch)  # Input already has channel dimension
#             test_preds.append(test_outputs)
#             test_labels.append(y_batch)
        
#         test_preds = torch.cat(test_preds)
#         test_labels = torch.cat(test_labels)
#         test_pred = torch.sigmoid(test_preds).cpu().numpy() > 0.5  # Use sigmoid for binary classification
#         test_accuracy = accuracy_score(test_labels.cpu(), test_pred)
#         test_roc_auc = roc_auc_score(test_labels.cpu(), torch.sigmoid(test_preds).cpu())

#     print(f'Train Accuracy: {train_accuracy:.3f}, Train ROC-AUC: {train_roc_auc:.3f}')
#     print(f'Test Accuracy: {test_accuracy:.3f}, Test ROC-AUC: {test_roc_auc:.3f}')

def LSTM_CNN_MODEL(tot_df, waveform, lstm_input_size, hidden_size, lstm_output_size, num_epochs, batch_size, num_epochs_cnn):

    total_data = tot_df

    # Split data by PatientID
    unique_patient_ids = total_data['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

    train_df = total_data[total_data['PatientID'].isin(train_ids)]
    test_df = total_data[total_data['PatientID'].isin(test_ids)]

    # Data preprocessing function
    def preprocess_data(df):
        data_dict = {}
        labels = {}

        for patient_id in df['PatientID'].unique():
            patient_data = df[df['PatientID'] == patient_id]
            data_list = []

            for i in range(len(patient_data)):
                d = patient_data[waveform].iloc[i]

                if len(d) > 12:
                    keys_to_exclude = {'V3R', 'V4R', 'V7'}
                    filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
                else:
                    filtered_dict = d

                single_data = np.vstack(filtered_dict.values())
                if single_data.shape != (12, 5000):
                    array_2500 = single_data
                    x_2500 = np.linspace(0, 12, single_data.shape[1])
                    x_5000 = np.linspace(0, 12, 5000)
                    linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
                    array_5000 = linear_interpolator(x_5000)
                    data_list.append(array_5000.astype(np.float32))  # Ensure data is float32
                else:
                    data_list.append(single_data.astype(np.float32))  # Ensure data is float32

            data_dict[patient_id] = np.array(data_list, dtype=np.float32)
            labels[patient_id] = patient_data['Label'].values[0]

        return data_dict, labels

    # Preprocess data
    train_data, train_labels = preprocess_data(train_df)
    test_data, test_labels = preprocess_data(test_df)

    # Normalize data (channel-wise)
    scaler = StandardScaler()

    for patient_id in train_data.keys():
        num_ecgs = train_data[patient_id].shape[0]
        train_data[patient_id] = train_data[patient_id].reshape(-1, train_data[patient_id].shape[-1])
        train_data[patient_id] = scaler.fit_transform(train_data[patient_id])
        train_data[patient_id] = train_data[patient_id].reshape(num_ecgs, 5000, 12)

    for patient_id in test_data.keys():
        num_ecgs = test_data[patient_id].shape[0]
        test_data[patient_id] = test_data[patient_id].reshape(-1, test_data[patient_id].shape[-1])
        test_data[patient_id] = scaler.transform(test_data[patient_id])
        test_data[patient_id] = test_data[patient_id].reshape(num_ecgs, 5000, 12)

    # Define a custom dataset for loading ECG data
    class ECGDataset(Dataset):
        def __init__(self, data_dict, labels):
            self.data_dict = data_dict
            self.labels = labels
            self.patient_ids = list(data_dict.keys())

        def __len__(self):
            return len(self.patient_ids)

        def __getitem__(self, idx):
            patient_id = self.patient_ids[idx]
            ecgs = self.data_dict[patient_id]
            label = self.labels[patient_id]
            return torch.tensor(ecgs, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    # Create DataLoader
    lstm_batch_size = 1
    train_dataset = ECGDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=lstm_batch_size, shuffle=True)

    test_dataset = ECGDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=lstm_batch_size, shuffle=False)

    # Define the LSTM model for embedding
    class LSTMEmbedder(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMEmbedder, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = lstm_out[:, -1, :]  # Use the output of the last time step
            embedding = self.fc(lstm_out)
            return embedding

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_embedder = LSTMEmbedder(lstm_input_size, hidden_size, lstm_output_size).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lstm_embedder.parameters(), lr=0.001)

    # Train the LSTM model
    for epoch in range(num_epochs):
        lstm_embedder.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.squeeze(0).to(device)  # Remove batch dimension
            y_batch = y_batch.to(device)
            optimizer.zero_grad()

            embeddings = lstm_embedder(X_batch)
            outputs = embeddings.mean(dim=0)  # Mean of all embeddings for the patient

            loss = criterion(outputs.unsqueeze(0), y_batch)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Extract embeddings for the training and test data
    def extract_embeddings(data_loader, model):
        embeddings = []
        labels = []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.squeeze(0).to(device)
                y_batch = y_batch.to(device)
                patient_embeddings = model(X_batch)
                patient_embedding = patient_embeddings.mean(dim=0)
                embeddings.append(patient_embedding.cpu().numpy())
                labels.append(y_batch.cpu().numpy())
        return np.array(embeddings, dtype=np.float32), np.array(labels)

    train_embeddings, train_labels_array = extract_embeddings(train_loader, lstm_embedder)
    test_embeddings, test_labels_array = extract_embeddings(test_loader, lstm_embedder)

    # Print the shape of the embeddings and labels
    print(f'Train embeddings shape: {train_embeddings.shape}')
    print(f'Train labels shape: {train_labels_array.shape}')
    print(f'Test embeddings shape: {test_embeddings.shape}')
    print(f'Test labels shape: {test_labels_array.shape}')

    # 설정 변수
    input_size = lstm_output_size  # Embedding size from LSTM
    batch_size = batch_size
    num_epochs_cnn = num_epochs_cnn

    # Define the CNN model for prediction
    class ComplexCNNModel(nn.Module):
        def __init__(self):
            super(ComplexCNNModel, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
            self.bn2 = nn.BatchNorm1d(128)
            self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(256)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(256 * (input_size // 8), 256)  # Adjust based on the new input size
            self.fc2 = nn.Linear(256, 1)
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(self.relu(self.bn3(self.conv3(x))))
            x = x.view(x.size(0), -1)  # Use x.size(0) to ensure the correct batch size
            x = self.dropout(self.relu(self.fc1(x)))
            x = self.fc2(x)
            return x

    # Initialize the CNN model
    cnn_model = ComplexCNNModel().to(device)

    # Define loss and optimizer
    cnn_criterion = nn.BCEWithLogitsLoss()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    # Convert embeddings and labels to PyTorch tensors and create DataLoader
    train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    train_labels_tensor = torch.tensor(train_labels_array.squeeze(), dtype=torch.float32).to(device)  # Ensure labels are in the correct shape and type
    test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).unsqueeze(1).to(device)  # Add channel dimension
    test_labels_tensor = torch.tensor(test_labels_array.squeeze(), dtype=torch.float32).to(device)  # Ensure labels are in the correct shape and type

    train_embed_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
    train_embed_loader = DataLoader(train_embed_dataset, batch_size=batch_size, shuffle=True)

    test_embed_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
    test_embed_loader = DataLoader(test_embed_dataset, batch_size=batch_size, shuffle=False)

    # Train the CNN model
    for epoch in range(num_epochs_cnn):
        cnn_model.train()

        for X_batch, y_batch in train_embed_loader:
            cnn_optimizer.zero_grad()

            outputs = cnn_model(X_batch).squeeze(1)  # 모델 출력의 크기를 [batch_size]로 변경

            if y_batch.dim() > 1:  # 타겟 텐서가 2차원인 경우
                y_batch = y_batch.squeeze(1)  # 타겟 텐서를 1차원으로 변환

            loss = cnn_criterion(outputs, y_batch)  # 손실 계산
            loss.backward()
            cnn_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'CNN Epoch {epoch+1}/{num_epochs_cnn}, Loss: {loss.item():.3f}')

    # Evaluate the CNN model
    cnn_model.eval()
    with torch.no_grad():
        train_preds = []
        train_labels = []
        for X_batch, y_batch in train_embed_loader:
            train_outputs = cnn_model(X_batch).squeeze(1)
            train_preds.append(train_outputs)
            train_labels.append(y_batch)

        train_preds = torch.cat(train_preds)
        train_labels = torch.cat(train_labels)
        train_pred = torch.sigmoid(train_preds).cpu().numpy() > 0.5  # Use sigmoid for binary classification
        train_accuracy = accuracy_score(train_labels.cpu(), train_pred)
        train_roc_auc = roc_auc_score(train_labels.cpu(), torch.sigmoid(train_preds).cpu())

        test_preds = []
        test_labels = []
        for X_batch, y_batch in test_embed_loader:
            test_outputs = cnn_model(X_batch).squeeze(1)
            test_preds.append(test_outputs)
            test_labels.append(y_batch)

        test_preds = torch.cat(test_preds)
        test_labels = torch.cat(test_labels)
        test_pred = torch.sigmoid(test_preds).cpu().numpy() > 0.5  # Use sigmoid for binary classification
        test_accuracy = accuracy_score(test_labels.cpu(), test_pred)
        test_roc_auc = roc_auc_score(test_labels.cpu(), torch.sigmoid(test_preds).cpu())

    print(f'Train Accuracy: {train_accuracy:.3f}, Train ROC-AUC: {train_roc_auc:.3f}')
    print(f'Test Accuracy: {test_accuracy:.3f}, Test ROC-AUC: {test_roc_auc:.3f}')


# class LSTM_CNN_MODEL_TABULAR_ADDED:
#     def __init__(self, tot_df, waveform, lstm_input_size, hidden_size, lstm_output_size, mlp_input_size, mlp_hidden_size, mlp_output_size, num_epochs, batch_size, num_epochs_cnn, tabular_features):
#         self.tot_df = tot_df
#         self.waveform = waveform
#         self.lstm_input_size = lstm_input_size
#         self.hidden_size = hidden_size
#         self.lstm_output_size = lstm_output_size
#         self.mlp_input_size = mlp_input_size
#         self.mlp_hidden_size = mlp_hidden_size
#         self.mlp_output_size = mlp_output_size
#         self.num_epochs = num_epochs
#         self.batch_size = batch_size
#         self.num_epochs_cnn = num_epochs_cnn
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         # self.tabular_features = ['Age', 'Gender', 'scoliosis OP', 'Vent', 'amiodarone', 'Ivabradine', 'carvedilol', 'other betablocker', 'digoxin', 'Entresto']
#         self.tabular_features = tabular_features

#     def preprocess_data(self, df):
#         data_dict = {}
#         labels = {}
#         tabular_data_dict = {}

#         for patient_id in df['PatientID'].unique():
#             patient_data = df[df['PatientID'] == patient_id]
#             data_list = []

#             for i in range(len(patient_data)):
#                 d = patient_data[self.waveform].iloc[i]

#                 if len(d) > 12:
#                     keys_to_exclude = {'V3R', 'V4R', 'V7'}
#                     filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
#                 else:
#                     filtered_dict = d

#                 single_data = np.vstack(filtered_dict.values())
#                 if single_data.shape != (12, 5000):
#                     array_2500 = single_data
#                     x_2500 = np.linspace(0, 12, single_data.shape[1])
#                     x_5000 = np.linspace(0, 12, 5000)
#                     linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
#                     array_5000 = linear_interpolator(x_5000)
#                     data_list.append(array_5000.astype(np.float32))  # Ensure data is float32
#                 else:
#                     data_list.append(single_data.astype(np.float32))  # Ensure data is float32

#             data_dict[patient_id] = np.array(data_list, dtype=np.float32)
#             tabular_data_dict[patient_id] = patient_data[self.tabular_features].values[0].astype(np.float32)
#             labels[patient_id] = patient_data['Label'].values[0]

#         return data_dict, tabular_data_dict, labels

#     def prepare_data(self):
#         total_data = self.tot_df
#         unique_patient_ids = total_data['PatientID'].unique()
#         train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

#         train_df = total_data[total_data['PatientID'].isin(train_ids)]
#         test_df = total_data[total_data['PatientID'].isin(test_ids)]

#         train_data, train_tabular_data, train_labels = self.preprocess_data(train_df)
#         test_data, test_tabular_data, test_labels = self.preprocess_data(test_df)

#         scaler = StandardScaler()

#         for patient_id in train_data.keys():
#             num_ecgs = train_data[patient_id].shape[0]
#             train_data[patient_id] = train_data[patient_id].reshape(-1, train_data[patient_id].shape[-1])
#             train_data[patient_id] = scaler.fit_transform(train_data[patient_id])
#             train_data[patient_id] = train_data[patient_id].reshape(num_ecgs, 5000, 12)

#         for patient_id in test_data.keys():
#             num_ecgs = test_data[patient_id].shape[0]
#             test_data[patient_id] = test_data[patient_id].reshape(-1, test_data[patient_id].shape[-1])
#             test_data[patient_id] = scaler.transform(test_data[patient_id])
#             test_data[patient_id] = test_data[patient_id].reshape(num_ecgs, 5000, 12)

#         self.train_data = train_data
#         self.train_tabular_data = train_tabular_data
#         self.train_labels = train_labels
#         self.test_data = test_data
#         self.test_tabular_data = test_tabular_data
#         self.test_labels = test_labels

#     def create_loaders(self):
#         class ECGDataset(Dataset):
#             def __init__(self, data_dict, tabular_data_dict, labels):
#                 self.data_dict = data_dict
#                 self.tabular_data_dict = tabular_data_dict
#                 self.labels = labels
#                 self.patient_ids = list(data_dict.keys())

#             def __len__(self):
#                 return len(self.patient_ids)

#             def __getitem__(self, idx):
#                 patient_id = self.patient_ids[idx]
#                 ecgs = self.data_dict[patient_id]
#                 tabular_data = self.tabular_data_dict[patient_id]
#                 label = self.labels[patient_id]
#                 return torch.tensor(ecgs, dtype=torch.float32), torch.tensor(tabular_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

#         lstm_batch_size = 1
#         train_dataset = ECGDataset(self.train_data, self.train_tabular_data, self.train_labels)
#         train_loader = DataLoader(train_dataset, batch_size=lstm_batch_size, shuffle=True)

#         test_dataset = ECGDataset(self.test_data, self.test_tabular_data, self.test_labels)
#         test_loader = DataLoader(test_dataset, batch_size=lstm_batch_size, shuffle=False)

#         self.train_loader = train_loader
#         self.test_loader = test_loader

#     def train_lstm(self):
#         class LSTMEmbedder(nn.Module):
#             def __init__(self, input_size, hidden_size, output_size):
#                 super(LSTMEmbedder, self).__init__()
#                 self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#                 self.fc = nn.Linear(hidden_size, output_size)

#             def forward(self, x):
#                 lstm_out, _ = self.lstm(x)
#                 lstm_out = lstm_out[:, -1, :]  # Use the output of the last time step
#                 embedding = self.fc(lstm_out)
#                 return embedding

#         lstm_embedder = LSTMEmbedder(self.lstm_input_size, self.hidden_size, self.lstm_output_size).to(self.device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(lstm_embedder.parameters(), lr=0.001)

#         for epoch in range(self.num_epochs):
#             lstm_embedder.train()

#             for X_batch, _, y_batch in self.train_loader:
#                 X_batch = X_batch.squeeze(0).to(self.device)  # Remove batch dimension
#                 y_batch = y_batch.to(self.device)
#                 optimizer.zero_grad()

#                 embeddings = lstm_embedder(X_batch)
#                 outputs = embeddings.mean(dim=0)  # Mean of all embeddings for the patient

#                 loss = criterion(outputs.unsqueeze(0), y_batch)
#                 loss.backward()
#                 optimizer.step()

#             print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')

#         self.lstm_embedder = lstm_embedder

#     def extract_embeddings(self, data_loader, model):
#         embeddings = []
#         labels = []
#         model.eval()
#         with torch.no_grad():
#             for X_batch, _, y_batch in data_loader:
#                 X_batch = X_batch.squeeze(0).to(self.device)
#                 y_batch = y_batch.to(self.device)
#                 patient_embeddings = model(X_batch)
#                 patient_embedding = patient_embeddings.mean(dim=0)
#                 embeddings.append(patient_embedding.cpu().numpy())
#                 labels.append(y_batch.cpu().numpy())
#         return np.array(embeddings, dtype=np.float32), np.array(labels)

#     def train_mlp(self):
#         class MLP(nn.Module):
#             def __init__(self, input_size, hidden_size, output_size):
#                 super(MLP, self).__init__()
#                 self.fc1 = nn.Linear(input_size, hidden_size)
#                 self.relu = nn.ReLU()
#                 self.fc2 = nn.Linear(hidden_size, output_size)

#             def forward(self, x):
#                 x = self.relu(self.fc1(x))
#                 x = self.fc2(x)
#                 return x

#         mlp = MLP(self.mlp_input_size, self.mlp_hidden_size, self.mlp_output_size).to(self.device)
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.Adam(mlp.parameters(), lr=0.001)

#         tabular_data = np.array([self.train_tabular_data[pid] for pid in self.train_data.keys()], dtype=np.float32)
#         labels = np.array([self.train_labels[pid] for pid in self.train_data.keys()], dtype=np.int64)

#         tabular_data_tensor = torch.tensor(tabular_data, dtype=torch.float32).to(self.device)
#         labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

#         dataset = TensorDataset(tabular_data_tensor, labels_tensor)
#         data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

#         for epoch in range(self.num_epochs):
#             mlp.train()
#             for tabular_batch, labels_batch in data_loader:
#                 tabular_batch = tabular_batch.to(self.device)
#                 labels_batch = labels_batch.to(self.device)
#                 optimizer.zero_grad()

#                 outputs = mlp(tabular_batch)
#                 loss = criterion(outputs, labels_batch)
#                 loss.backward()
#                 optimizer.step()

#             print(f'MLP Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')

#         self.mlp = mlp

#     def extract_mlp_embeddings(self, data_loader, model):
#         embeddings = []
#         labels = []
#         model.eval()
#         with torch.no_grad():
#             for _, tabular_batch, y_batch in data_loader:
#                 tabular_batch = tabular_batch.to(self.device)
#                 y_batch = y_batch.to(self.device)
#                 embedding = model(tabular_batch)
#                 embeddings.append(embedding.cpu().numpy())
#                 labels.append(y_batch.cpu().numpy())
#         return np.array(embeddings, dtype=np.float32), np.array(labels)

#     def train_cnn(self):
#         input_size = self.lstm_output_size + self.mlp_output_size

#         class ComplexCNNModel(nn.Module):
#             def __init__(self):
#                 super(ComplexCNNModel, self).__init__()
#                 self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
#                 self.bn1 = nn.BatchNorm1d(64)
#                 self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
#                 self.bn2 = nn.BatchNorm1d(128)
#                 self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
#                 self.bn3 = nn.BatchNorm1d(256)
#                 self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#                 self.fc1 = nn.Linear(256 * (input_size // 8), 256)  # Adjust based on the new input size
#                 self.fc2 = nn.Linear(256, 1)
#                 self.dropout = nn.Dropout(0.5)
#                 self.relu = nn.ReLU()

#             def forward(self, x):
#                 x = self.pool(self.relu(self.bn1(self.conv1(x))))
#                 x = self.pool(self.relu(self.bn2(self.conv2(x))))
#                 x = self.pool(self.relu(self.bn3(self.conv3(x))))
#                 x = x.view(x.size(0), -1)  # Use x.size(0) to ensure the correct batch size
#                 x = self.dropout(self.relu(self.fc1(x)))
#                 x = self.fc2(x)
#                 return x

#         cnn_model = ComplexCNNModel().to(self.device)
#         cnn_criterion = nn.BCEWithLogitsLoss()
#         cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

#         train_lstm_embeddings, _ = self.extract_embeddings(self.train_loader, self.lstm_embedder)
#         test_lstm_embeddings, _ = self.extract_embeddings(self.test_loader, self.lstm_embedder)
#         train_mlp_embeddings, train_labels_array = self.extract_mlp_embeddings(self.train_loader, self.mlp)
#         test_mlp_embeddings, test_labels_array = self.extract_mlp_embeddings(self.test_loader, self.mlp)

#         # Ensure the MLP embeddings are 2D
#         train_mlp_embeddings = train_mlp_embeddings.reshape(train_mlp_embeddings.shape[0], -1)
#         test_mlp_embeddings = test_mlp_embeddings.reshape(test_mlp_embeddings.shape[0], -1)

#         train_embeddings = np.concatenate([train_lstm_embeddings, train_mlp_embeddings], axis=1)
#         test_embeddings = np.concatenate([test_lstm_embeddings, test_mlp_embeddings], axis=1)

#         train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add channel dimension
#         train_labels_tensor = torch.tensor(train_labels_array.squeeze(), dtype=torch.float32).to(self.device)  # Ensure labels are in the correct shape and type
#         test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).unsqueeze(1).to(self.device)  # Add channel dimension
#         test_labels_tensor = torch.tensor(test_labels_array.squeeze(), dtype=torch.float32).to(self.device)  # Ensure labels are in the correct shape and type

#         train_embed_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
#         train_embed_loader = DataLoader(train_embed_dataset, batch_size=self.batch_size, shuffle=True)

#         test_embed_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
#         test_embed_loader = DataLoader(test_embed_dataset, batch_size=self.batch_size, shuffle=False)

#         for epoch in range(self.num_epochs_cnn):
#             cnn_model.train()

#             for X_batch, y_batch in train_embed_loader:
#                 cnn_optimizer.zero_grad()

#                 outputs = cnn_model(X_batch)  # Input already has channel dimension
#                 loss = cnn_criterion(outputs.squeeze(), y_batch)  # Adjust the shape for BCEWithLogitsLoss
#                 loss.backward()
#                 cnn_optimizer.step()

#             if (epoch + 1) % 10 == 0:
#                 print(f'CNN Epoch {epoch+1}/{self.num_epochs_cnn}, Loss: {loss.item():.3f}')

#         self.cnn_model = cnn_model
#         self.train_embed_loader = train_embed_loader
#         self.test_embed_loader = test_embed_loader

#     def evaluate(self):
#         self.cnn_model.eval()
#         with torch.no_grad():
#             train_preds = []
#             train_labels = []
#             for X_batch, y_batch in self.train_embed_loader:
#                 train_outputs = self.cnn_model(X_batch)  # Input already has channel dimension
#                 train_preds.append(train_outputs)
#                 train_labels.append(y_batch)

#             train_preds = torch.cat(train_preds)
#             train_labels = torch.cat(train_labels)
#             train_pred = torch.sigmoid(train_preds).cpu().numpy() > 0.5  # Use sigmoid for binary classification
#             train_accuracy = accuracy_score(train_labels.cpu(), train_pred)
#             train_roc_auc = roc_auc_score(train_labels.cpu(), torch.sigmoid(train_preds).cpu())

#             test_preds = []
#             test_labels = []
#             for X_batch, y_batch in self.test_embed_loader:
#                 test_outputs = self.cnn_model(X_batch)  # Input already has channel dimension
#                 test_preds.append(test_outputs)
#                 test_labels.append(y_batch)

#             test_preds = torch.cat(test_preds)
#             test_labels = torch.cat(test_labels)
#             test_pred = torch.sigmoid(test_preds).cpu().numpy() > 0.5  # Use sigmoid for binary classification
#             test_accuracy = accuracy_score(test_labels.cpu(), test_pred)
#             test_roc_auc = roc_auc_score(test_labels.cpu(), torch.sigmoid(test_preds).cpu())

#         print(f'Train Accuracy: {train_accuracy:.3f}, Train ROC-AUC: {train_roc_auc:.3f}')
#         print(f'Test Accuracy: {test_accuracy:.3f}, Test ROC-AUC: {test_roc_auc:.3f}')

#     def run(self):
#         self.prepare_data()
#         self.create_loaders()
#         self.train_lstm()
#         self.train_mlp()
#         self.train_cnn()
#         self.evaluate()


class LSTM_CNN_MODEL_TABULAR_ADDED:
    def __init__(self, tot_df, waveform, lstm_input_size, hidden_size, lstm_output_size, mlp_input_size, mlp_hidden_size, mlp_output_size, num_epochs, batch_size, num_epochs_cnn, tabular_features):
        self.tot_df = tot_df
        self.waveform = waveform
        self.lstm_input_size = lstm_input_size
        self.hidden_size = hidden_size
        self.lstm_output_size = lstm_output_size
        self.mlp_input_size = mlp_input_size
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_output_size = mlp_output_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_epochs_cnn = num_epochs_cnn
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tabular_features = tabular_features

    def preprocess_data(self, df):
        data_dict = {}
        labels = {}
        tabular_data_dict = {}

        for patient_id in df['PatientID'].unique():
            patient_data = df[df['PatientID'] == patient_id]
            data_list = []

            for i in range(len(patient_data)):
                d = patient_data[self.waveform].iloc[i]

                if len(d) > 12:
                    keys_to_exclude = {'V3R', 'V4R', 'V7'}
                    filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
                else:
                    filtered_dict = d

                single_data = np.vstack(filtered_dict.values())
                if single_data.shape != (12, 5000):
                    array_2500 = single_data
                    x_2500 = np.linspace(0, 12, single_data.shape[1])
                    x_5000 = np.linspace(0, 12, 5000)
                    linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
                    array_5000 = linear_interpolator(x_5000)
                    data_list.append(array_5000.astype(np.float32))
                else:
                    data_list.append(single_data.astype(np.float32))

            data_dict[patient_id] = np.array(data_list, dtype=np.float32)
            tabular_data_dict[patient_id] = patient_data[self.tabular_features].values[0].astype(np.float32)
            labels[patient_id] = patient_data['Label'].values[0]

        return data_dict, tabular_data_dict, labels

    def prepare_data(self):
        total_data = self.tot_df
        unique_patient_ids = total_data['PatientID'].unique()
        train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

        train_df = total_data[total_data['PatientID'].isin(train_ids)]
        test_df = total_data[total_data['PatientID'].isin(test_ids)]

        train_data, train_tabular_data, train_labels = self.preprocess_data(train_df)
        test_data, test_tabular_data, test_labels = self.preprocess_data(test_df)

        scaler = StandardScaler()

        for patient_id in train_data.keys():
            num_ecgs = train_data[patient_id].shape[0]
            train_data[patient_id] = train_data[patient_id].reshape(-1, train_data[patient_id].shape[-1])
            train_data[patient_id] = scaler.fit_transform(train_data[patient_id])
            train_data[patient_id] = train_data[patient_id].reshape(num_ecgs, 5000, 12)

        for patient_id in test_data.keys():
            num_ecgs = test_data[patient_id].shape[0]
            test_data[patient_id] = test_data[patient_id].reshape(-1, test_data[patient_id].shape[-1])
            test_data[patient_id] = scaler.transform(test_data[patient_id])
            test_data[patient_id] = test_data[patient_id].reshape(num_ecgs, 5000, 12)

        self.train_data = train_data
        self.train_tabular_data = train_tabular_data
        self.train_labels = train_labels
        self.test_data = test_data
        self.test_tabular_data = test_tabular_data
        self.test_labels = test_labels

    def create_loaders(self):
        class ECGDataset(Dataset):
            def __init__(self, data_dict, tabular_data_dict, labels):
                self.data_dict = data_dict
                self.tabular_data_dict = tabular_data_dict
                self.labels = labels
                self.patient_ids = list(data_dict.keys())

            def __len__(self):
                return len(self.patient_ids)

            def __getitem__(self, idx):
                patient_id = self.patient_ids[idx]
                ecgs = self.data_dict[patient_id]
                tabular_data = self.tabular_data_dict[patient_id]
                label = self.labels[patient_id]
                return torch.tensor(ecgs, dtype=torch.float32), torch.tensor(tabular_data, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)  # Label을 float으로 변경

        lstm_batch_size = 1
        train_dataset = ECGDataset(self.train_data, self.train_tabular_data, self.train_labels)
        train_loader = DataLoader(train_dataset, batch_size=lstm_batch_size, shuffle=True)

        test_dataset = ECGDataset(self.test_data, self.test_tabular_data, self.test_labels)
        test_loader = DataLoader(test_dataset, batch_size=lstm_batch_size, shuffle=False)

        self.train_loader = train_loader
        self.test_loader = test_loader

    def train_lstm(self):
        class LSTMEmbedder(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMEmbedder, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]  # Use the output of the last time step
                embedding = self.fc(lstm_out)
                return embedding

        lstm_embedder = LSTMEmbedder(self.lstm_input_size, self.hidden_size, self.lstm_output_size).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(lstm_embedder.parameters(), lr=0.001)

        for epoch in range(self.num_epochs):
            lstm_embedder.train()

            for X_batch, _, y_batch in self.train_loader:
                X_batch = X_batch.squeeze(0).to(self.device)  # Remove batch dimension
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()

                embeddings = lstm_embedder(X_batch)
                outputs = embeddings.mean(dim=0)  # Mean of all embeddings for the patient

                loss = criterion(outputs.unsqueeze(0), y_batch.long())  # CrossEntropyLoss expects long dtype
                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')

        self.lstm_embedder = lstm_embedder

    def extract_embeddings(self, data_loader, model):
        embeddings = []
        labels = []
        model.eval()
        with torch.no_grad():
            for X_batch, _, y_batch in data_loader:
                X_batch = X_batch.squeeze(0).to(self.device)
                y_batch = y_batch.to(self.device)
                patient_embeddings = model(X_batch)
                patient_embedding = patient_embeddings.mean(dim=0)
                embeddings.append(patient_embedding.cpu().numpy())
                labels.append(y_batch.cpu().numpy())
        return np.array(embeddings, dtype=np.float32), np.array(labels)

    def train_mlp(self):
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        mlp = MLP(self.mlp_input_size, self.mlp_hidden_size, self.mlp_output_size).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(mlp.parameters(), lr=0.001)

        tabular_data = np.array([self.train_tabular_data[pid] for pid in self.train_data.keys()], dtype=np.float32)
        labels = np.array([self.train_labels[pid] for pid in self.train_data.keys()], dtype=np.int64)

        tabular_data_tensor = torch.tensor(tabular_data, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        dataset = TensorDataset(tabular_data_tensor, labels_tensor)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            mlp.train()
            for tabular_batch, labels_batch in data_loader:
                tabular_batch = tabular_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)
                optimizer.zero_grad()

                outputs = mlp(tabular_batch)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

            print(f'MLP Epoch {epoch+1}/{self.num_epochs}, Loss: {loss.item()}')

        self.mlp = mlp

    def extract_mlp_embeddings(self, data_loader, model):
        embeddings = []
        labels = []
        model.eval()
        with torch.no_grad():
            for _, tabular_batch, y_batch in data_loader:
                tabular_batch = tabular_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                embedding = model(tabular_batch)
                embeddings.append(embedding.cpu().numpy())
                labels.append(y_batch.cpu().numpy())
        return np.array(embeddings, dtype=np.float32), np.array(labels)

    def train_cnn(self):
        input_size = self.lstm_output_size + self.mlp_output_size

        class ComplexCNNModel(nn.Module):
            def __init__(self):
                super(ComplexCNNModel, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm1d(64)
                self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
                self.bn2 = nn.BatchNorm1d(128)
                self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm1d(256)
                self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
                self.fc1 = nn.Linear(256 * (input_size // 8), 256)
                self.fc2 = nn.Linear(256, 1)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.bn1(self.conv1(x))))
                x = self.pool(self.relu(self.bn2(self.conv2(x))))
                x = self.pool(self.relu(self.bn3(self.conv3(x))))
                x = x.view(x.size(0), -1)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x

        cnn_model = ComplexCNNModel().to(self.device)
        cnn_criterion = nn.BCEWithLogitsLoss()
        cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

        train_lstm_embeddings, _ = self.extract_embeddings(self.train_loader, self.lstm_embedder)
        test_lstm_embeddings, _ = self.extract_embeddings(self.test_loader, self.lstm_embedder)
        train_mlp_embeddings, train_labels_array = self.extract_mlp_embeddings(self.train_loader, self.mlp)
        test_mlp_embeddings, test_labels_array = self.extract_mlp_embeddings(self.test_loader, self.mlp)

        train_mlp_embeddings = train_mlp_embeddings.reshape(train_mlp_embeddings.shape[0], -1)
        test_mlp_embeddings = test_mlp_embeddings.reshape(test_mlp_embeddings.shape[0], -1)

        train_embeddings = np.concatenate([train_lstm_embeddings, train_mlp_embeddings], axis=1)
        test_embeddings = np.concatenate([test_lstm_embeddings, test_mlp_embeddings], axis=1)

        train_embeddings_tensor = torch.tensor(train_embeddings, dtype=torch.float32).unsqueeze(1).to(self.device)
        train_labels_tensor = torch.tensor(train_labels_array, dtype=torch.float32).to(self.device)  # squeeze 제거
        test_embeddings_tensor = torch.tensor(test_embeddings, dtype=torch.float32).unsqueeze(1).to(self.device)
        test_labels_tensor = torch.tensor(test_labels_array, dtype=torch.float32).to(self.device)  # squeeze 제거

        train_embed_dataset = TensorDataset(train_embeddings_tensor, train_labels_tensor)
        train_embed_loader = DataLoader(train_embed_dataset, batch_size=self.batch_size, shuffle=True)

        test_embed_dataset = TensorDataset(test_embeddings_tensor, test_labels_tensor)
        test_embed_loader = DataLoader(test_embed_dataset, batch_size=self.batch_size, shuffle=False)

        for epoch in range(self.num_epochs_cnn):
            cnn_model.train()

            for X_batch, y_batch in train_embed_loader:
                cnn_optimizer.zero_grad()

                outputs = cnn_model(X_batch).squeeze(1)  # 모델 출력의 크기를 [batch_size]로 변경

                # y_batch가 2차원인 경우 1차원으로 변환
                if y_batch.dim() == 2 and y_batch.size(1) == 1:
                    y_batch = y_batch.squeeze(1)

                loss = cnn_criterion(outputs, y_batch)
                loss.backward()
                cnn_optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'CNN Epoch {epoch+1}/{self.num_epochs_cnn}, Loss: {loss.item():.3f}')

        self.cnn_model = cnn_model
        self.train_embed_loader = train_embed_loader
        self.test_embed_loader = test_embed_loader

    def evaluate(self):
        self.cnn_model.eval()
        with torch.no_grad():
            train_preds = []
            train_labels = []
            for X_batch, y_batch in self.train_embed_loader:
                train_outputs = self.cnn_model(X_batch).squeeze(1)
                train_preds.append(train_outputs)
                train_labels.append(y_batch.squeeze(1))

            train_preds = torch.cat(train_preds)
            train_labels = torch.cat(train_labels)
            train_pred = torch.sigmoid(train_preds).cpu().numpy() > 0.5
            train_accuracy = accuracy_score(train_labels.cpu(), train_pred)
            train_roc_auc = roc_auc_score(train_labels.cpu(), torch.sigmoid(train_preds).cpu())

            test_preds = []
            test_labels = []
            for X_batch, y_batch in self.test_embed_loader:
                test_outputs = self.cnn_model(X_batch).squeeze(1)
                test_preds.append(test_outputs)
                test_labels.append(y_batch.squeeze(1))

            test_preds = torch.cat(test_preds)
            test_labels = torch.cat(test_labels)
            test_pred = torch.sigmoid(test_preds).cpu().numpy() > 0.5
            test_accuracy = accuracy_score(test_labels.cpu(), test_pred)
            test_roc_auc = roc_auc_score(test_labels.cpu(), torch.sigmoid(test_preds).cpu())

        print(f'Train Accuracy: {train_accuracy:.3f}, Train ROC-AUC: {train_roc_auc:.3f}')
        print(f'Test Accuracy: {test_accuracy:.3f}, Test ROC-AUC: {test_roc_auc:.3f}')

    def run(self):
        self.prepare_data()
        self.create_loaders()
        self.train_lstm()
        self.train_mlp()
        self.train_cnn()
        self.evaluate()


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
from scipy.interpolate import interp1d

def preprocess_data(df):
    data_list = []
    for i in range(len(df)):
        d = df['WaveForm'].iloc[i]
        if len(d) > 12:
            keys_to_exclude = {'V3R', 'V4R', 'V7'}
            filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
        else:
            filtered_dict = d

        single_data = np.vstack(filtered_dict.values())
        if single_data.shape != (12, 5000):
            array_2500 = single_data
            x_2500 = np.linspace(0, 12, single_data.shape[1])
            x_5000 = np.linspace(0, 12, 5000)
            linear_interpolator = interp1d(x_2500, array_2500, kind='linear')
            array_5000 = linear_interpolator(x_5000)
            data_list.append(array_5000.T)  # (12, 5000) -> (5000, 12)
        else:
            data_list.append(single_data.T)  # (12, 5000) -> (5000, 12)
    return np.array(data_list)

def create_tf_dataset(X, y, shuffle, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.patch_embeddings = tf.keras.layers.Conv1D(filters=hidden_size, kernel_size=patch_size, strides=patch_size)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1, 1, self.hidden_size), trainable=True, name="cls_token")
        num_patches = input_shape[1] // self.patch_size
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.hidden_size), trainable=True, name="position_embeddings"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        inputs_shape = tf.shape(inputs)  
        embeddings = self.patch_embeddings(inputs, training=training)
        cls_tokens = tf.repeat(self.cls_token, repeats=inputs_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens, embeddings), axis=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation="gelu", dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool = False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

# class Block(tf.keras.layers.Layer):
#     def __init__(
#         self,
#         num_heads,
#         attention_dim,
#         attention_bias,
#         mlp_dim,
#         attention_dropout=0.0,
#         sd_survival_probability=1.0,
#         activation="gelu",
#         dropout=0.0,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.norm_before = tf.keras.layers.LayerNormalization()
#         self.attn = tf.keras.layers.MultiHeadAttention(
#             num_heads,
#             attention_dim // num_heads,
#             use_bias=attention_bias,
#             dropout=attention_dropout,
#         )
#         self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
#         self.norm_after = tf.keras.layers.LayerNormalization()
#         self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

#     def call(self, inputs, training=False):
#         x = self.norm_before(inputs, training=training)
#         x = self.attn(x, x, training=training)
#         x = self.stochastic_depth([inputs, x], training=training)
#         x2 = self.norm_after(x, training=training)
#         x2 = self.mlp(x2, training=training)
#         return self.stochastic_depth([x, x2], training=training)

class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dim,
        attention_bias,
        mlp_dim,
        attention_dropout=0.0,
        sd_survival_probability=1.0,
        activation="gelu",
        dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias=attention_bias,
            dropout=attention_dropout,
        )
        self.stochastic_depth = tfa.layers.StochasticDepth(sd_survival_probability)
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim=mlp_dim, activation=activation, dropout=dropout)

    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training=training)
        x = self.stochastic_depth([inputs, x], training=training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return self.stochastic_depth([x, x2], training=training)

    def get_attention_scores(self, inputs):
        # 여기서 attention scores를 계산하여 반환합니다.
        # x는 norm_before을 통과한 값이어야 하므로, 이를 활용해 attention을 계산합니다.
        norm_inputs = self.norm_before(inputs, training=False)
        _, attention_scores = self.attn(norm_inputs, norm_inputs, return_attention_scores=True)
        return attention_scores


class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_dim,
        num_classes,
        dropout=0.0,
        sd_survival_probability=1.0,
        attention_bias=False,
        attention_dropout=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embeddings = ViTEmbeddings(patch_size, hidden_size, dropout)
        sd = tf.linspace(1.0, sd_survival_probability, depth)
        self.blocks = [
            Block(
                num_heads,
                attention_dim=hidden_size,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                mlp_dim=mlp_dim,
                sd_survival_probability=(sd[i].numpy().item()),
                dropout=dropout,
            )
            for i in range(depth)
        ]

        self.norm = tf.keras.layers.LayerNormalization()

        self.head = tf.keras.layers.Dense(num_classes)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.embeddings(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        x = x[:, 0]  # take only cls_token
        return self.head(x)

    def get_last_selfattention(self, inputs: tf.Tensor):
        x = self.embeddings(inputs, training=False)
        for block in self.blocks[:-1]:
            x = block(x, training=False)
        return self.blocks[-1].get_attention_scores(x)

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import random
import os

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def run_vit_training(df, batch_size=32, epochs=20, test_size=0.1, random_state=42, 
                     patch_size=10, hidden_size=768, depth=12, num_heads=12, mlp_dim=256):
    # 시드 설정
    set_seeds(random_state)
    
    # GPU 사용 시 결정적 동작 설정
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.enable_op_determinism()

    # PatientID별로 데이터 나누기
    unique_patient_ids = df['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=test_size, random_state=random_state)

    train_df = df[df['PatientID'].isin(train_ids)]
    test_df = df[df['PatientID'].isin(test_ids)]

    # 데이터 전처리 수행
    X_train = preprocess_data(train_df)
    y_train = np.array(train_df['Label'])
    X_test = preprocess_data(test_df)
    y_test = np.array(test_df['Label'])

    # 데이터 정규화 (각 채널별로)
    scaler = StandardScaler()
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.reshape(-1, 5000, 12)  # (12, 5000) -> (5000, 12)

    X_test = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(-1, 5000, 12)  # (12, 5000) -> (5000, 12)

    # 텐서플로우 데이터셋 생성
    train_ds = create_tf_dataset(X_train, y_train, shuffle=True, batch_size=batch_size)
    val_ds = create_tf_dataset(X_test, y_test, shuffle=False, batch_size=batch_size)

    # ViT 모델 생성
    vit = VisionTransformer(
        patch_size=patch_size,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_classes=1,
        sd_survival_probability=0.9,
    )

    # 모델 컴파일 및 학습
    optimizer = tf.keras.optimizers.Adam(0.0001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.AUC(from_logits=True, name="roc_auc"),
               tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    vit.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    cbs = [tf.keras.callbacks.ModelCheckpoint("vit_best/", monitor="val_roc_auc", save_best_only=True, save_weights_only=True)]

    vit.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)

    # 최종 테스트 결과 평가
    test_loss, test_roc_auc, test_accuracy = vit.evaluate(val_ds, verbose=1)

    # ROC-AUC 곡선을 위한 예측
    y_pred = vit.predict(val_ds)
    y_pred = tf.nn.sigmoid(y_pred).numpy().flatten()  # logits을 확률로 변환

    # ROC 곡선 계산
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # ROC 곡선 그리기
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    return vit, {"Test Loss": test_loss, "Test ROC AUC": test_roc_auc, "Test Accuracy": test_accuracy}


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_ecg_attention(ecg_data, model, lead_index=11, weights_path='vit_best/'):

    # 배치 차원 추가 (1, 5000, 12)
    input_data = np.expand_dims(ecg_data, axis=0)
    
    # 모델 가중치 로드
    model.load_weights(weights_path)

    # TensorFlow 텐서로 변환
    input_data_tf = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # 모델에서 self-attention 맵을 얻음
    attn = model.get_last_selfattention(input_data_tf)

    # CLS 토큰 관련 attention 스코어 추출
    attn = attn[0, :, 0, 1:]
    
    # Attention 스코어를 5000 샘플로 확장
    attn = tf.transpose(attn, (1, 0))  # (250, 6) -> (6, 250)
    attn = tf.expand_dims(tf.expand_dims(attn, 0), 0)
    attn = tf.image.resize(attn, (1, 5000))[0, 0]
    
    # ECG 데이터와 모든 head의 attention 값 평균
    ecg_data_lead = ecg_data[:, lead_index]  # 선택한 리드의 ECG 데이터
    attn_avg = np.mean(attn, axis=1)  # 모든 head의 attention 값의 평균

    # x 축 좌표 생성
    x = np.linspace(0, len(ecg_data_lead), len(ecg_data_lead))

    # 전체 ECG 데이터를 검정색 라인으로 플롯
    fig, ax = plt.subplots(figsize=(30, 8))
    ax.plot(x, ecg_data_lead, color='black', linewidth=1, label=f'Lead {lead_index + 1}')

    # 컬러맵과 노멀라이제이션 설정
    cmap = plt.get_cmap('Reds')
    norm = Normalize(vmin=np.min(attn_avg), vmax=np.max(attn_avg))

    # 강조할 부분만을 별도로 계산
    points = np.array([x, ecg_data_lead]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(attn_avg)  # 평균화된 attn 값에 따라 색상을 설정
    lc.set_linewidth(2)

    # 강조된 부분을 검정색 라인 위에 덮어쓰기
    ax.add_collection(lc)

    # 컬러바 추가
    fig.colorbar(lc, ax=ax, label='Average Attention Score')

    # 그래프 설정
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(ecg_data_lead.min(), ecg_data_lead.max())
    ax.set_title(f'ECG Lead {lead_index + 1} with Highlighted Average Attention')

    plt.show()





import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

def plot_ecg_attention_per_head(ecg_data, model, head_list =[1,2], lead_list =[1,2], weights_path='vit_best/'):
    # 배치 차원 추가 (1, 5000, 12)
    input_data = np.expand_dims(ecg_data, axis=0)
    
    # 모델 가중치 로드
    model.load_weights(weights_path)

    # TensorFlow 텐서로 변환
    input_data_tf = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # 모델에서 self-attention 맵을 얻음
    attn = model.get_last_selfattention(input_data_tf)

    # CLS 토큰 관련 attention 스코어 추출
    attn = attn[0, :, 0, 1:]  # (num_heads, num_patches)
    
    num_heads = attn.shape[0]
    num_leads = ecg_data.shape[1]

    # Attention 스코어를 5000 샘플로 확장
    attn = tf.transpose(attn, (1, 0))  # (num_patches, num_heads)
    attn = tf.expand_dims(tf.expand_dims(attn, 0), 0)
    attn = tf.image.resize(attn, (1, 5000))[0, 0]  # (5000, num_heads)

    # 그래프 생성
    fig, axes = plt.subplots(num_heads, num_leads, figsize=(5*num_leads, 4*num_heads))
    fig.suptitle("ECG Leads with Attention Scores for Each Head", fontsize=16)

    for head in head_list:
        for lead in lead_list:
            ax = axes[head, lead]
            
            ecg_data_lead = ecg_data[:, lead]
            attn_head = attn[:, head]

            # x 축 좌표 생성
            x = np.linspace(0, len(ecg_data_lead), len(ecg_data_lead))

            # 전체 ECG 데이터를 검정색 라인으로 플롯
            ax.plot(x, ecg_data_lead, color='black', linewidth=1)

            # 컬러맵과 노멀라이제이션 설정
            cmap = plt.get_cmap('Reds')
            norm = Normalize(vmin=np.min(attn_head), vmax=np.max(attn_head))

            # 강조할 부분만을 별도로 계산
            points = np.array([x, ecg_data_lead]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(attn_head)
            lc.set_linewidth(2)

            # 강조된 부분을 검정색 라인 위에 덮어쓰기
            ax.add_collection(lc)

            # 그래프 설정
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(ecg_data_lead.min(), ecg_data_lead.max())
            ax.set_title(f'Head {head+1}, Lead {lead+1}')
            ax.set_xticks([])
            ax.set_yticks([])

    # 컬러바 추가
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(lc, cax=cbar_ax, label='Attention Score')

    plt.tight_layout()
    plt.show()

# 사용 예시:
# ecg_data = ... # 여기에 실제 ECG 데이터를 로드하세요
# model = ... # 여기에 실제 ViT 모델을 로드하세요
# plot_ecg_attention_per_head(ecg_data, model, weights_path='path/to/your/weights')











import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def run_vit_training_patient(df, batch_size=32, epochs=20, test_size=0.1, random_state=42,
                             patch_size = 10,
                    hidden_size = 768,
                    depth = 12,
                    num_heads = 12,
                    mlp_dim =256):
    # PatientID별로 데이터 나누기
    unique_patient_ids = df['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=test_size, random_state=random_state)

    train_df = df[df['PatientID'].isin(train_ids)]
    test_df = df[df['PatientID'].isin(test_ids)]

    # 데이터 전처리 수행
    X_train = preprocess_data(train_df)
    y_train = np.array(train_df['Label'])
    X_test = preprocess_data(test_df)
    y_test = np.array(test_df['Label'])

    # 데이터 정규화 (각 채널별로)
    scaler = StandardScaler()
    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.reshape(-1, 12, 5000)

    X_test = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(-1, 12, 5000)

    # 텐서플로우 데이터셋 생성
    train_ds = create_tf_dataset(X_train, y_train, shuffle=True, batch_size=batch_size)
    val_ds = create_tf_dataset(X_test, y_test, shuffle=False, batch_size=batch_size)

    # ViT 모델 생성
    vit = VisionTransformer(
        patch_size=patch_size,  # 더 작은 patch_size를 사용하여 더 많은 패치를 생성
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        num_classes=1,
        sd_survival_probability=0.9,
    )

    # 모델 컴파일 및 학습
    optimizer = tf.keras.optimizers.Adam(0.0001)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.AUC(from_logits=True, name="roc_auc"), 'accuracy']
    vit.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    cbs = [tf.keras.callbacks.ModelCheckpoint("vit_best_patients/", monitor="val_roc_auc", save_best_only=True, save_weights_only=True)]

    vit.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=cbs)

    # 최종 테스트 결과 평가
    test_loss, test_roc_auc, test_accuracy = vit.evaluate(val_ds, verbose=1)

    return vit,{"Test Loss": test_loss, "Test ROC AUC": test_roc_auc, "Test Accuracy": test_accuracy}


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv1D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.models import Model


def create_vit_tabular_cnn_model(vit_model, num_tabular_features, hidden_size, num_classes=1):
    # ViT 모델의 출력
    vit_input = Input(shape=(12, 5000))  # ViT 모델의 입력
    vit_output = vit_model(vit_input)  # ViT의 출력, shape: (batch_size, hidden_size)
    
    # Tabular 피처 입력
    tabular_input = Input(shape=(num_tabular_features,))  # 테이블형 피처의 입력
    
    # ViT 출력과 Tabular 피처 결합
    combined = Concatenate()([vit_output, tabular_input])
    
    # CNN 계층에 보내기 위해 결합된 출력을 적절한 형태로 변환
    combined_reshaped = tf.expand_dims(combined, -1)  # (batch_size, combined_size, 1)
    
    # CNN 계층
    cnn_output = Conv1D(filters=64, kernel_size=3, activation='relu')(combined_reshaped)
    cnn_output = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn_output)
    cnn_output = GlobalAveragePooling1D()(cnn_output)
    
    # 최종 Dense Layer로 예측 수행
    x = Dense(hidden_size, activation='relu')(cnn_output)
    x = Dense(hidden_size // 2, activation='relu')(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    # 최종 모델 정의
    model = Model(inputs=[vit_input, tabular_input], outputs=output)
    return model

def create_vit_tabular_cnn_model(vit_model, num_tabular_features, hidden_size, num_classes=1):
    # ViT 모델의 출력
    vit_input = Input(shape=(12, 5000))  # ViT 모델의 입력
    vit_output = vit_model(vit_input)  # ViT의 출력, shape: (batch_size, hidden_size)
    
    # Tabular 피처 입력
    tabular_input = Input(shape=(num_tabular_features,))  # 테이블형 피처의 입력
    
    # ViT 출력과 Tabular 피처 결합
    combined = Concatenate()([vit_output, tabular_input])
    
    # CNN 계층에 보내기 위해 결합된 출력을 적절한 형태로 변환
    combined_reshaped = tf.expand_dims(combined, -1)  # (batch_size, combined_size, 1)
    
    # CNN 계층
    cnn_output = Conv1D(filters=64, kernel_size=3, activation='relu')(combined_reshaped)
    cnn_output = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn_output)
    cnn_output = GlobalAveragePooling1D()(cnn_output)
    
    # 최종 Dense Layer로 예측 수행
    x = Dense(hidden_size, activation='relu')(cnn_output)
    x = Dense(hidden_size // 2, activation='relu')(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    # 최종 모델 정의
    model = Model(inputs=[vit_input, tabular_input], outputs=output)
    return model


import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.layers import Input, Dense, Concatenate, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def set_seed(seed=42):
    # Python 랜덤 시드 설정
    random.seed(seed)
    # NumPy 랜덤 시드 설정
    np.random.seed(seed)
    # TensorFlow 랜덤 시드 설정
    tf.random.set_seed(seed)

def run_vit_tabular_cnn_training(
    df,
    vit_model,
    tabular_features,  # Tabular feature의 이름 리스트를 인자로 받음
    patch_size=10,
    hidden_size=768,
    depth=12,
    num_heads=12,
    mlp_dim=256,
    dropout=0.1,
    attention_dropout=0.0,
    learning_rate=0.0001,
    batch_size=32,
    epochs=20,
    test_size=0.1,
    random_state=42,
    sd_survival_probability=0.9,
):
    # 시드 설정 (모델 학습의 일관성을 위해)
    set_seed(random_state)
    
    # PatientID를 기준으로 데이터 나누기
    unique_patient_ids = df['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=test_size, random_state=random_state)

    train_df = df[df['PatientID'].isin(train_ids)]
    test_df = df[df['PatientID'].isin(test_ids)]

    # ViT 입력 데이터 전처리 (WaveForm 열에서 시계열 데이터를 전처리)
    X_train_vit = preprocess_data(train_df)
    y_train = np.array(train_df['Label'])
    X_test_vit = preprocess_data(test_df)
    y_test = np.array(test_df['Label'])

    # Tabular 데이터 추출
    # 사용자가 지정한 열들만 선택하여 Tabular 데이터를 구성
    X_train_tabular = np.array(train_df[tabular_features])
    X_test_tabular = np.array(test_df[tabular_features])

    # 데이터 정규화 (각 채널별로)
    scaler = StandardScaler()
    X_train_vit = X_train_vit.reshape(-1, X_train_vit.shape[-1])
    X_train_vit = scaler.fit_transform(X_train_vit)
    X_train_vit = X_train_vit.reshape(-1, 12, 5000)

    X_test_vit = X_test_vit.reshape(-1, X_test_vit.shape[-1])
    X_test_vit = scaler.transform(X_test_vit)
    X_test_vit = X_test_vit.reshape(-1, 12, 5000)

    # ViT 및 Tabular 데이터를 결합한 모델 생성
    combined_model = create_vit_tabular_cnn_model(vit_model, len(tabular_features), hidden_size)

    # 모델 컴파일
    combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=[tf.keras.metrics.AUC(name='roc_auc'), 'accuracy'])

    # 모델 학습
    combined_model.fit([X_train_vit, X_train_tabular], y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    # 모델 평가 (테스트 데이터에서)
    test_loss, test_roc_auc, test_accuracy = combined_model.evaluate([X_test_vit, X_test_tabular], y_test, verbose=1)

    # 테스트 결과 출력
    print(f"Test Loss: {test_loss}")
    print(f"Test ROC AUC: {test_roc_auc}")
    print(f"Test Accuracy: {test_accuracy}")

    return combined_model, test_loss, test_roc_auc, test_accuracy

def create_vit_tabular_cnn_model(vit_model, num_tabular_features, hidden_size, num_classes=1):
    # ViT 모델의 출력
    vit_input = Input(shape=(12, 5000))  # ViT 모델의 입력
    vit_output = vit_model(vit_input)  # ViT의 출력, shape: (batch_size, hidden_size)
    
    # Tabular 피처 입력
    tabular_input = Input(shape=(num_tabular_features,))  # 테이블형 피처의 입력
    
    # ViT 출력과 Tabular 피처 결합
    combined = Concatenate()([vit_output, tabular_input])
    
    # CNN 계층에 보내기 위해 결합된 출력을 적절한 형태로 변환
    combined_reshaped = tf.expand_dims(combined, -1)  # (batch_size, combined_size, 1)
    
    # CNN 계층
    cnn_output = Conv1D(filters=64, kernel_size=3, activation='relu')(combined_reshaped)
    cnn_output = Conv1D(filters=128, kernel_size=3, activation='relu')(cnn_output)
    cnn_output = GlobalAveragePooling1D()(cnn_output)
    
    # 최종 Dense Layer로 예측 수행
    x = Dense(hidden_size, activation='relu')(cnn_output)
    x = Dense(hidden_size // 2, activation='relu')(x)
    output = Dense(num_classes, activation='sigmoid')(x)

    # 최종 모델 정의
    model = Model(inputs=[vit_input, tabular_input], outputs=output)
    return model





import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# VisionTransformer 클래스를 PyTorch 모델과 호환되도록 래핑
class WrappedVisionTransformer(nn.Module):
    def __init__(self, vit_model, hidden_size):
        super(WrappedVisionTransformer, self).__init__()
        self.vit_model = vit_model
        self.hidden_size = hidden_size

    def forward(self, x):
        # PyTorch tensor를 TensorFlow로 전달하기 위해 numpy로 변환
        x = x.cpu().detach().numpy()  # Convert PyTorch tensor to numpy
        x = self.vit_model(x, training=False).numpy()  # VisionTransformer 모델 통과
        x = torch.tensor(x, dtype=torch.float32)  # PyTorch 텐서로 다시 변환
        return x

def run_vit_tabular_model(tot_df, batch_size, epoch, waveform, tabular_features):
    total_data = tot_df

    # PatientID별로 데이터 나누기
    unique_patient_ids = total_data['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

    train_df = total_data[total_data['PatientID'].isin(train_ids)]
    test_df = total_data[total_data['PatientID'].isin(test_ids)]

    # 데이터 전처리 함수 정의
    def preprocess_data(df):
        data_list = []
        tabular_data_list = []
        for i in range(len(df)):
            d = df[waveform].iloc[i]
            if len(d) > 12:
                keys_to_exclude = {'V3R', 'V4R', 'V7'}
                filtered_dict = {key: value for key, value in d.items() if key not in keys_to_exclude}
            else:
                filtered_dict = d

            single_data = np.vstack(filtered_dict.values())
            if single_data.shape != (12, 5000):
                x_2500 = np.linspace(0, 12, single_data.shape[1])
                x_5000 = np.linspace(0, 12, 5000)
                single_data = np.interp(x_5000, x_2500, single_data.T).T
            data_list.append(single_data)
            tabular_data_list.append(df[tabular_features].iloc[i].values)
        return np.array(data_list), np.array(tabular_data_list)

    # 데이터 전처리 수행
    X_train, X_train_tabular = preprocess_data(train_df)
    y_train = np.array(train_df['Label'])
    X_test, X_test_tabular = preprocess_data(test_df)
    y_test = np.array(test_df['Label'])

    # 데이터 정규화 (각 채널별로)
    scaler = StandardScaler()

    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_train = scaler.fit_transform(X_train)
    X_train = X_train.reshape(-1, 12, 5000)

    X_test = X_test.reshape(-1, X_test.shape[-1])
    X_test = scaler.transform(X_test)
    X_test = X_test.reshape(-1, 12, 5000)

    # Tabular 데이터 정규화
    tabular_scaler = StandardScaler()
    X_train_tabular = tabular_scaler.fit_transform(X_train_tabular)
    X_test_tabular = tabular_scaler.transform(X_test_tabular)

    # Convert numpy arrays to PyTorch tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_train_tabular = torch.tensor(X_train_tabular, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    X_test_tabular = torch.tensor(X_test_tabular, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    # model.py에서 VisionTransformer 클래스 가져오기
    vit_model_tf = VisionTransformer(
        patch_size=10,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_dim=256,
        num_classes=768,
        dropout=0.1
    )

    # TensorFlow 모델을 PyTorch에서 사용할 수 있도록 래핑
    vit_model = WrappedVisionTransformer(vit_model_tf, hidden_size=768)

    # ViT와 Tabular 데이터를 결합한 모델 정의
    class ViTTabularModel(nn.Module):
        def __init__(self, vit_model, num_tabular_features):
            super(ViTTabularModel, self).__init__()
            self.vit_model = vit_model
            self.fc_tabular = nn.Linear(num_tabular_features, 64)
            self.fc_combined = nn.Linear(768 + 64, 1)  # Combining ViT output and tabular data
            self.dropout = nn.Dropout(0.5)
            self.relu = nn.ReLU()
            
        def forward(self, x, tabular_data):
            x = self.vit_model(x)  # ViT 모델 통과
            x = x.to(tabular_data.device)  # ViT 출력이 Tabular 데이터와 동일한 장치에 있는지 확인
            tabular_data = self.relu(self.fc_tabular(tabular_data))  # Tabular 데이터 처리
            combined = torch.cat((x, tabular_data), dim=1)  # ViT 출력과 Tabular 데이터 결합
            x = self.dropout(self.relu(self.fc_combined(combined)))
            return x.squeeze(-1)

    # 모델 초기화, 손실 함수 및 옵티마이저 정의
    model = ViTTabularModel(vit_model, num_tabular_features=len(tabular_features)).to(device)
    criterion = nn.BCEWithLogitsLoss()  # 이진 분류를 위한 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    num_epochs = epoch

    # DataLoader 생성
    train_data = torch.utils.data.TensorDataset(X_train, X_train_tabular, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정

        train_loss = 0.0
        for inputs, tabular_data, labels in train_loader:
            inputs, tabular_data, labels = inputs.to(device), tabular_data.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs, tabular_data)
            loss = criterion(outputs, labels)
            
            # Backward pass 및 최적화
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)

        train_loss = train_loss / len(train_loader.dataset)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')

    print("Training complete.")

    # 테스트 데이터 평가
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test, X_test_tabular)
        test_outputs = test_outputs.squeeze()  # 여분의 차원 제거

    # 로짓을 확률로 변환한 후 이진 예측으로 변환
    test_probabilities = torch.sigmoid(test_outputs)
    test_predictions = torch.round(test_probabilities)

    # PyTorch 텐서를 numpy 배열로 변환
    test_predictions_np = test_predictions.cpu().numpy()
    y_test_np = y_test.cpu().numpy()

    # 정확도 계산
    test_accuracy = accuracy_score(y_test_np, test_predictions_np)

    # ROC AUC 점수 계산
    test_roc_auc = roc_auc_score(y_test_np, test_probabilities.cpu().numpy())

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test ROC AUC Score: {test_roc_auc:.4f}')

