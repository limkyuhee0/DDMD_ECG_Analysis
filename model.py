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
from tf.
set_seed(42)
def CNN_MODEL(tot_df,batch_size, epoch, waveform):
    total_data = tot_df

    # PatientID별로 데이터 나누기
    unique_patient_ids = total_data['PatientID'].unique()
    train_ids, test_ids = train_test_split(unique_patient_ids, test_size=0.1, random_state=42)

    train_df = total_data[total_data['PatientID'].isin(train_ids)]
    test_df = total_data[total_data['PatientID'].isin(test_ids)]

    # 데이터 전처리 함수 정의
    def preprocess_data(df):
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