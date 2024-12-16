
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
from src.model import RegressionModel
from src.utils import load_esm_model, get_esm_embedding

class CRISPRDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.esm_model, self.batch_converter = load_esm_model()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        gRNA_embedding = get_esm_embedding(row["gRNA"], self.esm_model, self.batch_converter)
        target_embedding = get_esm_embedding(row["Target DNA"], self.esm_model, self.batch_converter)
        features = torch.cat((gRNA_embedding, target_embedding), dim=0)
        label = torch.tensor(row["Average Score"], dtype=torch.float32)
        return features, label

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)

    # Load processed data
    df = pd.read_excel("output/gRNA_TargetDNA_AverageScore_Combined.xlsx")

    # Clean and ensure data correctness
    # Drop rows with missing scores if any
    df = df.dropna(subset=["Average Score", "gRNA"])

    dataset = CRISPRDataset(df)

    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = RegressionModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    output_log = "output/training_output.txt"

    with open(output_log, "w") as log_file:
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features)
                    loss = criterion(outputs.squeeze(), labels)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            log_msg = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            print(log_msg)
            log_file.write(log_msg + "\n")

    # Save the model
    model_save_path = "output/crispr_regression_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    train_model()
