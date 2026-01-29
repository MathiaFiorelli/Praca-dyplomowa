# Neural Network module

from time import time
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# set style matplotlib
sns.set_style("whitegrid")


class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4, restore_best_weights=True):
        """
        patience – ile epok czekamy na poprawę zanim przerwiemy
        min_delta – minimalna zmiana w stracie uznawana za poprawę
        restore_best_weights – czy przywrócić najlepsze wagi po zatrzymaniu
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.best_state_dict = None
        self.early_stop = False

    def __call__(self, current_loss, model):
        if self.best_loss is None:
            self.best_loss = current_loss
            self.best_state_dict = model.state_dict()
        elif current_loss > self.best_loss - self.min_delta:
            # brak poprawy
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    print("🔁 Przywracam najlepsze wagi modelu...")
                    model.load_state_dict(self.best_state_dict)
        else:
            # poprawa - zapisujemy najlepszy stan modelu
            self.best_loss = current_loss
            self.best_state_dict = model.state_dict()
            self.counter = 0



# Autoencoder 1
class Autoencoder_Linear1(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),

        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 6),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Autoencoder 2
class Autoencoder_Linear2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 6),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# Autoencoder 3
class Autoencoder_Linear3(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(6, 5),
            nn.ReLU(),
            nn.Linear(5, 4),
            nn.ReLU(),
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(1, 2),
            nn.ReLU(),
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.ReLU(),
            nn.Linear(4, 5),
            nn.ReLU(),
            nn.Linear(5, 6),
            nn.Tanh()
            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
# def early_stopping(loss_list, patience=10, min_delta=1e-4):
#     """Early training stop

#     Args:
#         loss_list (list): list of losses until now
#         patience (int, optional): after this number of epochs the check will be performed. Defaults to 10.
#         min_delta (float, optional): _description_. Defaults to 1e-4.

#     Returns:
#         boolean: flag to stop. Stop if True
#     """
#     if len(loss_list) < patience:
#         return False
#     else:
#         loss_list = [round(x, 4) for x in loss_list]

#         recent_losses = loss_list[-patience:]
#         deltas = [abs(recent_losses[i] - recent_losses[i - 1]) for i in range(1, len(recent_losses))]
#         return all(delta < min_delta for delta in deltas)
        

def train_model(model,df,num_epochs=1000):
    
    early_stopping = EarlyStopping(patience=30, min_delta=1e-4)

    tensor = torch.tensor(df.values, dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                            lr=1e-3,
                            weight_decay=1e-5)
    
    loss_list = []
    for epoch in tqdm(range(num_epochs)):
        recon = model(tensor)
        loss = criterion(recon, tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())


        # po policzeniu avg_epoch_total_loss
        early_stopping(loss.item(), model)
        if early_stopping.early_stop:
            print(f"\n⛔ Early stopping na epoce {epoch}")
            break
        

    return model,loss_list

def analyze_model(model,df,num_epochs=1000):
    tensor = torch.tensor(df.values, dtype=torch.float32)
    start_neural_time = time()
    model, losses = train_model(model,df,num_epochs)
    end_neural_time = time() - start_neural_time
    with torch.no_grad():
        recon = model(tensor)
        mse_neural = torch.mean((recon - tensor) ** 2).item()
        mae_neural = torch.mean(torch.abs(recon - tensor)).item()

    return model, mse_neural, mae_neural, losses, end_neural_time