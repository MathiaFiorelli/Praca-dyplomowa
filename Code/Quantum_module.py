
# pip install pennylane torch
import torch
import torch.nn as nn
import pennylane as qml
#from pennylane import numpy as pnp
import pandas as pd
from time import time
import Neural_Network_module
from tqdm import tqdm


df = pd.read_excel(
    r'C:\Users\mathi\OneDrive\Documents\Studia\SGH\Semestr 4\Praca dyplomowa/Output/Preprocessed.xlsx', sheet_name='Preprocessed DF')

n_qubits = len(df.iloc[0, :])  # Number of features
n_latent = 2  # Number of latent qubits i.e. comression to 2 qubits
n_trash = n_qubits - n_latent
n_layers = 2          # głębokość ansatzu (warstw entanglingowych)

latent_wires = list(range(n_latent))                  # np. [0,1]
trash_wires = list(range(n_latent, n_qubits))        # np. [2,3,4,5]


dev = qml.device("default.qubit", wires=n_qubits, shots=None)

# Przydatne: kształt wag StronglyEntanglingLayers
weight_shape = (n_layers, n_qubits)


# ---- ENCODER ----
def encoder(weights):
    """Encoder function.
    Entangles qubits and sets their Y rotation

    Args:
        weights (_type_): _description_
    """
    # iterate over layers
    for layer in range(weights.shape[0]):
        # Encode rotations
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)

        # Entangling CNOTs in a chain
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

# ---- DECODER ----


def decoder(weights):
    """Decoder Function
    Entangles qubits and sets their Y rotation

    Args:
        weights (_type_): _description_
    """
   # iterate over layers
    for layer in range(weights.shape[0]):
        # Encode rotations
        for i in range(n_qubits):
            qml.RY(weights[layer, i], wires=i)

        # Entangling CNOTs in a chain
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

# Qnode Full Autoencoder


@qml.qnode(dev, interface="torch", diff_method="best")
def decode_qnode(x, encoder_weight, decoder_weight):
    """Autoencoder QNODE

    Args:
        x (list): list of entry data
        encoder_weight (torch.tensor): list of weights for the encoder
        decoder_weight (torch.tensor): list of weights for the decoder

    Returns:
        torch.tensor: list of expected values of Pauli Z measurements of each cubits. Values [-1,1]
    """
    # Loading data into quantum state
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')

    # Encode data
    encoder(encoder_weight)

    # Decode data
    decoder(decoder_weight)

    # Perform PauliZ measurment and get expected values for tyhose measurments
    pauli_z_exp = [qml.expval(qml.PauliZ(qubit_no))
                   for qubit_no in range(n_qubits)]

    return pauli_z_exp

# Qnode for trash wires probabilty measurement


@qml.qnode(dev, interface="torch", diff_method="best")
def trash_zero_qnode(x, encoder_weights):
    """QNODE used to measure the probability of trash qubits in bottleneck layer

    Args:
        x (list): input data
        encoder_weights (torch.tensor): encoder weights

    Returns:
        torch.tensor: list of probabilities of measuring the state |0...0> on the trash wires
    """

    # Load data into quantum state
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')
    encoder(encoder_weights)

    probs = qml.probs(wires=trash_wires)
    return probs

# Qnode for latent state measurement


@qml.qnode(dev, interface="torch", diff_method="best")
def latent_state_values(x, encoder_weights):
    """Qnode used to measure the latent (nottleneck) layer

    Args:
        x (list): list of values used as entry
        encoder_weights (torch.tensor): list of weights for the encoder

    Returns:
        torch.tensor: list of expected values of Pauli Z measurements of each latent cubits. Values [-1,1]
    """
    # Load data into quantum state
    qml.AngleEmbedding(x, wires=range(n_qubits), rotation='Y')

    # Encode data: entangler and rotations
    encoder(encoder_weights)

    # Perform PauliZ measurment and get expected values for tyhose measurments
    pauli_z_exp = [qml.expval(qml.PauliZ(w)) for w in latent_wires]

    return pauli_z_exp


def bloch_vector_latent(dm):

    vecs = []
    for i in range(n_latent):
        bx = qml.math.real(qml.math.trace(dm @ qml.math.kron(*[
            qml.math.eye(2) if j != i else qml.math.array(
                [[0, 1], [1, 0]], like=dm)
            for j in range(n_latent)
        ])))
        by = qml.math.real(qml.math.trace(dm @ qml.math.kron(*[
            qml.math.eye(2) if j != i else qml.math.array(
                [[0, -1j], [1j, 0]], like=dm)
            for j in range(n_latent)
        ])))
        bz = qml.math.real(qml.math.trace(dm @ qml.math.kron(*[
            qml.math.eye(2) if j != i else qml.math.array(
                [[1, 0], [0, -1]], like=dm)
            for j in range(n_latent)
        ])))
        vecs.append(torch.stack([bx, by, bz]))
    return vecs

# ---- MODEL PyTorch ----


class QuantumAutoencoder(nn.Module):
    def __init__(self):
        """initialize en/decoder weights
        """
        super().__init__()
        # Parametry enkodera i dekodera (trainable)
        self.encoder_weight = nn.Parameter(
            torch.rand(size=weight_shape)*0.2-0.1)
        self.decoder_weight = nn.Parameter(
            torch.rand(size=weight_shape)*0.2-0.1)

    @staticmethod
    def _scale_input(x):
        """Scaling function left for legacy purposes. Returns unaltered value. Earlier it clamped on [-pi,pi]

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        scaled = torch.clamp(x, -3.14159, 3.14159)  # legacy
        return x

    def forward(self, row):
        """Forward pass of class

        Args:
            row (torch.tensor): single row of torch tensor containing data

        Returns:
            torch.tensor, torch.tensor: list of reconstructed data, probability of trash wires being in |0...0>
        """
        x_prim = decode_qnode(row, self.encoder_weight, self.decoder_weight)

        # Measure probabilities of trash wires being in |0...0>
        probs_trash = trash_zero_qnode(
            row, self.encoder_weight)  # 2^n_trash elementów

        # index 0 corresponds to |0...0>
        p_zero_trash = probs_trash[0]

        return x_prim, p_zero_trash

    @staticmethod
    def loss_fn(row, x_prim, p_zero_trash, alpha=1.0, beta=0.5):
        """Loss function

        Args:
            row (torch.tensor): single row of torch tensor containing data
            x_prim (torch.tensor): reconstructed data
            p_zero (float32): probability of trash wires being in |0...0>
            alpha (float, optional): reconstruction loss factor. Defaults to 1.0.
            beta (float, optional): compression loss factor. Defaults to 0.5.

        Returns:
            torch.tensor : 
            total loss, 
        """
        x_prim = torch.tensor(x_prim, dtype=torch.float32)
        recon_loss = torch.mean((x_prim - row)**2)
        compression_loss = (1.0 - p_zero_trash)
        total_loss = alpha*recon_loss + beta*compression_loss
        return total_loss

    def latent_bloch(self, x):
        """Wektory Blocha latentnych kubitów (łatwiejsze do logowania/inspekcji)."""
        dm_lat = self.encode_latent_dm(x)
        return bloch_vector_latent(dm_lat)

def main():
    global df
    torch.manual_seed(0)
    model = QuantumAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=0.02)

    df = df.sample(n=200, random_state=42).reset_index(drop=True)
    X = torch.tensor(df.values, dtype=torch.float32)


    BATCH_SIZE = 10
    BATCH_NUM = 5
    losses = []
    star_time_qantum = time()
    for epoch in tqdm(range(1000)):
        total_epoch_loss = 0.0
        for batch_idx in range(BATCH_NUM):
            idx = torch.randint(0, X.shape[0], (BATCH_SIZE,))
            x_sample = X[idx]

            total_batch_loss = 0.0
            opt.zero_grad()  # Zero the gradients
            for x in x_sample:
                # Forward pass -> get reconstructed x and p_zero
                x_prim, p_zero_trash = model(x)
                # Get loss for current row in current batch
                batch_loss = model.loss_fn(
                    row=x, x_prim=x_prim, p_zero_trash=p_zero_trash, alpha=1.0, beta=0.5)
                # Accumulate loss over the batch over the rows (tensor)
                total_batch_loss += batch_loss
                # Get value to accumulate
                total_epoch_loss += batch_loss.item()

            loss = total_batch_loss / len(x_sample)  # Average loss over the batch
            loss.backward()
            opt.step()

        losses.append(total_epoch_loss / BATCH_NUM)

        if Neural_Network_module.early_stopping(losses, patience=20, min_delta=1e-3):
            print(f"Early stopping at epoch {epoch+1}")
            break
    end_time_qantum = time() - star_time_qantum

if __name__ == "__main__":
    main()