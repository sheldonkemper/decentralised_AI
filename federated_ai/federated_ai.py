import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import syft as sy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cryptography.fernet import Fernet
import pickle

# 🔹 Connect to a PySyft Domain Node
node = sy.login(email="user@domain.com", password="password")

# 🔹 Generate & Store Encryption Key (Needs Secure Sharing)
key = Fernet.generate_key()
cipher = Fernet(key)

# 🔹 Define a Simple AI Model (Fully Compatible With PySyft 0.8+)
class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 🔹 Encrypt Parameters with Shape Preservation
def encrypt_parameters(parameters):
    """Encrypts model parameters while preserving shape and dtype."""
    param_data = {"arrays": [p.tolist() for p in parameters], "shapes": [p.shape for p in parameters]}
    serialized_data = pickle.dumps(param_data)
    encrypted_data = cipher.encrypt(serialized_data)
    return encrypted_data

# 🔹 Decrypt Parameters and Restore Shape
def decrypt_parameters(encrypted_data):
    """Decrypts and restores model parameters to their original shape."""
    decrypted_bytes = cipher.decrypt(encrypted_data)
    param_data = pickle.loads(decrypted_bytes)
    restored_params = [torch.tensor(arr, dtype=torch.float32).reshape(shape) for arr, shape in zip(param_data["arrays"], param_data["shapes"])]
    return restored_params

# 🔹 Load Federated Data
def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_data = datasets.MNIST(".", train=True, download=True, transform=transform)
    federated_train_data = train_data.federate(node)  # 🔹 Distribute dataset securely to the node
    return DataLoader(federated_train_data, batch_size=32, shuffle=True)

# 🔹 Train Model Federated Locally
def train(model, data_loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for images, labels in data_loader:
        images, labels = images.send(node), labels.send(node)  # 🔹 Send dataset to PySyft node
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        images.get(), labels.get()  # 🔹 Retrieve results back from the node

# 🔹 Federated AI Execution
if __name__ == "__main__":
    print("Starting Secure Federated AI Node with PySyft 0.8.4...")

    model = AIModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Load Federated Dataset
    train_loader = load_data()

    # Train the model
    train(model, train_loader, optimizer)

    # Encrypt and Store Model Parameters
    encrypted_parameters = encrypt_parameters([p.cpu().detach().numpy() for p in model.parameters()])
    print("Model training complete. Parameters encrypted successfully.")
