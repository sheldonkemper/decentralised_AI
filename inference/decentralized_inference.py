import torch
import torch.nn as nn
import requests
import io

# Define the same AIModel architecture used during training
class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def download_model(ipfs_hash: str) -> bytes:
    """
    Downloads the model file from IPFS using a public gateway.
    Replace the URL with your preferred gateway if necessary.
    """
    url = f"https://ipfs.io/ipfs/{ipfs_hash}"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"Successfully downloaded model from IPFS: {ipfs_hash}")
        return response.content
    else:
        raise Exception(f"Failed to download model from IPFS (status code: {response.status_code})")

if __name__ == "__main__":
    # Replace with your model's IPFS hash
    model_ipfs_hash = "QmYourModelHashHere"

    # Download the model file from IPFS
    model_bytes = download_model(model_ipfs_hash)
    
    # Load the model state from the downloaded bytes
    buffer = io.BytesIO(model_bytes)
    state_dict = torch.load(buffer, map_location=torch.device('cpu'))
    
    # Initialize the model and load the state dict
    model = AIModel()
    model.load_state_dict(state_dict)
    model.eval()
    
    # Example inference: use a dummy input (or load your own data)
    dummy_input = torch.randn(1, 28 * 28)
    output = model(dummy_input)
    print("Inference output:", output)
