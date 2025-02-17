import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define a simple neural network for MNIST classification
class AIModel(nn.Module):
    def __init__(self):
        super(AIModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Federated Learning Client using Flower
class FLClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = AIModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        # Prepare the dataset (MNIST) for local training
        transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.view(-1))
        ])
        self.train_data = datasets.MNIST(".", train=True, download=True, transform=transform)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=32, shuffle=True
        )

    # Accepts an optional config parameter
    def get_parameters(self, config=None):
        params = [p.detach().numpy() for p in self.model.parameters()]
        print("[GET PARAMETERS] Returning model parameters")
        return params

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p)
        print("[SET PARAMETERS] Updated model parameters")

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        batches = 0
        for images, labels in self.train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            batches += 1
        avg_loss = total_loss / batches if batches > 0 else 0.0
        print(f"[FIT] Average Loss: {avg_loss:.4f}")
        return self.get_parameters(), len(self.train_data), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct, total = 0, 0
        print("[EVALUATE] Starting evaluation...")
        with torch.no_grad():
            for images, labels in self.train_loader:
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total > 0:
            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct / total
        else:
            avg_loss, accuracy = float('inf'), 0.0
        print(f"[EVALUATE] Average Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
        # Always return a valid tuple: (loss, num_examples, metrics)
        return float(avg_loss), total, {"accuracy": accuracy}

if __name__ == "__main__":
    print("Starting Federated AI Client...")
    # This continues to use the deprecated API call for now
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient())
