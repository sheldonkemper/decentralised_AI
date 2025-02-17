import flwr as fl
from flwr.server.server import ServerConfig

if __name__ == "__main__":
    # Create a ServerConfig object with 10 rounds
    config = ServerConfig(num_rounds=10)
    fl.server.start_server(server_address="127.0.0.1:8080", config=config)
