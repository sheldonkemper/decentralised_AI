import flwr as fl

# Define a minimal dummy client that satisfies Flower's API.
class DummyClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        return []
    def set_parameters(self, parameters):
        pass
    def fit(self, parameters, config):
        return [], 0, {}
    def evaluate(self, parameters, config):
        return 0.0, 0, {}

if __name__ == "__main__":
    print("Starting Dummy Client...")
    client = DummyClient().to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
