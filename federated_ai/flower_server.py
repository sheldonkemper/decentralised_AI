import flwr as fl

if __name__ == "__main__":
    # Start the Flower server on 127.0.0.1:8080
    fl.server.start_server(server_address="127.0.0.1:8080")
