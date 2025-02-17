import subprocess

if __name__ == "__main__":
    # Use subprocess to call the new Flower Supernode CLI command.
    # This will start the Flower server (Supernode) on 127.0.0.1:8080 using an insecure channel.
    subprocess.run(["flower-supernode", "--insecure", "--superlink=127.0.0.1:8080"])
