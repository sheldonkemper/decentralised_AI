import json
import requests
from web3 import Web3

# Function to upload a model file to IPFS via Infura
def upload_to_ipfs(model_path):
    with open(model_path, 'rb') as f:
        response = requests.post(
            "https://ipfs.infura.io:5001/api/v0/add",
            files={"file": f}
        )
    ipfs_hash = response.json()["Hash"]
    print(f"Model uploaded to IPFS with hash: {ipfs_hash}")
    return ipfs_hash

# Function to store the IPFS hash on the blockchain via a smart contract
def store_model_hash_on_chain(model_hash):
    # Connect to Ethereum using Infura
    infura_url = "https://goerli.infura.io/v3/YOUR_INFURA_PROJECT_ID"
    web3 = Web3(Web3.HTTPProvider(infura_url))
    contract_address = "0xYourContractAddress"
    # Replace with your contract's ABI JSON string or load from a file
    abi = json.loads('''[YOUR_CONTRACT_ABI]''')

    contract = web3.eth.contract(address=contract_address, abi=abi)
    account_address = "0xYourWalletAddress"
    nonce = web3.eth.getTransactionCount(account_address)

    txn = contract.functions.proposeModel(model_hash).buildTransaction({
        'chainId': 5,  # Goerli testnet chain ID
        'gas': 100000,
        'gasPrice': web3.toWei('20', 'gwei'),
        'nonce': nonce,
    })
    
    signed_txn = web3.eth.account.sign_transaction(txn, private_key="YOUR_PRIVATE_KEY")
    txn_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)
    print(f"Transaction sent with hash: {txn_hash.hex()}")

if __name__ == "__main__":
    # Example: Upload a trained model file and store its hash on chain
    model_hash = upload_to_ipfs("ai_model.pth")
    store_model_hash_on_chain(model_hash)
