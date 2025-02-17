// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DecentralizedAI {
    struct ModelUpdate {
        uint256 id;
        string modelHash;
        address proposer;
        uint256 votes;
        bool accepted;
    }

    mapping(uint256 => ModelUpdate) public updates;
    mapping(address => uint256) public reputation;
    uint256 public updateCount;
    address public owner;

    event ModelProposed(uint256 id, string modelHash, address proposer);
    event ModelAccepted(uint256 id, string modelHash);

    constructor() {
        owner = msg.sender;
    }

    function proposeModel(string memory modelHash) public {
        updateCount++;
        updates[updateCount] = ModelUpdate(updateCount, modelHash, msg.sender, 0, false);
        emit ModelProposed(updateCount, modelHash, msg.sender);
    }

    function voteOnModel(uint256 id) public {
        require(updates[id].id == id, "Model update does not exist");
        require(!updates[id].accepted, "Already accepted");

        updates[id].votes += 1;
        reputation[msg.sender] += 1;

        if (updates[id].votes > 10) { // Threshold for acceptance
            updates[id].accepted = true;
            emit ModelAccepted(id, updates[id].modelHash);
        }
    }

    function getModelHash(uint256 id) public view returns (string memory) {
        require(updates[id].accepted, "Model not accepted yet");
        return updates[id].modelHash;
    }
}
