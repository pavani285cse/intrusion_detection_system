import hashlib
import time
import json
import random

class BGPoWBlockchain:
    def __init__(self, n_nodes=5, difficulty_target=0x0000FFFF * (2 ** 208)):
        self.n_nodes = n_nodes
        self.difficulty_target = difficulty_target
        self.chain = []
        
        # Node properties for consensus simulation
        self.nodes = []
        # Ensure random reproducibility based on previous seeds if set globally
        for i in range(self.n_nodes):
            self.nodes.append({
                "id": i,
                "energy": random.uniform(0.5, 1.0),
                "cybersecurity_score": random.uniform(0.7, 1.0)
            })
            
        self.phi = 1.0 # Reward/penalty multiplier
        self.detection_coefficient = 0.8
        
        # Statistics
        self.total_blocks_mined = 0
        self.consensus_success = 0
        self.consensus_attempts = 0
        
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = {
            "index": 0,
            "previous_hash": "0" * 64,
            "timestamp": time.time(),
            "nonce": 0,
            "merkle_root": "0" * 64,
            "detection_result": None,
            "hash": ""
        }
        genesis_block["hash"] = self.compute_hash(genesis_block)
        self.chain.append(genesis_block)

    def compute_hash(self, block):
        # SHA256(SHA256(previous_hash + timestamp + nonce + merkle_root))
        data_string = f"{block['previous_hash']}{block['timestamp']}{block['nonce']}{block['merkle_root']}"
        h1 = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
        h2 = hashlib.sha256(h1.encode('utf-8')).hexdigest()
        return h2

    def compute_merkle_root(self, detection_result):
        # Only one detection result per block in this application
        res_string = json.dumps(detection_result, sort_keys=True)
        return hashlib.sha256(res_string.encode('utf-8')).hexdigest()

    def mine_block(self, previous_hash, timestamp, merkle_root):
        # Find nonce such that hash_hex = SHA256(SHA256(data_string + str(nonce)))
        # data_string is structured here for mining loop efficiency
        data_string = f"{previous_hash}{timestamp}"
        
        nonce = 0
        max_attempts = 100000
        
        for _ in range(max_attempts):
            test_string = f"{data_string}{nonce}{merkle_root}"
            h1 = hashlib.sha256(test_string.encode('utf-8')).hexdigest()
            hash_hex = hashlib.sha256(h1.encode('utf-8')).hexdigest()
            
            if int(hash_hex, 16) < self.difficulty_target:
                return nonce, hash_hex
                
            nonce += 1
            
        return nonce, hash_hex # Return best effort to avoid infinite loop

    def select_nodes(self):
        # Node selection probability: P_ji = detection_coefficient / (energy_i * cybersecurity_score_i)
        probs = []
        for node in self.nodes:
            p = self.detection_coefficient / (node["energy"] * node["cybersecurity_score"])
            probs.append(p)
            
        sum_p = sum(probs)
        probs = [p / sum_p for p in probs]
        return probs

    def reach_consensus(self, block):
        self.consensus_attempts += 1
        probs = self.select_nodes()
        
        # Simulate voting based on checking hash and difficulty
        hash_val = int(block["hash"], 16)
        is_valid = hash_val < self.difficulty_target
        
        votes = 0
        for i in range(self.n_nodes):
            vote = is_valid 
            if vote:
                votes += 1
                
        consensus_reached = (votes > self.n_nodes / 2)
        
        if consensus_reached:
            self.phi = min(1.0, self.phi * 1.05) # Cap reward at 1.0
            self.consensus_success += 1
        else:
            self.phi = self.phi * 0.95 # Apply penalty
            
        return consensus_reached

    def add_block(self, detection_result):
        # Add one block per detected intrusion
        prev_block = self.chain[-1]
        
        timestamp = time.time()
        merkle_root = self.compute_merkle_root(detection_result)
        
        nonce, hash_hex = self.mine_block(prev_block["hash"], timestamp, merkle_root)
        
        new_block = {
            "index": len(self.chain),
            "previous_hash": prev_block["hash"],
            "timestamp": timestamp,
            "nonce": nonce,
            "merkle_root": merkle_root,
            "detection_result": detection_result,
            "hash": hash_hex
        }
        
        if self.reach_consensus(new_block):
            self.chain.append(new_block)
            self.total_blocks_mined += 1

    def validate_chain(self):
        # Validate chain integrity
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            if current["previous_hash"] != previous["hash"]:
                return False
                
            data_string = f"{current['previous_hash']}{current['timestamp']}{current['nonce']}{current['merkle_root']}"
            h1 = hashlib.sha256(data_string.encode('utf-8')).hexdigest()
            h2 = hashlib.sha256(h1.encode('utf-8')).hexdigest()
            
            if current["hash"] != h2:
                return False
                
        return True

    def print_chain_summary(self):
        print("\n--- Blockchain Summary ---")
        print(f"Total blocks mined (Intrusions logged): {self.total_blocks_mined}")
        print(f"Consensus rate: {(self.consensus_success / max(1, self.consensus_attempts)) * 100:.2f}%")
        print(f"Chain Integrity Valid: {self.validate_chain()}")
        print("--------------------------\n")
