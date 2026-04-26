from sklearn.model_selection import train_test_split
from stage1 import run_stage1
from stage2 import run_stage2
from stage3_egnnn import EGNNNClassifier
from stage4_groa import GROAOptimizer
from stage5_blockchain import BGPoWBlockchain
from evaluation import EvaluationEngine
import numpy as np

def main():
    print("--- Running Stage 1: Preprocessing ---")
    # Using KDDTrain+.txt and KDDTest+.txt from the dataset folder
    s1 = run_stage1("dataset/KDDTrain+.txt", "dataset/KDDTest+.txt")
    
    print("--- Running Stage 2: Feature Selection ---")
    s2 = run_stage2(s1["train_df"], s1["test_df"])
    
    # Prepare arrays
    X_train = s2["train_df"].drop("label", axis=1)
    y_train = s2["y_train"]
    X_test = s2["test_df"].drop("label", axis=1)
    y_test = s2["y_test"]
    
    # Ensure they are numpy arrays
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.values
        X_test = X_test.values
    if not isinstance(y_train, np.ndarray):
        y_train = y_train.values
        y_test = y_test.values
        
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
    )
    
    print("--- Running Stage 3 & 4: Initializing EGNNN and GROA Optimization ---")
    input_dim = X_train.shape[1]
    model = EGNNNClassifier(input_dim=input_dim, layer_sizes=[64, 128, 256, 128, 64], n_classes=5)
    
    optimizer = GROAOptimizer(model, X_train, y_train, X_val, y_val, pop_size=30, max_iter=100)
    best_weights = optimizer.optimize()
    model.load_weights(best_weights)
    
    print("--- Training Final EGNNN Model ---")
    history = model.fit(X_train, y_train, X_val, y_val)
    
    print("--- Running Stage 5: Blockchain Logging ---")
    blockchain = BGPoWBlockchain()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    for i, (pred, true_idx) in enumerate(zip(y_pred, y_test)):
        label_name = s2["label_encoder"].inverse_transform([pred])[0]
        if label_name != "Normal":
            blockchain.add_block({
                "sample_id": i,
                "predicted_label": label_name,
                "true_label": s2["label_encoder"].inverse_transform([true_idx])[0],
                "confidence": float(y_proba[i].max()),
                "is_intrusion": True
            })
            
    blockchain.print_chain_summary()
    
    print("--- Evaluation ---")
    engine = EvaluationEngine(model, blockchain, s2["label_encoder"])
    engine.run_full_evaluation(X_test, y_test)
    engine.plot_training_history(history)

if __name__ == "__main__":
    main()
