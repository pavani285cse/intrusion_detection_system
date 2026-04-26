# EGNNN-GROA-BGPoW-IDS-CC Framework

**Evolutionary Gravitational Neocognitron Neural Network espoused Blockchain-based Intrusion Detection Framework for Enhancing Cybersecurity in a Cloud Computing Environment**

This repository contains the full start-to-end implementation of the cloud cybersecurity intrusion detection pipeline proposed in the aforementioned research paper. The framework combines advanced feature engineering, a metaheuristic-optimized deep learning classifier, and a lightweight blockchain logging mechanism.

---

## 🏗️ Start-to-End Workflow

The main entry point for the entire pipeline is `main.py`. Running `python main.py` triggers the following end-to-end workflow:

### Stage 1: Data Preprocessing (DRFLLS)
**File:** `stage1.py`  
**Core Function:** `run_stage1()`
1. **Data Loading:** Loads the raw NSL-KDD dataset (`KDDTrain+.txt` and `KDDTest+.txt`) from the `dataset/` directory.
2. **Missing Value Processing:** Temporarily injects and then imputes missing values using Local Least Squares (7 neighbors) to simulate real-world data corruption resilience.
3. **Redundant Feature Removal:** Uses Pearson Correlation (with a threshold of 0.9) to eliminate highly collinear features.
4. **Target Encoding:** Normalizes all remaining features into a bounded range `[0, 1]` using Min-Max scaling. Outputs clean data frames.

### Stage 2: Feature Selection (DRFSA)
**File:** `stage2.py`  
**Core Function:** `run_stage2()`
1. **Label Encoding:** Encodes string classes ("Normal", "DoS", "Probe", "R2L", "U2R") into integer indices integers `0-4`.
2. **Feature Ranking:** Evaluates monotonic relationships using Spearman's Rank Correlation Coefficient (SRCC).
3. **Iterative Backward Elimination:** Removes the lowest-ranked features step-by-step while testing performance against a quick Random Forest classifier on a validation split.
4. **Reconciliation:** Filters down to exactly the highly performing subset aligned with the features highlighted in the paper. Returns the `X_train`, `X_test`, `y_train`, and `y_test` datasets used for neural network training.

### Stage 3: Evolutionary Gravitational Neural Network (EGNNN)
**File:** `stage3_egnnn.py`  
**Core Class:** `EGNNNClassifier`
*   **Architecture Setup:** Instantiates a 5-class PyTorch neural network with 5 hidden layers `[64, 128, 256, 128, 64]`, ReLU activations, and Dropout (`0.3`).
*   **Method: `gravitational_update()`:** Evaluates fitness for every neuron using an approximation metric derived from layer weight mass, dynamically computing gravitational attraction parameters to update velocities and weights at each training epoch.
*   **Method: `evolutionary_mutation()`:** Provides a 3% chance at each epoch to add random Gaussian noise to parameters to escape local minima.
*   **Method: `fit()`:** Uses `Adam` optimizers with inverse class-frequency weights to adjust for traffic type imbalance over up to 300 epochs alongside early stopping.

### Stage 4: Optimization (GROA)
**File:** `stage4_groa.py`  
**Core Class:** `GROAOptimizer`
*   **Method: `initialize_population()`:** Spawns 30 pseudo-fishes (different full weight configurations for the un-trained EGNNN).
*   **Method: `optimize()`:** Iterates 100 times to hunt for the perfect set of neural network initial weights via the GarraRufa algorithm.
*   **Internal Workflow:** It tracks local and global bests (`pbest`, `gbest`), performs positional velocity updates factoring in energy decay, executes random local perturbations for fine-tuning, and eventually feeds the globally best weights directly into the `EGNNNClassifier`.

### Stage 5: Security Logging via Blockchain (BGPoW)
**File:** `stage5_blockchain.py`  
**Core Class:** `BGPoWBlockchain`
1. **Initialization:** Spawns 5 validator "nodes", evaluating each on pseudo-metrics like cybersecurity score and energy parameters.
2. **Method: `add_block()`:** Appends any traffic flagged as an intrusion (everything except "Normal") to the next block. 
3. **Method: `mine_block()`:** Utilizes a highly optimized subset hashing proof of work that prevents infinite loops via bounding checks against `difficulty_target`. 
4. **Method: `reach_consensus()`:** Asks the nodes sequentially if the newest mined hash meets criteria, penalizing or rewarding the global chain multiplier `phi`.
5. **Method: `validate_chain()`:** Recomputes `merkle_roots` and `SHA256` iterations sequentially across all blocks to guarantee immutable data integrity.

### Stage 6: Evaluation & Reporting
**File:** `evaluation.py`  
**Core Class:** `EvaluationEngine`
1. **Method: `run_full_evaluation()`:** Executes predictions across `X_test` and funnels through metric evaluations.
2. **Metrics Display:** Captures per-class details (Sensitivity, Specificity, Accuracy, F1) as well as Macro-Average metrics.
3. **Comparison Engine:** Prints a formatted performance comparison showing where the GROA-EGNNN model stands alongside baseline counterparts (BiLSTM, DBN-ResNet).
4. **Visual Plots Generation:** 
    - `evaluation_roc.png` (AUC Curve mapping via One-Vs-Rest strategies)
    - `evaluation_confusion_matrix.png` (Seaborn Heatmap of predictions)
    - `evaluation_training_history.png` (Training and Validation F1 vs Loss metrics tracking)

---

## 🚀 Execution
Ensure you have the required metrics configured. Run all of it using:
```bash
python main.py
```
