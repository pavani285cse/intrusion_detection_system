import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class EGNNN(nn.Module):
    def __init__(self, input_dim=20, layer_sizes=[64, 128, 256, 128, 64], n_classes=5):
        super(EGNNN, self).__init__()
        self.layers = nn.ModuleList()
        in_dim = input_dim
        for size in layer_sizes:
            self.layers.append(nn.Linear(in_dim, size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.3))
            in_dim = size
        self.output_layer = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

class EGNNNClassifier:
    def __init__(self, input_dim=20, layer_sizes=[64, 128, 256, 128, 64], n_classes=5):
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.n_classes = n_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        
        # Velocity for gravitational update tracking
        self.velocities = []
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.velocities.append(torch.zeros_like(param).to(self.device))
            
    def build_model(self):
        return EGNNN(self.input_dim, self.layer_sizes, self.n_classes).to(self.device)

    def gravitational_update(self, epoch):
        """Gravitational Weight Update Mechanism"""
        G0 = 100.0
        alpha = 0.02
        epsilon = 1e-10
        G_w = G0 * np.exp(-alpha * epoch)
        
        with torch.no_grad():
            v_idx = 0
            for name, param in self.model.named_parameters():
                if 'weight' in name:
                    n_neurons = param.shape[0]
                    
                    # Compute fitness per neuron using weight norms combined with global state
                    # as a proxy to computationally expensive individual F1 ablation.
                    fitness = torch.norm(param, dim=1) 
                    
                    worst_fitness = torch.min(fitness)
                    best_fitness = torch.max(fitness)
                    
                    if best_fitness == worst_fitness:
                        mass = torch.ones_like(fitness) / n_neurons
                    else:
                        mass = (fitness - worst_fitness) / (best_fitness - worst_fitness + epsilon)
                        
                    total_mass = torch.sum(mass) + epsilon
                    M = mass / total_mass  # M_j(w) = mass_j(w) / sum(all masses in layer)
                    
                    # Calculate Acceleration of neuron i
                    a = torch.zeros_like(param)
                    
                    # Distance between weight vectors of neurons
                    dist = torch.cdist(param, param, p=2) 
                    
                    for i in range(n_neurons):
                        r = torch.rand(n_neurons).to(self.device)
                        
                        direction = param - param[i] # distance vector
                        
                        # F_ij / M_i part: F_ij(w) = G(w) * M_i(w) * M_j(w) / (dist + eps)
                        mag = r * G_w * M / (dist[i] + epsilon)
                        mag[i] = 0.0 # Exclude self relation j != i
                        
                        a[i] = torch.sum(mag.unsqueeze(1) * direction, dim=0)
                    
                    # Velocity update: v_i(w+1) = rand() * v_i(w) + a_i(w)
                    r_v = torch.rand(n_neurons, 1).to(self.device)
                    self.velocities[v_idx] = r_v * self.velocities[v_idx] + a
                    
                    # Clamp velocities and weights to prevent exploding gradients
                    self.velocities[v_idx] = torch.clamp(self.velocities[v_idx], -0.1, 0.1)
                    
                    # Position (weight) update: x_i(w+1) = x_i(w) + v_i(w+1)
                    param.add_(self.velocities[v_idx])
                    param.data = torch.clamp(param.data, -5.0, 5.0)
                    
                    v_idx += 1

    def evolutionary_mutation(self):
        """After each epoch, apply mutation to weights with probability 0.03"""
        with torch.no_grad():
            for param in self.model.parameters():
                mask = (torch.rand(param.shape) < 0.03).float().to(self.device)
                noise = torch.normal(mean=0.0, std=0.01, size=param.shape).to(self.device)
                param.add_(mask * noise)

    def fit(self, X_train, y_train, X_val, y_val):
        """Train the neural network"""
        # Handle class imbalance using weighted CrossEntropyLoss
        class_counts = np.bincount(y_train, minlength=self.n_classes)
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        weights = weights / np.sum(weights) * self.n_classes
        class_weights = torch.FloatTensor(weights).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.LongTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        patience = 20
        best_val_f1 = -1
        epochs_no_improve = 0
        best_weights = None
        history = {'loss': [], 'f1': [], 'val_f1': []}
        
        for epoch in range(300):
            self.model.train()
            epoch_losses = []
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            self.evolutionary_mutation()
            self.gravitational_update(epoch)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()
                val_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
                
                train_outputs = self.model(X_train_t)
                train_preds = torch.argmax(train_outputs, dim=1).cpu().numpy()
                train_f1 = f1_score(y_train, train_preds, average='macro', zero_division=0)
                
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            history['f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/300 - Loss: {avg_loss:.4f} - Train F1: {train_f1:.4f} - Val F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                torch.save(best_weights, "egnnn_best_weights.pt")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
                    
        if best_weights:
            self.model.load_state_dict(best_weights)
            
        return history

    def predict(self, X_test):
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_test_t)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X_test):
        self.model.eval()
        X_test_t = torch.FloatTensor(X_test).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_test_t)
            proba = torch.softmax(outputs, dim=1).cpu().numpy()
        return proba

    def load_weights(self, path_or_array):
        """
        Load weights from either a saved .pt file path or a flat numpy array.
        Handles both GROA optimizer (array) and live IDS (string path).
        """
        if isinstance(path_or_array, str):
            try:
                state_dict = torch.load(path_or_array, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[EGNNN] Successfully loaded weights from '{path_or_array}'.")
            except Exception as e:
                print(f"[EGNNN] ERROR loading weights from '{path_or_array}': {e}")
        else:
            # Handle GROA array loading
            weights_vector = path_or_array
            idx = 0
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    length = param.numel()
                    param.copy_(torch.tensor(weights_vector[idx:idx+length]).view(param.shape).to(self.device))
                    idx += length

    def eval_mode(self):
        """
        Sets the PyTorch model to evaluation mode.
        Should be called once at startup before processing real-time packets.
        Disables dropout layers and prepares for inference.
        """
        self.model.eval()

    def predict_single(self, features_tensor):
        """
        Runs a forward pass on a single packet feature tensor.
        Input: torch.Tensor of shape (1, 18)
        Output: tuple (label_index, confidence, probabilities)
        """
        # Ensure we are not tracking gradients for inference
        with torch.no_grad():
            features_tensor = features_tensor.to(self.device)
            # Run the forward pass to get raw logits
            logits = self.model(features_tensor)
            
            # Apply softmax to get normalized probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Find the class index with the highest probability
            label_index = int(np.argmax(probabilities))
            
            # Get the confidence value (max probability)
            confidence = float(probabilities[label_index])
            
        return label_index, confidence, probabilities
