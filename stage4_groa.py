import numpy as np
import torch
from sklearn.metrics import f1_score

class GROAOptimizer:
    def __init__(self, model, X_train, y_train, X_val, y_val, pop_size=30, max_iter=100):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = torch.FloatTensor(X_val).to(model.device)
        self.y_val = y_val
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.energy_decay = 0.95
        self.local_search_prob = 0.7
        self.step_size = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        
        # Ensure random reproducibility
        np.random.seed(42)
        
        # Total number of parameters in the EGNNN model
        self.dim = 0
        for name, param in self.model.model.named_parameters():
            self.dim += param.numel()
            
        self.population = []
        self.velocities = []
        self.pbest_positions = []
        self.pbest_fitness = []
        
        self.gbest_position = None
        self.gbest_fitness = -1.0

    def initialize_population(self):
        # 1. Initialize population: 30 random weight configurations for EGNNN
        for _ in range(self.pop_size):
            pos = np.random.normal(0, 0.1, self.dim)
            vel = np.zeros(self.dim)
            self.population.append(pos)
            self.velocities.append(vel)
            self.pbest_positions.append(pos.copy())
            self.pbest_fitness.append(-1.0)
            
    def evaluate_fitness(self, weights):
        # 2. Evaluate each fish's fitness = macro F1-score on validation set
        self.model.load_weights(weights)
        self.model.model.eval()
        with torch.no_grad():
            outputs = self.model.model(self.X_val)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
        return f1_score(self.y_val, preds, average='macro', zero_division=0)

    def velocity_update(self, i, k):
        # 4. Velocity update per fish per iteration:
        # u_i(k+1) = alpha * u_i(k) + c1 * r1 * (F_i(k) - Z_i(k)) + c2 * r2 * (L(k) - Z_i(k))
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        alpha = self.energy_decay
        u_i = self.velocities[i]
        Z_i = self.population[i]
        F_i = self.pbest_positions[i]
        L = self.gbest_position
        
        new_u_i = alpha * u_i + self.c1 * r1 * (F_i - Z_i) + self.c2 * r2 * (L - Z_i)
        self.velocities[i] = new_u_i
        
        # 5. Position update: Z_i(k+1) = Z_i(k) + step_size * u_i(k+1)
        self.population[i] = self.population[i] + self.step_size * new_u_i

    def local_search(self):
        # 6. Local search: with probability 0.7, perturb best fish slightly
        if np.random.rand() < self.local_search_prob:
            Z_best_new = self.gbest_position + np.random.normal(0, 0.01, self.dim) * self.step_size
            new_fitness = self.evaluate_fitness(Z_best_new)
            if new_fitness > self.gbest_fitness:
                self.gbest_position = Z_best_new
                self.gbest_fitness = new_fitness

    def optimize(self):
        self.initialize_population()
        
        for k in range(self.max_iter):
            for i in range(self.pop_size):
                fit = self.evaluate_fitness(self.population[i])
                
                # 3. Track personal best and global best
                if fit > self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fit
                    self.pbest_positions[i] = self.population[i].copy()
                    
                if fit > self.gbest_fitness:
                    self.gbest_fitness = fit
                    self.gbest_position = self.population[i].copy()
            
            for i in range(self.pop_size):
                self.velocity_update(i, k)
                
            self.local_search()
            
            # 7. Convergence update (feeding behavior)
            # Clip position values to valid weight range [-5, 5]
            for i in range(self.pop_size):
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                
            if (k + 1) % 10 == 0:
                print(f"GROA Iteration {k+1}/{self.max_iter} - Best Fitness: {self.gbest_fitness:.4f}")
                
        # 8. Return best weight vector
        return self.gbest_position
