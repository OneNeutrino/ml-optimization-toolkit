"""
Bayesian Optimization implementation for hyperparameter tuning.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

class BayesianOptimizer:
    def __init__(self, parameter_space, objective_function):
        """
        Initialize Bayesian Optimizer.
        
        Args:
            parameter_space (dict): Dictionary defining parameter ranges
            objective_function (callable): Function to optimize
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.X_observed = []
        self.y_observed = []
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            n_restarts_optimizer=10,
            random_state=42
        )
    
    def _acquisition_function(self, X, method='ucb', kappa=2.576):
        """
        Compute acquisition function value.
        
        Args:
            X (array-like): Points to evaluate
            method (str): Acquisition function type ('ucb', 'ei', 'poi')
            kappa (float): Exploration-exploitation trade-off parameter
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if method == 'ucb':
            return mu + kappa * sigma
        elif method == 'ei':
            # Expected Improvement
            y_best = np.max(self.y_observed)
            imp = mu - y_best
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            return ei
        elif method == 'poi':
            # Probability of Improvement
            y_best = np.max(self.y_observed)
            Z = (mu - y_best) / sigma
            return norm.cdf(Z)
    
    def optimize(self, n_iterations=50, acquisition='ucb'):
        """
        Run optimization process.
        
        Args:
            n_iterations (int): Number of optimization iterations
            acquisition (str): Acquisition function to use
        """
        for i in range(n_iterations):
            # Generate random candidates
            X_candidates = self._generate_candidates()
            
            # Select next point
            if len(self.X_observed) > 0:
                self.gp.fit(self.X_observed, self.y_observed)
                acq_values = self._acquisition_function(X_candidates, method=acquisition)
                next_idx = np.argmax(acq_values)
            else:
                next_idx = np.random.randint(len(X_candidates))
            
            # Evaluate objective
            X_next = X_candidates[next_idx]
            y_next = self.objective_function(X_next)
            
            # Update observations
            self.X_observed.append(X_next)
            self.y_observed.append(y_next)
        
        # Find best parameters
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]