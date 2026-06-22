import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List


class LinearRegression:
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        m, n = X.shape
        
        self.weights = np.zeros(n)
        self.bias = 0
        
        print("\n" + "="*60)
        print("LINEAR REGRESSION - TRAINING")
        print("="*60)
        print(f"Dataset: {m} samples, {n} features")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Iterations: {self.iterations}")
        print("-"*60)
        
        for iteration in range(self.iterations):
            y_pred = X @ self.weights + self.bias
            
            mse_loss = np.mean((y_pred - y) ** 2)
            self.losses.append(mse_loss)
            
            error = y_pred - y
            dw = (2 / m) * (X.T @ error)
            db = (2 / m) * np.sum(error)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (iteration + 1) % 200 == 0:
                print(f"Iteration {iteration + 1:4d} | Loss: {mse_loss:.6f}")
        
        print(f"Iteration {self.iterations:4d} | Loss: {mse_loss:.6f}")
        print("-"*60)
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


class LogisticRegression:
    
    def __init__(self, learning_rate: float = 0.01, iterations: int = 1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
        self.losses = []
        
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        m, n = X.shape
        
        self.weights = np.zeros(n)
        self.bias = 0
        
        print("\n" + "="*60)
        print("LOGISTIC REGRESSION - TRAINING")
        print("="*60)
        print(f"Dataset: {m} samples, {n} features")
        print(f"Classes: {np.unique(y).tolist()}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Iterations: {self.iterations}")
        print("-"*60)
        
        for iteration in range(self.iterations):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            epsilon = 1e-15
            bce_loss = -np.mean(y * np.log(y_pred + epsilon) + 
                               (1 - y) * np.log(1 - y_pred + epsilon))
            self.losses.append(bce_loss)
            
            error = y_pred - y
            dw = (1 / m) * (X.T @ error)
            db = (1 / m) * np.sum(error)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (iteration + 1) % 200 == 0:
                print(f"Iteration {iteration + 1:4d} | Loss: {bce_loss:.6f}")
        
        print(f"Iteration {self.iterations:4d} | Loss: {bce_loss:.6f}")
        print("-"*60)
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias:.6f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return self.sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def demo_linear_regression():
    print("\n" + "="*60)
    print("DEMO 1: LINEAR REGRESSION (House Price Prediction)")
    print("="*60)
    
    np.random.seed(42)
    m = 50
    
    X = np.column_stack([
        np.random.uniform(1000, 5000, m),
        np.random.uniform(0, 50, m)
    ])
    
    true_weights = np.array([150, -2000])
    true_bias = 50000
    noise = np.random.normal(0, 10000, m)
    y = X @ true_weights + true_bias + noise
    
    print("\nDATASET DETAILS:")
    print(f"   Training samples: {m}")
    print(f"   Features: 2 (house_size, house_age)")
    print(f"   True model: price = 150*size - 2000*age + 50000")
    print(f"\n   Sample data (first 5 rows):")
    print(f"   {'Size (sqft)':<15} {'Age (years)':<15} {'Price ($)':<15}")
    print("   " + "-"*45)
    for i in range(min(5, m)):
        print(f"   {X[i, 0]:<15.1f} {X[i, 1]:<15.1f} {y[i]:<15.1f}")
    
    model_lr = LinearRegression(learning_rate=0.00001, iterations=1000)
    model_lr.fit(X, y)
    
    print("\nPREDICTIONS ON TEST DATA:")
    test_samples = np.array([
        [3000, 10],
        [2000, 5],
        [4000, 20],
    ])
    predictions = model_lr.predict(test_samples)
    print(f"   {'Size':<10} {'Age':<10} {'Predicted Price':<20}")
    print("   " + "-"*40)
    for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
        print(f"   {sample[0]:<10.0f} {sample[1]:<10.0f} ${pred:>15,.2f}")
    
    train_score = model_lr.score(X, y)
    print(f"\nMODEL PERFORMANCE:")
    print(f"   R-squared Score (Training): {train_score:.4f}")
    print(f"   (1.0 is perfect, >0.9 is excellent)")
    
    return model_lr, X, y


def demo_logistic_regression():
    print("\n" + "="*60)
    print("DEMO 2: LOGISTIC REGRESSION (Tumor Classification)")
    print("="*60)
    
    np.random.seed(42)
    m = 100
    
    class_0_size = np.random.normal(2, 0.5, m // 2)
    class_0_age = np.random.normal(35, 8, m // 2)
    
    class_1_size = np.random.normal(4, 0.7, m // 2)
    class_1_age = np.random.normal(55, 10, m // 2)
    
    X = np.vstack([
        np.column_stack([class_0_size, class_0_age]),
        np.column_stack([class_1_size, class_1_age])
    ])
    
    y = np.hstack([np.zeros(m // 2), np.ones(m // 2)]).astype(int)
    
    print("\nDATASET DETAILS:")
    print(f"   Training samples: {m}")
    print(f"   Features: 2 (tumor_size, patient_age)")
    print(f"   Classes: 0 (Benign), 1 (Malignant)")
    print(f"   Class distribution: {np.sum(y == 0)} benign, {np.sum(y == 1)} malignant")
    print(f"\n   Sample data (first 3 benign, 3 malignant):")
    print(f"   {'Size (cm)':<15} {'Age (years)':<15} {'Label':<10}")
    print("   " + "-"*40)
    for i in range(3):
        print(f"   {X[i, 0]:<15.2f} {X[i, 1]:<15.1f} {int(y[i]):<10}")
    print("   ...")
    for i in range(m // 2, m // 2 + 3):
        print(f"   {X[i, 0]:<15.2f} {X[i, 1]:<15.1f} {int(y[i]):<10}")
    
    model_log = LogisticRegression(learning_rate=0.1, iterations=1000)
    model_log.fit(X, y)
    
    print("\nPREDICTIONS ON NEW PATIENTS:")
    test_patients = np.array([
        [1.5, 30],
        [5.0, 60],
        [3.5, 45],
    ])
    probas = model_log.predict_proba(test_patients)
    predictions = model_log.predict(test_patients)
    
    print(f"   {'Size':<10} {'Age':<10} {'P(Benign)':<15} {'P(Malignant)':<15} {'Prediction':<15}")
    print("   " + "-"*65)
    labels = ["Benign", "Malignant"]
    for sample, proba, pred in zip(test_patients, probas, predictions):
        print(f"   {sample[0]:<10.1f} {sample[1]:<10.0f} {1-proba:<15.4f} {proba:<15.4f} {labels[pred]:<15}")
    
    train_accuracy = model_log.score(X, y)
    print(f"\nMODEL PERFORMANCE:")
    print(f"   Training Accuracy: {train_accuracy:.2%}")
    
    return model_log, X, y


def visualize_results(model_lr, X_lr, y_lr, model_log, X_log, y_log):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Linear & Logistic Regression - Pure NumPy Implementation', 
                 fontsize=14, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(model_lr.losses, linewidth=2, color='#2E86AB')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Mean Squared Error', fontsize=11)
    ax.set_title('Linear Regression - Training Loss Curve', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    y_pred_lr = model_lr.predict(X_lr)
    ax.scatter(y_lr, y_pred_lr, alpha=0.6, s=60, color='#A23B72', edgecolors='black', linewidth=0.5)
    min_val = min(y_lr.min(), y_pred_lr.min())
    max_val = max(y_lr.max(), y_pred_lr.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price ($)', fontsize=11)
    ax.set_ylabel('Predicted Price ($)', fontsize=11)
    ax.set_title('Linear Regression - Actual vs Predicted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(model_log.losses, linewidth=2, color='#F18F01')
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Binary Cross-Entropy Loss', fontsize=11)
    ax.set_title('Logistic Regression - Training Loss Curve', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    colors = ['#2E86AB', '#F18F01']
    for class_val in [0, 1]:
        mask = y_log == class_val
        ax.scatter(X_log[mask, 0], X_log[mask, 1], 
                  label=f'Class {class_val}', alpha=0.6, s=60, 
                  color=colors[class_val], edgecolors='black', linewidth=0.5)
    
    x_min, x_max = X_log[:, 0].min() - 0.5, X_log[:, 0].max() + 0.5
    y_min, y_max = X_log[:, 1].min() - 2, X_log[:, 1].max() + 2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    Z = model_log.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0.5], colors='red', linewidths=2, linestyles='--')
    ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightsalmon'], alpha=0.3)
    
    ax.set_xlabel('Feature 1 (Size)', fontsize=11)
    ax.set_ylabel('Feature 2 (Age)', fontsize=11)
    ax.set_title('Logistic Regression - Decision Boundary', fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('regression_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved as 'regression_results.png'")
    plt.show()


if __name__ == "__main__":
    model_lr, X_lr, y_lr = demo_linear_regression()
    
    model_log, X_log, y_log = demo_logistic_regression()
    
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    visualize_results(model_lr, X_lr, y_lr, model_log, X_log, y_log)
    
    print("\nDemo completed successfully!")
