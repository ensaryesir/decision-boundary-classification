# Interactive MLP Decision Boundary Classification

An interactive Python application that allows users to create multi-class datasets by clicking points on a 2D plane and visualize real-time decision boundaries of a NumPy-based Multi-Layer Perceptron (MLP) neural network.

## 🎯 Features

### Interactive Data Creation
- **Multi-class support**: Add/remove classes dynamically with automatic color assignment
- **Point placement**: Left-click to add points to the selected class
- **Real-time visualization**: Decision boundaries update automatically as you add points

### Neural Network Implementation
- **Pure NumPy MLP**: Custom implementation without PyTorch/TensorFlow dependencies
- **Multiple activations**: ReLU, Tanh, and Logistic (Sigmoid) functions
- **Flexible architecture**: Configurable hidden layers (e.g., `5,3` for two hidden layers)
- **Advanced training**: Batch processing, L2 regularization, and configurable hyperparameters

### User Interface
- **Interactive controls**: Radio buttons for class selection and activation functions
- **Parameter tuning**: Real-time adjustment of learning rate, epochs, batch size, and L2 regularization
- **Information panel**: Separate window showing model parameters and their effects
- **Visual feedback**: Color-coded decision regions and scatter plots

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd decision-boundary-classification

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Windows (Command Prompt):
.venv\Scripts\activate.bat
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Running the Application
```bash
python app.py
```

### Interface Overview
The application opens with two windows:
1. **Main Window**: Interactive plot with control panel
2. **Info Window**: Model parameters and recommendations table

### How to Use

#### 1. Class Management
- **Select Class**: Use radio buttons to choose the active class
- **Add Class**: Click "+ Class" to create a new class
- **Remove Class**: Click "- Class" to delete the selected class (minimum 2 classes required)

#### 2. Data Collection
- **Add Points**: Left-click anywhere on the plot to add a point to the selected class
- **Clear Data**: Click "Clear" to remove all points and start over

#### 3. Model Configuration
- **Activation Function**: Choose between ReLU, Tanh, or Logistic
- **Hidden Layers**: Enter comma-separated values (e.g., `5,3` for two layers with 5 and 3 neurons)
- **Learning Rate**: Controls training speed and stability
- **Epochs**: Number of training iterations
- **Batch Size**: Number of samples per training batch
- **L2 Regularization**: Prevents overfitting

#### 4. Training
- **Automatic Training**: Model trains automatically when you add points or change parameters
- **Manual Retrain**: Click "Retrain" to reinitialize and retrain the model

## 🔧 Technical Details

### MLP Implementation
- **Forward Pass**: Standard feedforward with configurable activation functions
- **Backpropagation**: Manual implementation of gradient descent
- **Weight Initialization**: He initialization for ReLU, Xavier for other activations
- **Loss Function**: Cross-entropy with softmax for multi-class classification

### Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Fast, prevents vanishing gradients
- **Tanh**: `f(x) = tanh(x)` - Bounded output, good for hidden layers
- **Logistic**: `f(x) = 1/(1 + e^(-x))` - Smooth, bounded between 0 and 1

### Visualization
- **Decision Boundaries**: Contour plots showing classification regions
- **Grid Resolution**: 200x200 grid for smooth boundary visualization
- **Color Mapping**: Automatic color assignment for up to 10 classes

## 📊 Parameter Guidelines

| Parameter | Typical Range | Effect |
|-----------|---------------|---------|
| Hidden Layers | 2-4 layers, 4-256 neurons | More layers = higher complexity, overfitting risk |
| Learning Rate | 0.001 - 0.1 | Higher = faster training, less stability |
| Epochs | 50 - 1000 | More epochs = better accuracy, overfitting risk |
| Batch Size | 16 - 128 | Smaller = faster updates, more noise |
| L2 Regularization | 0.0001 - 0.1 | Higher = simpler model, better generalization |

## 🎨 Example Use Cases

1. **Educational Tool**: Learn how neural networks create decision boundaries
2. **Algorithm Testing**: Compare different activation functions and architectures
3. **Data Visualization**: Understand how data distribution affects classification
4. **Parameter Tuning**: Experiment with hyperparameters interactively

## 🛠️ Dependencies

- **numpy** (≥1.24): Numerical computations and array operations
- **matplotlib** (≥3.7): Plotting and visualization
- **scikit-learn** (≥1.3): Additional utilities (though not used in core implementation)

## 📝 Notes

- The MLP is implemented from scratch using only NumPy
- No deep learning frameworks (PyTorch/TensorFlow) are required
- The application is designed for educational and experimental purposes
- Real-time training may be slower with large datasets or complex architectures

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.

## 📄 License

This project is open source and available under the MIT License.
