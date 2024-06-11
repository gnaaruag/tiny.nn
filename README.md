# Tiny NN

This repository contains a fully functional implementation of a Multilayer Perceptron (MLP) neural network built from scratch using only Python, NumPy, and Pandas.

## Installation

```
git clone https://github.com/gnaaruag/tiny.nn.git
cd neural-network-from-scratch
```
## Example Usage
1. Load the Dataset:

```
import pandas as pd
data = pd.read_csv("./Datasets/mnist.csv")
```

2. Define Network Architecture:

```
from Layers.DenseLayer import DenseLayer
from ActivationClasses.ReluActivation import ReluActivation
from ActivationClasses.SoftmaxActivation import Softmax
from Loss.CategoricalCrossEntropy import CategoricalCrossEntropy
from Optimizer.Adam import Optimizer_Adam

l1 = DenseLayer(784, 128)
act1 = ReluActivation()

l2 = DenseLayer(128, 64)
act2 = ReluActivation()

l3 = DenseLayer(64, 10)
act3 = Softmax()

loss = CategoricalCrossEntropy()
optimizer = Optimizer_Adam(learning_rate=0.001)
```

3. Train the Model:

```
epochs = 10
for epoch in range(epochs + 1):
    l1.forward_prop(X_train)
    act1.forward(l1.output)
    l2.forward_prop(act1.output)
    act2.forward(l2.output)
    l3.forward_prop(act2.output)
    act3.forward(l3.output)

    loss_fin = loss.calculate(act3.output, y_train)

    loss.backward(act3.output, y_train)
    act3.backward(loss.dinputs)
    l3.backward(act3.dinputs)
    act2.backward(l3.dinputs)
    l2.backward(act2.dinputs)
    act1.backward(l2.dinputs)
    l1.backward(act1.dinputs)

    optimizer.update_params(l1)
    optimizer.update_params(l2)
    optimizer.update_params(l3)

    predictions = np.argmax(act3.output, axis=1)
    accuracy = np.mean(predictions == y_train)
    print(f'Epoch: {epoch}, Loss: {loss_fin}, Accuracy: {accuracy}')

print(f'Final accuracy: {accuracy}')
print(f'Final loss: {loss_fin}')
```

4. Test the Model:

```
correct = 0
for i in range(len(X_test)):
    l1.forward_prop(X_test[i])
    act1.forward(l1.output)
    l2.forward_prop(act1.output)
    act2.forward(l2.output)
    l3.forward_prop(act2.output)
    act3.forward(l3.output)

    prediction = np.argmax(act3.output)
    if prediction == y_test[i]:
        correct += 1

    print(f'Label: {y_test[i]}, Prediction: {prediction}')

accuracy = correct / len(X_test)
print(f'Test Accuracy: {accuracy}')
```
