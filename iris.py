from sklearn import datasets

def calculate_output(instance):
  hidden_layer = sigmoid(np.dot(instance, weights0))
  output_layer = sigmoid(np.dot(hidden_layer, weights1))
  return output_layer[0]

#This loads the predefined iris flowers data based within sklearn.datasets
#The data is sets of 4 values - sepal length & width, petal length & width
iris = datasets.load_iris()

inputs = iris.data[0:100]
outputs = iris.target[0:100]
outputs = outputs.reshape(-1,1)

import numpy as np

def sigmoid(sum):
  return 1 / (1 + np.exp(-sum))

def sigmoid_derivative(sigmoid):
  return sigmoid * (1 - sigmoid)

#The second number within the random is the number of neurons
weights0 = 2 * np.random.random((4, 5)) - 1
#The first number within the random below matches above for neurons
weights1 = 2 * np.random.random((5,1)) - 1

epochs = 3000
learning_rate = 0.01

error = []

for epoch in range(epochs):
  input_layer = inputs
  sum_synapse0 = np.dot(input_layer, weights0)
  hidden_layer = sigmoid(sum_synapse0)

  sum_synapse1 = np.dot(hidden_layer, weights1)
  output_layer = sigmoid(sum_synapse1)

  error_output_layer = outputs - output_layer
  average = np.mean(abs(error_output_layer))
  
  if epoch % 1000 == 0:
    print('Epoch: ' + str(epoch + 1) + ' Error: ' + str(average))
    error.append(average)
  
  derivative_output = sigmoid_derivative(output_layer)
  delta_output = error_output_layer * derivative_output
  
  weights1T = weights1.T
  delta_output_weight = delta_output.dot(weights1T)
  delta_hidden_layer = delta_output_weight * sigmoid_derivative(hidden_layer)
  
  hidden_layerT = hidden_layer.T
  input_x_delta1 = hidden_layerT.dot(delta_output)
  weights1 = weights1 + (input_x_delta1 * learning_rate)
  
  input_layerT = input_layer.T
  input_x_delta0 = input_layerT.dot(delta_hidden_layer)
  weights0 = weights0 + (input_x_delta0 * learning_rate)
"""  
The following plot is for viewing the training based mainly on epochs and
learning rate. The training rate can also be influnced by the number of 
neurons within the weights (where weights 0 and 1 are set)

Use the chart to fine tune the three values to get the most efficient learning curve
"""
"""
runGraph = input("Run learning graph? y: ")
if runGraph.capitalize() == 'Y':
    import matplotlib.pyplot as plt
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.plot(error)
    plt.show()
"""

results = []
for idx,input in enumerate(inputs):
    results.append([iris.target_names[int(round(calculate_output(input)))], iris.target_names[int(outputs[idx][0])]])
print(len(results))
i = 0
for idx,r in enumerate(results):
    if r[0] != r[1]:
        i += 1
        print(f"row: {idx}, values:{r}")

print(i)
