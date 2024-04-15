"""
This is a basic neural network taking in data and using the data
to determine if borrower's probability of repaying the loan

"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import traceback

def sigmoid(sum):
    return 1 / (1 + np.exp(-sum))

def sigmoid_derivative(sigmoid):
    return sigmoid * (1 - sigmoid)

def calculate_output(instance, weights0, weights1):
    hidden_layer = sigmoid(np.dot(instance, weights0))
    output_layer = sigmoid(np.dot(hidden_layer, weights1))
    return output_layer[0]

def prepare_data(filepath):
    dataset = pd.read_csv(filepath)

    # To handle blanked values (the dataset has 3 missing age values.)
    dataset = dataset.dropna()

    # Extract and scale the input features (assuming columns are 'income', 'age', 'loan')
    inputs = dataset.iloc[:, 1:4].values  # Adjust indices as necessary
    scaler = MinMaxScaler()
    inputs_scaled = scaler.fit_transform(inputs)

    outputs = dataset.iloc[:, 4].values
    outputs = outputs.reshape(-1, 1)
    
    return inputs_scaled, outputs

def main():
    try:
        inputs, outputs = prepare_data('./data/credit_data.csv')

        # NOTE because of the random below the same data and code can give
        # a range of results.
        weights0 = 2 * np.random.random((3, 13)) - 1
        weights1 = 2 * np.random.random((13, 1)) - 1

        epochs = 30000
        learning_rate = 0.02

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
                print('Epoch: ' + str(epoch) + ' Error: ' + str(average))
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
        The following plot is for viewing the training based mainly on 
        epochs and learning rate. The training rate can also be influnced 
        by the number of neurons within the weights (where weights 0 and 1 are set)

        Use the chart to fine tune the three values to get the most efficient learning curve
        """
        """ runGraph = input("Run learning graph? y: ")
        if runGraph.capitalize() == 'Y':
            import matplotlib.pyplot as plt
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.plot(error)
            plt.show() """

        results = []
        for idx, input_instance in enumerate(inputs):
            result = round(calculate_output(input_instance, weights0, weights1))
            results.append([result, outputs[idx][0]])
        
        print("--------------------------")
        print(f"Good records: {len(results)}")
        incorrect_count = sum(1 for result in results if result[0] != result[1])
        print(f"Incorrect forecasts: {incorrect_count}")
    
    # Basic errorhandling mainly for example. As is, this really only does what
    # running the code in a IDE would do.
    except Exception as e:
        print(traceback.format_exc())
        
if __name__ == '__main__':
    main()