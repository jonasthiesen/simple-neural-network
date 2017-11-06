import numpy as np
import scipy.misc

from NeuralNetwork import NeuralNetwork

network = NeuralNetwork(784, 1, 200, 10, 0.1)

data_file = open("data/mnist_train.csv", 'r')
data_list_train = data_file.readlines()
data_file.close()

epochs = 1
for e in range(epochs):
    for record in data_list_train:
        input_data = record.split(',')
        scaled_input_data = (np.asfarray(input_data[1:]) / 255.0 * 0.99) + 0.01

        target_output = np.zeros(10) + 0.01
        target_output[int(input_data[0])] = 0.99

        network.train(scaled_input_data, target_output)

data_file = open("data/mnist_test.csv", 'r')
data_list_test = data_file.readlines()
data_file.close()

accuracy = []

for record in data_list_test:
    input_data = record.split(',')
    scaled_input_data = (np.asfarray(input_data[1:]) / 255.0 * 0.99) + 0.01
    
    target_output = np.zeros(10) + 0.01
    target_output[int(input_data[0])] = 0.99
    
    guess = np.argmax(network.predict(scaled_input_data))
    if (guess == int(input_data[0])):
        accuracy.append(1)
    else:
        accuracy.append(0)
    
    #print(guess, "=", input_data[0])

print(np.average(accuracy)*100, "%")

def prepare_image(src):
    load_image = scipy.misc.imread(src, flatten = True)
    prepare_image = 255.0 - load_image.reshape(784)
    scale_image = (prepare_image / 255.0 * 0.99) + 0.01
    
    return scale_image

image_src = "data/number_3.png"
print(np.argmax(network.predict(prepare_image(image_src))))

image_src_2 = "data/number_7.png"
print(np.argmax(network.predict(prepare_image(image_src_2))))