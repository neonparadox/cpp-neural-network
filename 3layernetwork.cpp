#include <bits/stdc++.h>
#include <cstdlib>
#include <cmath>
using namespace std;
const int input_size = 25; // 40x40 pixels
const int hidden_layer1_size = 40;
const int hidden_layer2_size = 40;
const int hidden_layer3_size = 40;
const int num_classes = 2;
mt19937_64 rng(chrono::steady_clock::now().time_since_epoch().count());
// Activation function
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}

// Training sample
struct TrainingSample
{
    std::vector<double> input;
    std::vector<double> target;

    TrainingSample(const std::vector<double> &input, const std::vector<double> &target)
        : input(input), target(target)
    {
    }
};

// Neural network
class NeuralNetwork
{
private:
    std::vector<std::vector<double>> weights1;
    std::vector<double> biases1;
    std::vector<std::vector<double>> weights2;
    std::vector<double> biases2;
    std::vector<std::vector<double>> weights3;
    std::vector<double> biases3;
    std::vector<std::vector<double>> weights4;
    std::vector<double> biases4;

public:
    std::vector<double> feedforward(const std::vector<double> &input)
    {
        std::vector<double> hidden1_output(hidden_layer1_size);
        for (int i = 0; i < hidden_layer1_size; i++)
        {
            double sum = biases1[i];
            for (int j = 0; j < input_size; j++)
            {
                sum += weights1[j][i] * input[j];
            }
            hidden1_output[i] = sigmoid(sum);
        }

        std::vector<double> hidden2_output(hidden_layer2_size);
        for (int i = 0; i < hidden_layer2_size; i++)
        {
            double sum = biases2[i];
            for (int j = 0; j < hidden_layer1_size; j++)
            {
                sum += weights2[j][i] * hidden1_output[j];
            }
            hidden2_output[i] = sigmoid(sum);
        }
		std::vector<double> hidden3_output(hidden_layer3_size);
        for (int i = 0; i < hidden_layer3_size; i++)
        {
            double sum = biases3[i];
            for (int j = 0; j < hidden_layer2_size; j++)
            {
                sum += weights3[j][i] * hidden2_output[j];
            }
            hidden3_output[i] = sigmoid(sum);
        }
        std::vector<double> output(num_classes);
        for (int i = 0; i < num_classes; i++)
        {
            double sum = biases4[i];
            for (int j = 0; j < hidden_layer3_size; j++)
            {
                sum += weights4[j][i] * hidden3_output[j];
            }
            output[i] = sigmoid(sum);
        }

        return output;
    }
    // Backpropagation function
    void backpropagation(const std::vector<double> &input, const std::vector<double> &target, double learning_rate)
    {
        // Feedforward
        std::vector<double> hidden1_output(hidden_layer1_size);
        for (int i = 0; i < hidden_layer1_size; i++)
        {
            double sum = biases1[i];
            for (int j = 0; j < input_size; j++)
            {
                sum += weights1[j][i] * input[j];
            }
            hidden1_output[i] = sigmoid(sum);
        }

        std::vector<double> hidden2_output(hidden_layer2_size);
        for (int i = 0; i < hidden_layer2_size; i++)
        {
            double sum = biases2[i];
            for (int j = 0; j < hidden_layer1_size; j++)
            {
                sum += weights2[j][i] * hidden1_output[j];
            }
            hidden2_output[i] = sigmoid(sum);
        }

        std::vector<double> hidden3_output(hidden_layer3_size);
        for (int i = 0; i < hidden_layer3_size; i++)
        {
            double sum = biases3[i];
            for (int j = 0; j < hidden_layer2_size; j++)
            {
                sum += weights3[j][i] * hidden2_output[j];
            }
            hidden3_output[i] = sigmoid(sum);
        }
        std::vector<double> output(num_classes);
        for (int i = 0; i < num_classes; i++)
        {
            double sum = biases4[i];
            for (int j = 0; j < hidden_layer3_size; j++)
            {
                sum += weights4[j][i] * hidden3_output[j];
            }
            output[i] = sigmoid(sum);
        }
        
    }
    void train(const std::vector<TrainingSample> &samples, int num_epochs, double learning_rate)
    {
        for (int epoch = 0; epoch < num_epochs; epoch++)
        {
			double losssum=0.0;
			int samplesize=samples.size();
			cout<<"Epoch: " << epoch << "\n";
            for (const TrainingSample &sample : samples)
            {	
				/*
                // Print the input sample
                std::cout << "Input sample: ";
                for (int i = 0; i < input_size; i++)
                {
                    std::cout << sample.input[i] << " ";
                }
                std::cout << std::endl;*/
                // Feedforward
                std::vector<double> hidden1_output = feedforward(sample.input);
                std::vector<double> hidden2_output = feedforward(hidden1_output);
                std::vector<double> hidden3_output = feedforward(hidden2_output);
                std::vector<double> output = feedforward(hidden3_output);
                // Print the output of the network
                /*std::cout << "Output: ";
                for (int i = 0; i < num_classes; i++)
                {
                    std::cout << output[i] << " ";
                }
                std::cout << std::endl;

                //print intended target and loss
                std::cout << "Target: ";
                for (int i = 0; i < num_classes; i++)
                {
                    std::cout << sample.target[i] << " ";
                }
               
                std::cout << std::endl;
                std::cout << "Loss: ";*/
                
                for (int i = 0; i < num_classes; i++)
                {
                    //std::cout << sample.target[i] - output[i] << " ";
                    
                    losssum+=abs(sample.target[i]-output[i]);
                }
                //std::cout << std::endl;
                

                // Add a delay
                //std::this_thread::sleep_for(std::chrono::milliseconds(5));
                // Backpropagation
                std::vector<double> output_delta(num_classes);
                for (int i = 0; i < num_classes; i++)
                {
                    output_delta[i] = (sample.target[i] - output[i]) * sigmoid_derivative(output[i]);
                }
				std::vector<double> hidden3_delta(hidden_layer3_size);
                for (int i = 0; i < hidden_layer3_size; i++)
                {
                    double error = 0.0;
                    for (int j = 0; j < num_classes; j++)
                    {
                        error += output_delta[j] * weights4[i][j];
                    }
                    hidden3_delta[i] = error * sigmoid_derivative(hidden3_output[i]);
                }
                std::vector<double> hidden2_delta(hidden_layer2_size);
                for (int i = 0; i < hidden_layer2_size; i++)
                {
                    double error = 0.0;
                    for (int j = 0; j < hidden_layer3_size; j++)
                    {
                        error += output_delta[j] * weights3[i][j];
                    }
                    hidden2_delta[i] = error * sigmoid_derivative(hidden2_output[i]);
                }

                std::vector<double> hidden1_delta(hidden_layer1_size);
                for (int i = 0; i < hidden_layer1_size; i++)
                {
                    double error = 0.0;
                    for (int j = 0; j < hidden_layer2_size; j++)
                    {
                        error += hidden2_delta[j] * weights2[i][j];
                    }
                    hidden1_delta[i] = error * sigmoid_derivative(hidden1_output[i]);
                }
                // Update weights and biases
                for (int i = 0; i < hidden_layer1_size; i++)
                {
                    biases1[i] += learning_rate * hidden1_delta[i];
                    for (int j = 0; j < input_size; j++)
                    {
                        weights1[j][i] += learning_rate * hidden1_delta[i] * sample.input[j];
                    }
                }

                for (int i = 0; i < hidden_layer2_size; i++)
                {
                    biases2[i] += learning_rate * hidden2_delta[i];
                    for (int j = 0; j < hidden_layer1_size; j++)
                    {
                        weights2[j][i] += learning_rate * hidden2_delta[i] * hidden1_output[j];
                    }
                }
                for (int i = 0; i < hidden_layer3_size; i++)
                {
                    biases3[i] += learning_rate * hidden3_delta[i];
                    for (int j = 0; j < hidden_layer2_size; j++)
                    {
                        weights3[j][i] += learning_rate * hidden3_delta[i] * hidden2_output[j];
                    }
                }
				 
                for (int i = 0; i < num_classes; i++)
                {
                    biases4[i] += learning_rate * output_delta[i];
                    for (int j = 0; j < hidden_layer3_size; j++)
                    {
                        weights4[j][i] += learning_rate * output_delta[i] * hidden3_output[j];
                    }
                }
            }
            cout<<"Average absolute loss for epoch " << epoch << ": " << losssum/(double)samplesize<<"\n";
        }
    }
    NeuralNetwork(int input_size, int hidden_layer1_size, int hidden_layer2_size, int hidden_layer3_size,int num_classes)
    {
        // Initialize weights and biases with random values
        weights1.resize(input_size);
        biases1.resize(hidden_layer1_size);
        for (int i = 0; i < input_size; i++)
        {
            weights1[i].resize(hidden_layer1_size);
            for (int j = 0; j < hidden_layer1_size; j++)
            {
                weights1[i][j] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
            }
        }
        for (int i = 0; i < hidden_layer1_size; i++)
        {
            biases1[i] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
        }

        weights2.resize(hidden_layer1_size);
        biases2.resize(hidden_layer2_size);
        for (int i = 0; i < hidden_layer1_size; i++)
        {
            weights2[i].resize(hidden_layer2_size);
            for (int j = 0; j < hidden_layer2_size; j++)
            {
                weights2[i][j] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
            }
        }
        for (int i = 0; i < hidden_layer2_size; i++)
        {
            biases2[i] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
        }

        weights3.resize(hidden_layer2_size);
        
        biases3.resize(hidden_layer3_size);
        for (int i = 0; i < hidden_layer2_size; i++)
        {
            weights3[i].resize(hidden_layer3_size);
            for (int j = 0; j < hidden_layer3_size; j++)
            {
                weights3[i][j] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
            }
        }
        for (int i = 0; i < hidden_layer3_size; i++)
        {
            biases3[i] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
        }

        weights4.resize(hidden_layer3_size);
        
        
        biases4.resize(num_classes);
        for (int i = 0; i < hidden_layer3_size; i++)
        {
            weights4[i].resize(num_classes);
            for (int j = 0; j < num_classes; j++)
            {
                weights4[i][j] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
            }
        }
        for (int i = 0; i < num_classes; i++)
        {
            biases4[i] = 2.0 * ((double)(rng()%10000000)/(double)10000000) - 1.0;
        }
        std::vector<double> feedforward(const std::vector<double> &input);
        // Backpropagation function

        void backpropagation(const std::vector<double> &input, const std::vector<double> &target, double learning_rate);

        void train(const std::vector<TrainingSample> &samples, int num_epochs, double learning_rate);
    }
};
int main()
{
    // Initialize the neural network
    NeuralNetwork neural_network(input_size, hidden_layer1_size, hidden_layer2_size,hidden_layer3_size, num_classes);
    // Load the training samples
    std::vector<TrainingSample> training_samples;
    for(int i = 0; i <600 ;i++){
        double ver = 0;
        double hor = 0;
        //insert custom task here, this task is a weird form of detecting horizontal/vertical lines
        std::vector<double> grid;
        for(int j = 0; j < 25; j++){
            grid.push_back((double)((int)rng() % 2));

        }
        for(int j = 0; j < 5; j++){
            int sum = 0;
            for(int k = 0; k < 5; k++){
                sum += grid[j*5 + k];
            }
            if(sum >= 5){
                ver = 1;
            }
        }
        for(int j = 0; j < 5; j++){
            int sum = 0;
            for(int k = 0; k < 5; k++){
                sum += grid[k*5 + j];
            }
            if(sum >= 5){
                hor = 1;
            }
        }
        //store the grid and the ver and hor values in a training sample
        vector<double> target(2);target[0]=ver;target[1]=hor;
        
        TrainingSample sample(grid,target);
        training_samples.push_back(sample);
    }

    // ... load the training samples here ...

    // Train the neural network
    neural_network.train(training_samples, 3000, 0.04);

    return 0;
}
