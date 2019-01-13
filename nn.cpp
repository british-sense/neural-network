#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>
#include "algebra.hpp"

std::mt19937 mt(0);  

std::vector<double> sigmoid(std::vector<double> x){
    std::vector<double> out(x.size());
    for(int i = 0; i < x.size(); i++) out[i] = 1 / (1 + exp(-x[i]));
    return out;
}

std::vector<double> relu(std::vector<double> x){
    std::vector<double> out(x.size());
    for(int i = 0; i < x.size(); i++) out[i] = x[i] > 0 ? x[i] : 0;
    return out;
}

void input_file(const std::string &filename, std::vector<double> &label, std::vector<std::vector<double> > &data){

    std::ifstream ifile(filename);
    if (ifile.fail()){
        std::cerr << "cannot open input file" << std::endl;
        return;
    }

    std::string str;
    for(int i = 0; getline(ifile, str); i++){
        std::istringstream stream(str);
        std::string tmp;
        getline(stream, tmp, ',');
        label.push_back(stoi(tmp));
        std::vector<double> tmp_data;
        while(getline(stream, tmp, ',')){
            tmp_data.push_back(stoi(tmp));
        }
        data.push_back(tmp_data);
    }
}

class neural_network{

    public:
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    std::vector<std::vector<double> > w_ih;
    std::vector<std::vector<double> > w_ho;
    double learning_rate;
    std::function<std::vector<double>(std::vector<double>)> activation_function;

    // initialization neural network
    void init(int input, int hidden, int output, double lr){

        input_nodes = input;
        hidden_nodes = hidden;
        output_nodes = output;

        w_ih = std::vector<std::vector<double> >(hidden, std::vector<double>(input));
        w_ho = std::vector<std::vector<double> >(output, std::vector<double>(hidden));
        std::normal_distribution<> ih_rand(0.0, std::pow(hidden, -0.5));
        for(int i = 0; i < input; i++) for(int j = 0; j < hidden; j++) w_ih[j][i] = ih_rand(mt);
        std::normal_distribution<> ho_rand(0.0, std::pow(output, -0.5));
        for(int i = 0; i < hidden; i++) for(int j = 0; j < output; j++) w_ho[j][i] = ho_rand(mt);

        learning_rate = lr;

        activation_function = sigmoid;
    }

    // traning function
    void train(std::vector<double> input_data, std::vector<double> target_data){

        std::vector<double> hidden_input = w_ih * input_data;
        std::vector<double> hidden_output =  activation_function(hidden_input);        

        std::vector<double> final_input = w_ho * hidden_output;
        std::vector<double> final_output = activation_function(final_input);

        std::vector<double> output_error = target_data - final_output;
        std::vector<double> hidden_error = trans(w_ho) * output_error;

        w_ho += learning_rate * ((output_error * final_output * (1. - final_output)) * hidden_output);
        w_ih += learning_rate * ((hidden_error * hidden_output * (1. - hidden_output)) * input_data);

    }

    std::vector<double> test(std::vector<double> input_data){

        std::vector<double> hidden_input = w_ih * input_data;
        std::vector<double> hidden_output =  activation_function(hidden_input);        

        std::vector<double> final_input = w_ho * hidden_output;
        std::vector<double> final_output = activation_function(final_input);

        return final_output;
    }
};

int main(){

    // neural network initialization
    int input_nodes = 784;
    int hidden_nodes = 100;
    int output_nodes = 10;
    double learning_rate = 0.2;
    neural_network net;
    net.init(input_nodes, hidden_nodes, output_nodes, learning_rate);

    // load training data
    std::string trainfile = "mnist_train.csv";
    std::vector<double> label;
    std::vector<std::vector<double> > data;
    input_file(trainfile, label, data);

    // training
    for(int i = 0; i < label.size(); i++){
        std::vector<double> input = ((data[i] / 255.) * 0.99) + 0.01;
        std::vector<double> target(output_nodes, 0.01);
        target[label[i]] = 0.99;
        net.train(input, target); 
    }

    // load test data
    std::string testfile = "mnist_test.csv";
    std::vector<double> answer_label;
    std::vector<std::vector<double> > test_data;
    input_file(testfile, answer_label, test_data);

    // test
    double score = 0.;
    for(int i = 0; i < answer_label.size(); i++){
        std::vector<double> input = ((test_data[i] / 255.) * 0.99) + 0.01;
        std::vector<double> output = net.test(input);
        int max_index = std::distance(output.begin(), max_element(output.begin(), output.end()));
        std::cout << "answer label : " << answer_label[i] << ", network answer : " << max_index << std::endl;
        if(max_index == answer_label[i]) score += 1.;
    }

    std::cout << "correct answer rate : " << score / answer_label.size() << std::endl;

    return 0;
}
