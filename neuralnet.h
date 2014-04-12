/*
 * neuralnet.h
 * The main class for neural network model
 *
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "transfunction.h"

class NeuralNet {
	using std::vector;
	using std::string;
public:
	NeuralNet(int input_n, int hidden_n, int output_n, TransFunction &tf);
    ~NeuralNet(){};

    // getter functions for input, hidden and output layer size
    inline int get_input_num();
    inline int get_hidden_num();
    inline int get_output_num();

    // setter functions for input, hidden and output layer size
    inline void set_input_num(int input_num);
    inline void set_hidden_num(int hidden_num);
    inline void set_output_num(int output_num);

    // setter function for input layer
    inline void set_input_layer(const vector<double> &input_layer);

    // getter function for output layer
    inline vector<double> get_output_layer();

    // getter function for weights update
    vector<vector<double> > get_input_hidden_d();
    vector<vector<double> > get_hidden_output_d();

    // load or save weights in a file
    bool load_weights(const string &input_filename);
    bool save_weights(const string &output_filename);

    // feed forward propagate from input layer to output layer
    void feed_forward(const vector<double> &input_layer);

    // back propagate from output layer to input layer by calculating the error gradient
    void back_prop();

    // calculate the basic error gradient from layers to layers backward
    void error_grad_output_hidden();
    void error_grad_hidden_input();

    void update_weights(double learn_rate);
    void update_weights(vector<vector<double> > &delta_in_hid, vector<vector<double> > &delta_hid_in, double learn_rate);

private:

	// number of neurons in input, hidden, output layers
    int input_num_;
    int hidden_num_;
    int output_num_;

    // activation function transfer the input value to the output value
    TransFunction active_func_;

    // 1D vectors to store the input, hidden and output layers
    vector<double> input_layer_;
    vector<double> hidden_layer_;
    vector<double> output_layer_;

    // 1D vector to store the delta for output neurons and hidden neurons
    vector<double> delta_output_;
    vector<double> delta_hidden_;

    // 2D vectors to store the weights between input layers to hidden and hidden to output layers
    vector<vector<double> > input_hidden_w_;
    vector<vector<double> > hidden_output_w_;

    // 2D vectors to store the change in the weights
    vector<vector<double> > input_hidden_d_;
    vector<vector<double> > hidden_output_d_;

    // utility functions to initialize the weights
    void initialize_weights();
};

#endif /* NEURALNET_H_ */
