/*
 * basenet.h
 * This is a simple version of neural network
 * There is no hidden layer for this simple net
 */

#ifndef BASENET_H_
#define BASENET_H_

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "transfunction.h"

class BaseNet {
	using std::vector;
	using std::string;
public:
	BaseNet(int input_size, TransFunction &tf);

	// utility functions to initialize the weights
	virtual void initialize_weights();

    // getter functions for input, hidden and output layer size
    inline int get_input_num();
    inline int get_hidden_num();
    inline int get_output_num();

    // setter functions for input, hidden layer size, the output layer size is fixed to 1 for ranking problem
    inline void set_input_num(int input_num);
    inline void set_hidden_num(int hidden_num);
    inline void set_output_num(int output_num);

    // setter function for input layer
    inline void set_input_layer(const vector<double> &input_layer);

    // getter function for output layer
    inline double get_output_result();

    // getter function for weights update
    inline vector<double> get_input_output_delta();

    // setter function for weights update
    inline void set_input_output_delta(vector<double> &in_out_d);
    // load or save weights in a file
    virtual bool load_weights(const string &input_filename);
    virtual bool save_weights(const string &output_filename);

    // feed forward propagate from input layer to output layer
    virtual void feed_forward(const vector<double> &input_layer);

    // back propagate from output layer to input layer by calculating the error gradient
    virtual void back_prop();

    virtual void update_weights(double learn_rate);
    virtual void update_weights(vector<vector<double> > &delta_in_hid, vector<double>  &delta_hid_in, double learn_rate);

protected:
    // number of neurons in input, hidden and output layers
    int input_num_;
    int output_num_;
    int hidden_num_;

    // activation function transfer the input value to the output value
    TransFunction active_func_;

    // variables to store the input, hidden and output layers
    vector<double> input_layer_;
    double output_layer_;

    // variable to store update for output neuron
    double delta_output_;
private:
    // 1D vector to store the weight from input layer to output layer
    vector<double>  input_output_w_;

    // 1D vector to store the weight change from input layer to output layer
    vector<double>  input_output_d_;

    // calculate the error gradient from output to input layer
    void error_grad_output_input();
};

#endif /* NEURALNET_H_ */
