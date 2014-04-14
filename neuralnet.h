/*
 * neuralnet.h
 *
 *  NeuralNet class is derived class of BaseNet
 *  It has one hidden layer
 */

#ifndef NEURALNET_H_
#define NEURALNET_H_

#include <iostream>
#include <fstream>
#include "basenet.h"

using std::vector;
class NeuralNet: public BaseNet {
public:
	NeuralNet(int input_size, int hidden_size, int output_size, TransFunction &tf);

	virtual void initialize_weights();

	// load and save weight parameters via files
	virtual bool load_weights(const string &input_filename);
	virtual bool save_weights(const string &output_filename);

	// getter methods for weights change input to hidden and hidden to output layers
	inline vector<vector<double> > get_input_hidden_delta();
	inline vector<vector<double> > get_hidden_output_delta();

	inline void set_input_hidden_delta(vector<vector<double> > &in_hid_d);
	inline void set_hidden_output_delta(vector<vector<double> > &hid_out_d);

	// feed forward from input layer to hidden layer then to output layer
	virtual void feed_forward(const vector<double> &input_var);

	// back propagation algorithm
	virtual void back_prop();

	// update the weights given the learning rate
	virtual void update_weights(double learn_rate);
private:
    // 1D vector to store the hidden layer
	vector<double> hidden_layer_;

	// 1D vector to store the update for hidden neurons
	vector<double> delta_hidden_;

	// variables to store the weights between input layers to hidden and hidden to output layers
	vector<vector<double> > input_hidden_w_;
	vector<double>  hidden_output_w_;

	// variables to store the change in the weights
	vector<vector<double> > input_hidden_d_;
	vector<double>  hidden_output_d_;

	// calculate the basic error gradient from layers to layers backward
	void error_grad_output_hidden();
	void error_grad_hidden_input();
};

#endif /* NEURALNET_H_ */
