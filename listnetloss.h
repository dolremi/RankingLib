/*
 * listnetloss.h
 *
 *  The loss function for listnet model, which is a cross entropy loss
 */

#ifndef LISTNETLOSS_H_
#define LISTNETLOSS_H_

#include <iostream>
#include <vector>
#include "neuralnet.h"

using std::vector;

class ListNetLoss {
public:
	ListNetLoss(NeuralNet &input_net, vector<vector<double> > &feature_lists, vector<double> &results);

	virtual double operator()();
	virtual double derivative();

protected:
	NeuralNet trained_net_;
	vector<vector<double> > feature_lists_;
	vector<double> results;

	// 3D vectors to store the delta weights between layers
	vector<vector<vector<double> > > input_hidden_d_list_;
	vector<vector<vector<double> > > hidden_output_d_list_;
	double top_prob();

};

#endif /* LISTNETLOSS_H_ */
