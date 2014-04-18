/*
 * listnetloss.h
 *
 *  The loss function for listnet model, which is a cross entropy loss
 */

#ifndef LISTNETLOSS_H_
#define LISTNETLOSS_H_

#include <cmath>
#include <iostream>
#include <vector>
#include "neuralnet.h"

using std::vector;

class ListLoss {
public:
	ListLoss(BaseNet &input_net, vector<vector<double> > &feature_lists, vector<double> &results, bool simple);

	virtual double operator()();
	virtual void derivative();
	void set_trained_net(BaseNet &input_net);

protected:
	BaseNet trained_net_;
	vector<vector<double> > feature_lists_;
	vector<double> results_;

	//  vectors to store the delta weights between layers
	vector<vector<vector<double> > > input_hidden_d_list_;
	vector<vector<double> >  hidden_output_d_list_;

    // 2D vectors to store the delta weights for simple neural net
	vector<vector<double> > input_output_d_list_;

	// A function to calculate the output score from Neural network for each document
	void predict_score();

	// A function to calculate exp (actual score) for each document
	void prob_score();

	// calculate the sum of all predict score
	void pred_score_sum();

	// calculate the sum of all actual score
	void prob_score_sum();

	// calculate the partial sum of the predict score given the position
	double pred_partial_sum(int i );

	double prob_partial_sum(int i );

	// variable to store the sum of the exp(desired_score) or denominator of the top prob
	double prob_denorm_;

	// variable to store the sum of results of neural net model
	double pred_denorm_;

	// boolean variable to judge if it is a simple neural net (no hidden layer)
	bool if_simple_;

	// 1D vector to store the prob of ground truth for each document
	vector<double> prob_list_;

	// 1D vector to store the prob of prediction value for each document
	vector<double> pred_list_;
private:

};

#endif /* LISTNETLOSS_H_ */
