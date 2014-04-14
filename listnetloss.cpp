/*
 * listnetloss.cpp
 *
 *  Created on: Apr 12, 2014
 *      Author: debian
 */

#include "listnetloss.h"

using std::vector;
using std::cout;
using std::endl;

ListNetLoss::ListNetLoss(NeuralNet &input_net, vector<vector<double> > &feature_lists, vector<double> &result, bool simple): trained_net_(input_net),
																												feature_lists_(feature_lists),
	                                                                                                             results_(result),
	                                                                                                             if_simple_(simple)
{
	// check if it is a simple neural net and set up the vectors to store the weight changes
	if(if_simple_)
	{
		input_output_d_list_.resize(result.size());
	}
	else{
		input_hidden_d_list_.resize(result.size());
		hidden_output_d_list_.resize(result.size());
	}

	// change the size of the prob list and pred list
	prob_list_.resize(results_.size());
	pred_list_.resize(results_.size());

	for(int i = 0; i < results_.size(); ++i)
	{
		prob_denorm_ += exp(results_[i]);
	}

	// store the top probability for each document given the score
	for(int i = 0; i < results_.size(); ++i)
	{
		prob_list_[i]  = exp(results_[i]) / prob_denorm_;
	}
}

// Calculation of cross entropy loss function
virtual double ListNetLoss::operator()()
{
	double results = 0;

	for(int i = 0; i < results_.size(); ++i)
	{
		results += pred_list_[i] * prob_list_[i];
	}

	return results;
}

virtual void ListNetLoss::derivative()
{




}
// Remember to call predict_score after instantiate the class
// it will save the top probability of each document into a list
virtual void ListNetLoss::predict_score()
{
	for(int i = 0; i < results_.size(); ++i)
	{
		pred_list_[i] = trained_net_.feed_forward(feature_lists_[i]);
	}

	for(int i = 0; i < pred_list_.size(); ++i)

	{
		pred_denorm_ += pred_list_[i];
	}

	for(int i = 0; i < pred_list_.size(); ++i)
	{
		pred_list_[i] = pred_list_[i] / pred_denorm_;
	}
}
