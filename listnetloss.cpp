/*
 * listnetloss.cpp
 *
 *  Created on: Apr 12, 2014
 *      Author: debian
 */

#include "listnetloss.h"

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

	for(int i = 0; i < results_.size(); ++i)
	{
	 prob_denorm_ += exp(results_[i]);
	}

	for(int i = 0; i < results_.size(); ++i)
	{
		prob_list_[i]  = exp(results_[i]) / prob_denorm_;
	}

	for(int i = 0; i < results_.size(); ++i)
	{

	}

}

virtual double ListNetLoss::operator()()
{

}

