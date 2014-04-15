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

}

// Calculation of cross entropy loss function
virtual double ListNetLoss::operator()()
{

	// calculate exp value for prediction score
	predict_score();
	pred_score_sum();

	// calculate exp value for actual score
	prob_score();
	prob_score_sum();


	double results = 0;

	for(int i = 0; i < results_.size(); ++i)
	{
		results += pred_list_[i] * prob_list_[i]/ (pred_denorm_ * prob_denorm_);
	}

	return results;
}

virtual void ListNetLoss::derivative()
{

	// calculate exp value for prediction score
	predict_score();
	pred_score_sum();

	// calculate exp value for actual score
	prob_score();
	prob_score_sum();

  // check if it is a simple net
  if(if_simple_)
  {
	  vector<double> delta_w(trained_net_.get_input_num() + 1,0);

	  input_output_d_list_.resize(results_.size());

	  // save the input output weight change for each document
	  for(int i = 0; i < input_output_d_list_.size(); ++i)
	  {
		 trained_net_.feed_forward(feature_lists_[i]);
		 trained_net_.back_prop();
		 input_output_d_list_[i] = trained_net_.get_input_output_delta();
	  }

	  // now calculate the input to output weight changes
	  for(int i = 0; i < results_.size(); ++i)
	  {
		  for(int j = 0; j < delta_w.size(); ++j)
		  {
			  delta_w[j] += (pred_partial_sum(i) * pred_list_[i]  - prob_list_[i] / prob_denorm_) * input_output_d_list_[i][j];
		  }
	  }

	  // set up the weight change for neural network
	  trained_net_.set_input_output_delta(delta_w);
  }
  else
  {
    // now with more complex neural network

	  //  vectors to store the weight changes from input to hidden layer and hidden layer to output layer
     vector<vector<double> > delta_input_hidden(trained_net_.get_input_num() + 1);
     vector<double>  delta_hidden_output(trained_net_.get_hidden_num() + 1,0);

     input_hidden_d_list_.resize(results_.size());
     hidden_output_d_list_.resize(results_.size());

     // save the input to hidden and hidden to output weight changes for each document
     for(int i = 0; i < input_output_d_list_.size(); ++i)
     {
     	trained_net_.feed_forward(feature_lists_[i]);
     	trained_net_.back_prop();
     	input_hidden_d_list_[i] = trained_net_.get_input_hidden_delta();
     	hidden_output_d_list_[i] = trained_net_.get_hidden_output_delta();
     }

     // now calculate the hidden to output weight changes
     for(int i = 0; i < results_.size(); ++i)
     {
    	for(int j = 0; j < delta_hidden_output.size(); ++j)
    		  {
    			  delta_hidden_output[j] += (pred_partial_sum(i) * pred_list_[i]  - prob_list_[i] / prob_denorm_) * hidden_output_d_list_[i][j];
    		  }
    	  }

     trained_net_.set_hidden_output_delta(delta_hidden_output);

     for(int i = 0; i < results_.size(); ++i)
     {
    	 for(int j = 0; j < delta_input_hidden.size(); ++j)
    	 {
    		 delta_input_hidden[i].resize(trained_net_.get_hidden_num() + 1, 0);
    		 for(int k = 0; k <= trained_net_.get_hidden_num(); ++k)
    		 {
    			 delta_input_hidden[i][k] += (pred_partial_sum(i) * pred_list_[i] - prob_list_[i]/ prob_denorm_) * input_hidden_d_list_[i][j][k];
    		 }
    	 }
     }

     trained_net_.set_input_hidden_delta(delta_input_hidden);
  }
}

// it will save the predict score(exp) of each document into a list
void ListNetLoss::predict_score()
{
	for(int i = 0; i < results_.size(); ++i)
	{
		pred_list_[i] = trained_net_.feed_forward(feature_lists_[i]);
		pred_list_[i] = exp(pred_list_[i]);
	}
}

// it will save the actual score(exp) of each document into a list
void ListNetLoss::prob_score()
{
	for(int i = 0; i < results_.size(); ++i)
	{
		prob_list_[i] = exp(results_[i]);
	}
}

void ListNetLoss::prob_score_sum()
{
	for(int i = 0; i < prob_list_.size(); ++i)
	{
		prob_denorm_ += prob_list_[i];
	}
}

void ListNetLoss::pred_score_sum()
{
	for(int i = 0; i < pred_list_.size(); ++i)

	{
		pred_denorm_ += pred_list_[i];
	}
}

// assume the pred_list has been created
double ListNetLoss::pred_partial_sum(int index)
{
	double sum = 0;

	for(int i = index; i < pred_list_.size(); ++i)
	{
		sum += pred_list_[i];
	}
	return sum;
}

// assume the prob_list has been created
double ListNetLoss::prob_partial_sum(int index)
{
	double sum = 0;

	for(int i = index; i < prob_list_.size(); ++i)
	{
		sum += prob_list_[i];
	}

	return sum;
}
