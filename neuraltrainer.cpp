/*
 * neuraltrainer.cpp
 *
 *  The trainer class to train the data set.
 *
 */

#include "neuraltrainer.h"

NeuralTrainer::NeuralTrainer(BaseNet &input_net, ListLoss &input_loss, bool based_iter, double lr = 0.01, double acc = 0.01, long max_iter = 500): trained_model_(input_net),
																						loss_func_(input_loss),
																						learn_rate_(lr),
																						current_error_(0),
																						epsilon_(acc),
																						epoch_(0),
																						max_epochs_(max_iter),
																						iter_based_(based_iter)
																						{
	// TODO Auto-generated constructor stub

}

bool NeuralTrainer::get_stop_condition()
{
	if(iter_based_)
	{
		return epoch_++ <= max_epochs_;
	}
	else
	{
		current_error_ = ListLoss();
		return current_error_ <= epsilon_;
	}

}

void NeuralTrainer::set_stop_parameters(bool base_iter, double acc, long max_iter)
{
	iter_based_ = base_iter;
	epsilon_ = acc;
	max_epochs_ = max_iter;
}

void NeuralTrainer::set_trained_model(BaseNet &input_model)
{
	trained_model_ = input_model;
}

BaseNet NeuralTrainer::get_trained_model()
{
	return trained_model_;
}

BaseNet NeuralTrainer::train_data(vector<vector<double> > &feature_list, vector<double> &score)
{
	while(get_stop_condition())
	{
		for(int i = 0; i < score.size(); ++i)
		{

		}
	}

	return get_trained_model();
}
