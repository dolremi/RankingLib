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
		return epoch_ <= max_epochs_;
	}
	else
	{
		current_error_ = ListLoss();
		return current_error_ <= epsilon_;
	}

}
