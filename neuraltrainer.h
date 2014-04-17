/*
 * neuraltrainer.h
 *
 *  Created on: Apr 16, 2014
 *      Author: debian
 */

#ifndef NEURALTRAINER_H_
#define NEURALTRAINER_H_

#include "basenet.h"
#include  "listnetloss.h"

#include <string>
class NeuralTrainer {
public:
	NeuralTrainer(BaseNet &input_net, ListLoss &input_loss, double lr, double acc = 0.01, long max_iter = 500, bool based_iter);

private:
	BaseNet trained_model_;

	ListLoss loss_func_;

	// learning parameter
	double learn_rate_;

	// epoch number
	int epoch_;
	int max_epochs_;

	// desired error
	double epsilon_;

	// current error in the training
	double current_error_;

	// if the stopping condition is based on the maximum number of epochs
	bool iter_based_;

	// utility function to get the stopping condition
	bool get_stop_condition();

};

#endif /* NEURALTRAINER_H_ */
