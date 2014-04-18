/*
 * neuraltrainer.h
 *
 *  trainer class to train the data set
 *
 */

#ifndef NEURALTRAINER_H_
#define NEURALTRAINER_H_

#include "basenet.h"
#include  "listnetloss.h"

#include <string>
class NeuralTrainer {
public:
	NeuralTrainer(BaseNet &input_net, ListLoss &input_loss, bool base_iter, double lr , double acc , long max_iter);

private:

	// Neural network based model
	BaseNet trained_model_;

	// listwise loss function
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
