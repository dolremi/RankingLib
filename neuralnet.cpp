/*
 * neuralnet.cpp
 *
 *  Created on: Apr 13, 2014
 *      Author: debian
 */

#include "neuralnet.h"

using std::cout;
using std::endl;
using std::fstream;

NeuralNet::NeuralNet(int input_size, int hidden_size, int output_size, TransFunction &tf): BaseNet(input_size, tf),
																						   hidden_num_(hidden_size),
																						   output_num_(output_size){

		cout << "input neurons No.\t hidden neurons No.\t output neurons No." << endl;
		cout << "\t" << input_num_ << "\t" << hidden_num_ << "\t" << output_num_ << endl;
		cout << "set up the input, hidden and output layers..." << endl;

		// set hidden values zeros add one bias neuron with -1
		hidden_layer_.resize( hidden_num_ + 1, 0 );
		hidden_layer_[hidden_num_] = -1;


}

// Remember to call initialize_weights() after the class instantiation
// each weight will assigned a small random number
// each weight delta will be set to zero
virtual void NeuralNet::initialize_weights()
{
	cout << "Now initialize the weights of full-size neural network..." << endl;

	// set the delta for hidden layer and the output layer
	delta_hidden_.resize( hidden_num_ + 1, 0);
	delta_output_ = 0;

	// set up the range of hidden and output layer weights
	double range_hidden = 1 / sqrt( static_cast<double>(input_num_));
	double range_output = 1 / sqrt( static_cast<double>(hidden_num_));

	input_hidden_w_.resize(input_num_ + 1);
	input_hidden_d_.resize(input_num_ + 1);

	hidden_output_w_.resize(hidden_num_ + 1);
	hidden_output_d_.resize(hidden_num_ + 1);

	// Set each weight with a small random number and each weight change with zero
	for(int i = 0; i <= input_num_; ++i)
	{
		for(int j = 0; j < hidden_num_; ++j)
		{
			input_hidden_w_[i].push_back(static_cast<double>(rand()%100 + 1) / 100  * 2 * range_hidden - range_hidden);
			input_hidden_d_[i].push_back(0);
		}
	}

	for(int i = 0; i <= hidden_num_; ++i)
	{
    	   hidden_output_w_[i] = static_cast<double>(rand()%100 + 1)/ 100 * 2 * range_output - range_output;
    	   hidden_output_d_[i] = 0;
	}

	cout << "The initialization of the weights have completed" << endl;
}

// load the weight vectors from an input file
virtual bool NeuralNet::load_weights(const string & input_filename)
{
  fstream input_file;
  input_file.open(input_filename.c_str(), std::ios::in);

  // The input file should have the format as following:
  // No. of input neurons  No. of hidden neurons No. of output neurons
  // then the weights from input to hidden layers
  // at last the weights from hidden to output layers
  if(input_file.is_open())
  {
	  int input_num = 0;
	  int output_num = 0;
	  int hidden_num = 0;

	  input_file >> input_num;
	  input_file >> hidden_num;
	  input_file >> output_num;

	  // check if the hidden layer exists and only one output variable
	  if(output_num != 1 || hidden_num == 0 )
	  {
		  cout << "Error: the size of output neurons or hidden neurons is not right, load weights fails." << endl;
		  return false;
	  }

	  // check if the size of neural network is the same as the current one
	  if(input_num_ != input_num || hidden_num_ != hidden_num)
	  {
		  cout << "Error: the size of neural network has been changed, please verify." << endl;
	  }

	  // update the input and hidden layer
	  set_input_num(input_num);
	  set_hidden_num(hidden_num);
	  input_layer_.resize(input_num_ + 1);
	  hidden_layer_.resize(hidden_num_ + 1);

	  // update the weights size
	  initialize_weights();

	  cout << "Read in the input layer to hidden layer weights..." << endl;

	  // read in input to hidden layer weights first
	  for(int i = 0 ; i <= input_num_; ++i)
	  {
		  for(int j = 0; j < hidden_num; ++j)
		  {
			  input_file >> input_hidden_w_[i][j];
		  }
	  }

	  cout << "Read in the hidden layer to output layer weights..." << endl;

	  // read in hidden to output layer weights then
	  for( int i = 0; i <= hidden_num_; ++i)
	  {
		  for(int j = 0; j < output_num_; ++j)
		  {
			  input_file >> hidden_output_w_[i][j];
		  }
	  }

	  cout << "The weights parameters have been loaded completely." << endl;
	  return true;
  }
  else
  {
	  cout << "Error! The input file cannot be opened. Please check the filename and pathname." << endl;
	  return false;
  }
}

// save the current weights into a file
bool NeuralNet::save_weights(const string & output_filename)
{
	fstream output_file;
	output_file.open(output_filename.c_str(), std::ios::out);

	if(output_file.is_open())
	{
		cout << "Save the size of input layer, hidden layer and output layer..." << endl;
        output_file << input_num_ << " " << hidden_num_ << " " << output_num_ << endl;

        cout << "Save the weights from input layer to hidden layer..." << endl;

        for(int i = 0; i <= input_num_; ++i)
        {
        	for(int j = 0; j < hidden_num_; ++j)
        	{
        		output_file << input_hidden_w_[i][j] << " ";
        	}
        	output_file << "\n";
        }

        cout << "Save the weights from hidden layer to output layer..." << endl;

        for(int i = 0; i <= hidden_num_; ++i)
        {
        	for(int j = 0; j < output_num_; ++j)
        	{
        		output_file << hidden_output_w_[i][j] << " ";
        	}
        	output_file << "\n";
        }

        cout << "All the weights have been saved" << endl;
		return true;
	}
	else
	{
		cout << "Error! The input file cannot be opened. Please check the filename and pathname." << endl;
		return false;
	}
}

inline vector<vector<double> > NeuralNet::get_input_hidden_delta()
{
	return input_hidden_d_;
}

inline vector<vector<double> > NeuralNet::get_hidden_output_delta()
{
	return hidden_output_d_;
}

// setter methods for weights change from input to hidden and hidden to output layer
inline void NeuralNet::set_input_hidden_delta(vector<vector<double> > &in_hid_d)
{
	input_hidden_d_ = in_hid_d;
}

inline void NeuralNet::set_hidden_output_delta(vector<vector<double> > &hid_out_d)
{
	hidden_output_d_ = hid_out_d;
}

// feed forward from input to hidden layers then hidden layers to output layers
virtual void NeuralNet::feed_forward(const vector<double> &input_var)
{
	// set up the input layer
	set_input_layer(input_layer);

	cout << " Setting up hidden layer from input layer ..." << endl;
	for(int j = 0; j < hidden_num_; ++j)
	{
		hidden_layer_[j] = 0;
		for(int i = 0; i <= input_num_; ++i)
		{
			hidden_layer_[j] += input_layer_[i] * input_hidden_w_[i][j];
		}

		hidden_layer_[j] = active_func_(hidden_layer_[j]);
	}

	cout << " Setting up output layer from hidden layer ..." << endl;

	output_layer_ = 0;
	for(int i = 0; i <= hidden_num_; ++i)
	{
		output_layer_ += hidden_layer_[i] * hidden_output_w_[i];
	}

	output_layer_ = active_func_(output_layer_);

	cout << "Now the feed forward is done." << endl;

}

void NeuralNet::error_grad_output_hidden()
{
	// The error gradient is calculated as delta = d(ActiveFunc)/d(totalInput) * input[i] = d(tranFunc)/d(w[i])
	cout << " Calculate the error gradient from output layer to hidden layer ..." << endl;

	delta_output_ = active_func_.derivative(output_layer_);

	for(int i = 0; i <= hidden_num_; ++i)
	{
	    hidden_output_d_[i] = delta_output_ * hidden_layer_[i];
	}
}

void NeuralNet::error_grad_hidden_input()
{
	// The error gradient is calculated as delta = d(ActiveFunc)/d(totalInput) * sum of weighted delta from hidden to output * w[i][j]
	cout << " Calculate the error gradient from input layer to hidden layer ..." << endl;

	// calculate the sum of weighted delta from the hidden to output layer
	for(int i = 0; i <  hidden_num_; ++i)
	{
		delta_hidden_[i] = 0;
		for(int j = 0; j < output_num_; ++j)
		{
		  	delta_hidden_[i] += hidden_output_w_[i][j] * delta_output_[j];
		}

		delta_hidden_[i] *= active_func_.derivative(hidden_layer_[i]);
	}

	// calculate the weight change from input to hidden layer
	for(int i = 0 ; i <= input_num_; ++i)
	{
		for(int j = 0; j < hidden_num_; ++j)
		{
			input_hidden_d_[i][j] = input_layer_[i]* delta_hidden_[j];
		}
	}
}

// back propagation algorithm simply call one two programs to calculate error gradient
void NeuralNet::back_prop()
{
	cout << " Start back propagation ..." << endl;

	error_grad_output_hidden();
	error_grad_hidden_input();

	cout << " The back propagation is done" << endl;
}

// update the weights from the internal deltas
void NeuralNet::update_weights(double learn_rate)
{
	for(int i = 0; i <= input_num_; ++i)
	{
		for(int j = 0; j < hidden_num_; ++j)
		{
			input_hidden_w_[i][j] -= learn_rate * input_hidden_d_[i][j];
		}
	}

	for(int i = 0; i <= hidden_num_; ++i)
	{
		for(int j = 0; j < output_num_; ++j)
		{
			hidden_output_w_[i][j] -= learn_rate * hidden_output_d_[i][j];
		}
	}
}
