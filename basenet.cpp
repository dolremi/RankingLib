/*
 * basenet.cpp
 *
 *  The implementation of neural network class
 */

#include "basenet.h"

using std::cout;
using std::endl;
using std::fstream;
using std::vector;

BaseNet::BaseNet(int input_size, TransFunction &tf):
input_num_(input_size), hidden_num_(0), output_num_(1), active_func_(tf){

	// set input values zeros add one bias neuron with -1
	input_layer_.resize( input_num_ + 1, 0 );
	input_layer_[input_num_] = -1;

	cout << "The base net has been setup." << endl;
}

// Remember to call initialize_weights() after initialize the class
// each weight will assigned a small random number
// each weight delta will be set to zero
virtual void BaseNet::initialize_weights()
{
	cout << "Now initialize the weights of base neural network..." << endl;

	// set up the range of output layer weights
	double range_output = 1 / sqrt( static_cast<double>(input_num_));

	input_output_w_.resize(input_num_ + 1);
	input_output_d_.resize(input_num_ + 1);

	// set each weight to a small random number and set each weight update to zero
	for(int i = 0; i <= input_num_; ++i)
	{

		input_output_w_[i] = (static_cast<double>(rand()%100 + 1) / 100  * 2 * range_output - range_output);
		input_output_d_[i] = 0;

	}

	cout << "The initialization of the weights have completed" << endl;
}

// load the weight vectors from an input file
virtual bool BaseNet::load_weights(const string & input_filename)
{
  fstream input_file;
  input_file.open(input_filename.c_str(), std::ios::in);

  // The input file should have the format as following:
  // No. of input neurons  No. of hidden neurons No. of output neurons
  // then weights from input layer to output
  if(input_file.is_open())
  {
	  int input_num = 0;
	  int output_num = 0;
	  int hidden_num = 0;

	  input_file >> input_num;
	  input_file >> hidden_num;
	  input_file >> output_num;

	  // check if there is no hidden layer and the output is only one variable
	  if(hidden_num != 0 || output_num != 1)
	  {
		  cout << "Error: the size of output neurons or hidden neurons is not right, load weights fails." << endl;
		  return false;
	  }

	  // check if the size of neural network is the same as the current one
	  if(input_num_ != input_num)
	  {
		  cout << "Warning: the size of neural network has been changed, please verify." << endl;
	  }

	  // update the input number and input layer and weights
	  set_input_num(input_num);
	  input_layer_.resize(input_num + 1);
	  initialize_weights();

	  cout << "Read in the input layer to output layer weights..." << endl;

	  // read in input to hidden layer weights first
	  for(int i = 0 ; i <= input_num_; ++i)
	  {
		  input_file >> input_output_w_[i];
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
virtual bool BaseNet::save_weights(const string & output_filename)
{
	fstream output_file;
	output_file.open(output_filename.c_str(), std::ios::out);

	if(output_file.is_open())
	{
		cout << "Save the size of input layer, hidden layer and output layer..." << endl;
        output_file << input_num_ << " " << hidden_num_ << " " << output_num_ << endl;

        cout << "Save the weights from input layer to output layer..." << endl;

        for(int i = 0; i <= input_num_; ++i)
        {
        	output_file << input_output_w_[i] << " ";
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

// setter methods for size of three layers
inline void BaseNet::set_input_num(int input_num)
{
	input_num_ = input_num;
}

inline void BaseNet::set_hidden_num(int hidden_num)
{
	hidden_num_ = hidden_num;
}

inline void BaseNet::set_output_num(int output_num)
{
	output_num_ = output_num;
}

// getter methods for size of three layers
inline int BaseNet::get_input_num()
{
	return input_num_;
}

inline int BaseNet::get_hidden_num()
{
	return hidden_num_;
}

inline int BaseNet::get_output_num()
{
	return output_num_;
}

inline void BaseNet::set_input_layer(const vector<double> &input_layer)
{
	if(input_layer_.size() != input_layer.size())
		cout << "Warning: the size of input layer has been changed." << endl;

	input_layer_ = input_layer;
}

inline double BaseNet::get_output_result()
{
	return output_layer_;
}

inline vector<double> BaseNet::get_input_output_delta()
{
	return input_output_d_;
}

inline void BaseNet::set_input_output_delta(vector<double> &in_out_d)
{
	input_output_d_ = in_out_d;
}
// simple feed forward model from input layer to output layer
virtual void BaseNet::feed_forward(const vector<double> &input_layer)
{
	// set up the input layer
	set_input_layer(input_layer);

	cout << " Setting up output layer from input layer ..." << endl;

	output_layer_ = 0;

	for(int i = 0; i <= input_num_; ++i)
	{
		output_layer_ += input_output_w_[i] * input_layer_[i];
		output_layer_ = active_func_(output_layer_);
	}

	cout << "Now the feed forward is done." << endl;
}

// error gradient calculate from output layer to input layer
void BaseNet::error_grad_output_input()
{
	// The error gradient is calculated as delta = d(ActiveFunc)/d(totalInput) * input[i] = d(tranFunc)/d(w[i])
	cout << " Calculate the error gradient from output layer to input layer ..." << endl;

	delta_output_ = active_func_.derivative(output_layer_);

	for(int i = 0; i <= hidden_num_; ++i)
	{
		input_output_d_[i] = delta_output_ * input_layer_[i];
	}

}

// back propagation algorithm simply call one sub programs to calculate error gradient
virtual void BaseNet::back_prop()
{
	cout << " Start back propagation ..." << endl;
	error_grad_output_input();
	cout << " The back propagation is done" << endl;
}

virtual void BaseNet::update_weights(double learn_rate)
{
	for(int i = 0; i <= input_num_ ; ++i)
	{
		input_output_w_[i] -= learn_rate * input_output_d_[i];
	}
}
