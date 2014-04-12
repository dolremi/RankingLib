/*
 * neuralnet.cpp
 *
 *  The implementation of neural network class
 */

#include "neuralnet.h"

using std::cout;
using std::endl;
using std::fstream;
using std::vector;

NeuralNet::NeuralNet(int input_n, int hidden_n, int output_n, TransFunction &tf):
input_num_(input_n), hidden_num_(hidden_n), output_num_(output_n), active_func_(tf){

	cout << "input neurons No.\t hidden neurons No.\t output neurons No." << endl;
	cout << "\t" << input_num_ << "\t" << hidden_num_ << "\t" << output_num_ << endl;
	cout << "set up the input, hidden and output layers..." << endl;
	// set input values zeros add one bias neuron with -1
	input_layer_.resize( input_num_ + 1, 0 );
	input_layer_[input_num_] = -1;

	// set hidden values zeros add one bias neuron with -1
	hidden_layer_.resize( hidden_num_ + 1, 0 );
	hidden_layer_[hidden_num_] = -1;

	// set output values zeros
	output_layer_.resize( output_num_ , 0 );
	cout << "Three layers have been setup." << endl;
	// initialize the weights
	initialize_weights();
}

// each weight will assigned a small random number
// each weight delta will be set to zero
void NeuralNet::initialize_weights()
{
	cout << "Now initialize the weights of neural network..." << endl;
	// set up the range of hidden and output layer weights
	double range_hidden = 1 / sqrt( static_cast<double>(input_num_));
	double range_output = 1 / sqrt( static_cast<double>(hidden_num_));

	input_hidden_w_.resize(input_num_ + 1);
	input_hidden_d_.resize(input_num_ + 1);

	hidden_output_w_.resize(hidden_num_ + 1);
	hidden_output_d_.resize(hidden_num_ + 1);

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
       for(int j = 0; j < output_num_; ++j)
       {
    	   hidden_output_w_[i].push_back(static_cast<double>(rand()%100 + 1)/ 100 * 2 * range_output - range_output);
    	   hidden_output_d_[i].push_back(0);
       }
	}

	cout << "The initialization of the weights have completed" << endl;
}

// load the weight vectors from an input file
bool NeuralNet::load_weights(const string & input_filename)
{
  fstream input_file;
  input_file.open(input_filename.c_str(), std::ios::in);

  // The input file should have the format as following:
  // No. of input neurons  No. of hidden neurons No. of output neurons
  // then the weights from input to hidden layers
  // at last the weights from hidden to output layers

  if(input_file.is_open())
  {
	  // check if the size of neural network is the same as the current one

	  int input_num = 0;
	  int output_num = 0;
	  int hidden_num = 0;

	  input_file >> input_num;
	  input_file >> hidden_num;
	  input_file >> output_num;

	  if(input_num_ != input_num || hidden_num_ != hidden_num || output_num_ != output_num)
	  {
		  cout << "Warning: the size of neural network has been changed, please verify." << endl;
	  }

	  // update the size of the current neural network
	  set_input_num(input_num);
	  set_hidden_num(hidden_num);
	  set_output_num(output_num);

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

// setter methods for size of three layers
inline void NeuralNet::set_input_num(int input_num)
{
	input_num_ = input_num;
}

inline void NeuralNet::set_hidden_num(int hidden_num)
{
	hidden_num_ = hidden_num;
}

inline void NeuralNet::set_output_num(int output_num)
{
	output_num_ = output_num;
}

// getter methods for size of three layers
inline int NeuralNet::get_input_num()
{
	return input_num_;
}

inline int NeuralNet::get_hidden_num()
{
	return hidden_num_;
}

inline int NeuralNet::get_output_num()
{
	return output_num_;
}

inline void NeuralNet::set_input_layer(const vector<double> &input_layer)
{
	if(input_layer_.size() != input_layer.size())
		cout << "Warning: the size of input layer has been changed." << endl;

	input_layer_ = input_layer;
}

inline vector<double> NeuralNet::get_output_layer()
{
	return output_layer_;
}

void NeuralNet::feed_forward(const vector<double> &input_layer)
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

		hidden_layer_[j] = TransFunction(hidden_layer_[j]);
	}

	cout << " Setting up output layer from hidden layer ..." << endl;
	for(int j =0; j < output_num_; ++j)
	{
		output_layer_[j] = 0;
		for(int i = 0; i <= hidden_num_; ++i)
		{
			output_layer_[j] += hidden_layer_[i] * hidden_output_w_[i][j];
		}

		output_layer_[j] = TransFunction(output_layer_[j]);
	}

	cout << "Now the feed forward is done." << endl;

}
