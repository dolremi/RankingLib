/*
 * sigmoid.h
 *
 *  Created on: Apr 11, 2014
 *      Author: debian
 */

#ifndef SIGMOID_H_
#define SIGMOID_H_

#include <cmath>

#include "transfunction.h"

class Sigmoid: public TransFunction {
	public:
	inline virtual double operator()(double x)
	{
		return 	1.0 / (1.0 + exp(-x));


	}
	inline virtual double derivative(double x)
	{
		return x * ( 1 - x );
	}
};

#endif /* SIGMOID_H_ */
