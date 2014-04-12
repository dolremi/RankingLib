/*
 * hypertangent.h
 *
 *  Created on: Apr 11, 2014
 *      Author: debian
 */

#ifndef HYPERTANGENT_H_
#define HYPERTANGENT_H_

#include <cmath>

#include "transfunction.h"

class HyperTangent: public TransFunction {
	public:
	inline virtual double operator()(double x)
	{
		return tanh(x);
	}
	inline virtual double derivative(double x)
	{
		return 1 - pow(x, 2);
	}
};

#endif /* HYPERTANGENT_H_ */
