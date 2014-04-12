/*
 * TransFunction.h
 *
 *  Created on: Apr 11, 2014
 *      Author: debian
 */

#ifndef TRANSFUNCTION_H_
#define TRANSFUNCTION_H_

class TransFunction {
	public:
	inline virtual double operator()(double x)
	{
		return x;
	}
	inline virtual double derivative(double x)
	{
		return 1;
	}
};

#endif /* TRANSFUNCTION_H_ */
