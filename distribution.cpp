#include<iostream>
#include<math.h>
#include<stdlib.h>
#include "distribution.h"
/**
@author Sandipan Sikdar
**/
#define pi 3.14159265

double* univariate_normal(float mu, float sigma, int freq){
	double* data = (double*)malloc(freq*sizeof(double));
	int i = 0;
	double u_1,u_2,z_1,z_2;
	while(i<freq){
		u_1 = rand()*(1.0/RAND_MAX);
		u_2 = rand()*(1.0/RAND_MAX);
		z_1 = sqrt(-2*log(u_1))*cos(2*pi*u_2);
		z_2 = sqrt(-2*log(u_1))*sin(2*pi*u_2);
		data[i] = z_1*sigma + mu;
		i++;		
	}
	return data;
}

double* exponential(float lambda, int freq){
	double* data = (double*)malloc(freq*sizeof(double));
	int i = 0;
	double u,z;
	while(i<freq){
		u = rand()*(1.0/RAND_MAX);
		z = (-1/lambda)*log(1-u);
		data[i] = z;
		i++;
	}
	return data;
}

