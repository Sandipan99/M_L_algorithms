#ifndef __DISTRIBUTION_H_INCLUDED__
#define __DISTRIBUTION_H_INCLUDED__
#include<armadillo>

double* univariate_normal(float,float,int); 
double* exponential(float,int);
arma::mat multivariate_normal(arma::vec,arma::mat,int,int);

#endif
