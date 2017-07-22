/*
simple application of metropolis-hastings sampling to obtain sigma of a bivariate normal distribution...
@author Sandipan Sikdar
*/

#include<iostream>
#include "distribution.h"
#include "common.h"
#include<fstream>

int main(int argc, char* argv[]){
	int dim = 2,freq = 1000;
	arma::vec mu = {0,0};
	arma::mat sigma = {{1,0},{0,1}};
	arma::mat data;
	data = multivariate_normal(mu,sigma,dim,freq);
	std::ofstream out;
	out.open("mul_var_dist");
	for(int j=0;j<freq;j++)
		out<<data(0,j) << "\t" <<data(1,j)<<std::endl;
	out.close();
	return 0;
}
