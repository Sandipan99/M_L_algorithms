#include<stdlib.h>
#include<time.h>
#include<iostream>
#include<vector>
#include<math.h>
#include<map>
#include<fstream>
#include<string>
#include<algorithm>
#include "stat.h"
#include "common.h"
/**

implements basic 3 layer neural network
input: feature matrix and class
output: accuracy
@author Sandipan Sikdar

**/
arma::mat calculate(arma::mat X, arma::mat W, arma::vec b){
	arma::mat z = X * W;
	for(int i=0;i<z.n_rows;i++){
		for(int j=0;j<z.n_cols;j++){
			z(i,j)+=b(j);
		}
	}
	return z;
}


double calculate_loss(arma::mat X, arma::mat W_1, arma::mat W_2, arma::vec b_1, arma::vec b_2){
	arma::mat z_1 = calculate(X,W_1,b_1);
	arma::mat a_1 = activate_tanh(z_1);
	std::cout << a_1.n_rows << "," << a_1.n_cols << std::endl;
	return 0.0;		
}

int main(int argc, char *argv[]){
	arma::mat X = read_csv("moon_datasets");
	std::cout << X.n_rows << " " << X.n_cols << std::endl;
	int num_exp = X.n_rows;
	int input_dim = X.n_cols;
	int output_dim = 2;
	int h_l = 3;
	double epsilon = 0.01; // learning rate
	double lambda = 0.01; // regularizer
	arma::mat W_1 = fill_random(input_dim,h_l);
	arma::vec b_1 = fill_random(h_l);
	arma::mat W_2 = fill_random(h_l,output_dim);
	arma::vec b_2 = fill_random(output_dim);
	std::cout << "initialization done " << std::endl;
	std::cout << calculate_loss(X,W_1,W_2,b_1,b_2);
	return 0;
}
