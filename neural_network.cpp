#include<stdlib.h>
#include<time.h>
#include<iostream>
#include<vector>
#include<math.h>
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
/*
void sanity(arma::mat A){
	int flag = 0;
	for(int i=0;i<A.n_rows;i++){
		double sum = 0.0;
		for(int j=0;j<A.n_cols;j++)
			sum+=A(i,j);
		std::cout << sum << std::endl;	
	}
}*/

arma::vec filter_prob(arma::mat A, arma::vec B){
	arma::vec C(A.n_rows);
	for(int i=0;i<A.n_rows;i++)
		C(i) = -log(A(i,B(i)));
	return C;	
}

double calculate_loss(arma::mat X, arma::mat W_1, arma::mat W_2, arma::vec b_1, arma::vec b_2, arma::vec y, double lambda){
	int train_size = X.n_rows;
	arma::mat z_1 = calculate(X,W_1,b_1);
	arma::mat a_1 = activate_tanh(z_1);
	arma::mat z_2 = calculate(a_1,W_2,b_2);
	arma::mat exp_score = element_exp(z_2);
	std::cout<<"exp_score obtained" << std::endl;
	arma::vec sum = col_sum(exp_score,0);
	std::cout<<"col sum obtained" << std::endl;
	arma::mat probs = div_op(exp_score,sum);
	std::cout<<probs.n_rows<<","<<probs.n_cols<<std::endl;
	arma::vec label_prob = filter_prob(probs,y);
	double loss = mean(label_prob)*train_size;
	loss+= lambda*(l2_norm(W_1)+l2_norm(W_2));
	loss/=train_size; 	
	return loss;		
}

int main(int argc, char *argv[]){
	arma::mat X = read_csv("moon_datasets");
	std::cout << X.n_rows << " " << X.n_cols << std::endl;
	int num_exp = X.n_rows;
	int input_dim = X.n_cols-1;
	int output_dim = 2;
	int h_l = 3;
	double epsilon = 0.01; // learning rate
	double lambda = 0.01; // regularizer
	arma::mat train = remove_col(X,-1);
	std::cout<< "split train matrix" << std::endl;
	arma::vec y = find_class(X);	
	arma::mat W_1 = fill_random(input_dim,h_l);
	arma::vec b_1 = fill_random(h_l);
	arma::mat W_2 = fill_random(h_l,output_dim);
	arma::vec b_2 = fill_random(output_dim);
	std::cout << "initialization done " << std::endl;
	std::cout << calculate_loss(train,W_1,W_2,b_1,b_2,y,lambda);
	return 0;
}
