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
		C(i) = (A(i,B(i)));
	return C;	
}

int argmax(arma::mat A){
	int ind = 0;
	double max = 0.0;
	for(int i=0;i<A.n_cols;i++){
		if(max<A(0,i)){
			max = A(0,i);
			ind = i;
		}
	}
	return ind;
}

arma::vec element_log(arma::vec A){
	for(int i=0;i<A.n_rows;i++)
		A(i) = log(A(i));
	return A;
}

void calculate_delta(arma::mat &A, arma::vec y){
	for(int i=0;i<A.n_rows;i++)
		A(i,y(i)) -= 1;	
}

void calculate_power(arma::mat &A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			A(i,j) = 1 - pow(A(i,j),2);
	}
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
	label_prob = element_log(label_prob);
	double loss = mean(label_prob)*train_size;
	loss+= lambda*(l2_norm(W_1)+l2_norm(W_2));
	loss/=train_size; 	
	return loss;		
}

double predict(arma::mat W_1, arma::mat W_2, arma::vec b_1, arma::vec b_2, arma::mat x){
	arma::mat z_1 = calculate(x,W_1,b_1);
	arma::mat a_1 = activate_tanh(z_1);
	arma::mat z_2 = calculate(a_1,W_2,b_2);
	arma::mat exp_score = element_exp(z_2);
	arma::vec sum = col_sum(exp_score,0);
	arma::mat probs = div_op(exp_score,sum);
	return argmax(probs);
}

double evaluate(arma::mat test,arma::mat W_1,arma::mat W_2,arma::vec b_1,arma::vec b_2,arma::vec y){
	double test_size = test.n_rows;
	int correct = 0;
	for(int i=0;i<test_size;i++){
		arma::mat x(1,test.n_cols);
		for(int j=0;j<test.n_cols;j++)
			x(0,j) = test(i,j);
		if(predict(W_1,W_2,b_1,b_2,x)==y(i))
			correct++;
	}
	return correct/test_size;
}

void build_model(arma::mat X, arma::mat &W_1, arma::mat &W_2, arma::vec &b_1, arma::vec &b_2, arma::vec y ,double epsilon, double lambda){
	int iterations = 1000;
	while(iterations>0){
		arma::mat z_1 = calculate(X,W_1,b_1);
		arma::mat a_1 = activate_tanh(z_1);
		arma::mat z_2 = calculate(a_1,W_2,b_2);
		arma::mat exp_score = element_exp(z_2);
		arma::vec sum = col_sum(exp_score,0);
		arma::mat probs = div_op(exp_score,sum);		

		arma::mat delta3 = probs;	
		calculate_delta(delta3,y);
		arma::mat dW_2 = (a_1.t()) * delta3;  
		arma::vec db2 = col_sum(delta3,1);
		calculate_power(a_1);
		arma::mat delta2 = (delta3 * W_2.t()) % a_1;
		arma::mat dW_1 = X.t() * delta2;
		arma::vec db1 = col_sum(delta2,1);

		dW_2 = dW_2 + mul_scalar(W_2,lambda);
		dW_1 = dW_1 + mul_scalar(W_1,lambda);
		

		W_1 = W_1 + mul_scalar(dW_1,-epsilon);
		W_2 = W_2 + mul_scalar(dW_2,-epsilon);
		b_1 = b_1 + mul_scalar(db1,-epsilon);
		b_2 = b_2 + mul_scalar(db2,-epsilon);
	
		iterations--;
	}	
}

int main(int argc, char *argv[]){
	arma::mat X = read_csv("moon_datasets");
	//std::cout << X.n_rows << " " << X.n_cols << std::endl;
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
	build_model(train,W_1,W_2,b_1,b_2,y,epsilon,lambda);
	std::cout<< "model learnt" << std::endl;
	std::cout<<" accuracy:" << evaluate(train,W_1,W_2,b_1,b_2,y);
	return 0;
}
