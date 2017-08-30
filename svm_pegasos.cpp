#include<stdlib.h>
#include<iostream>
#include<math.h>
#include<fstream>
#include "stat.h"
#include "common.h"
/**

implements support vector machine with pegasos subgradient solver (http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf)
input: feature matrix and class(two)
output: accuracy
@author Sandipan Sikdar

**/
int sign(double a){
	if(a<0)
		return -1;
	return 1;
}

arma::vec sample(arma::mat A, int num){
	arma::vec point(A.n_cols);
	for(int i=0;i<A.n_cols;i++)
		point(i) = A(num,i);
	return point;
}

void svn(arma::mat train, arma::vec y, arma::mat W, int train_size){
	for(int i=0;i<train.n_rows;i++){
		if(y(i)==0)
			y(i) = -1;
	}
	arma::mat X(train_size,train.n_cols);  // separating out the training set....
	for(int i=0;i<train_size;i++){
		for(int j=0;j<train.n_cols;j++)
			X(i,j)=train(i,j);
	}
	double lambda = 0.00024; // regularization prameter
	// pegasos algorithm-----
	for(int i=1;i<train_size;i++){	
		int num = rand()%train_size;
		arma::vec point = sample(X,num);
		double ita_t = 1/(i*lambda);
		double decision = 0.0;
		for(int j=0;j<train.n_cols;j++)
			decision+=W(j)*point(j);
		W = mul_scalar(W,1-ita_t*lambda);
		decision*=y(num);
		if(decision<1){
			point = mul_scalar(point,ita_t*y(num));		
			W = W + point; 
		}
	}
	std::cout << "Obtained Weights" << std::endl;
	// calculate accuracy-----
	int cc = 0,e_x=0;
	double r;
	for(int i=train_size;i<train.n_rows;i++){
		r = 0.0;
		e_x++;
		for(int j=0;j<train.n_cols;j++)
			r+=W(j)*train(i,j);
		if(sign(r)==y(i)){	
			cc++;
		}
	}
		
	std::cout << "accuracy: " << (double)(cc)/e_x << std::endl;
			
}


int main(int argc, char* argv[]){
	arma::mat X = read_csv("test_data");
	X = shuffle(X);
	arma::mat train = remove_col(X,-1);
	std::cout<< "split train matrix" << std::endl;
	std::cout<< "train " << train.n_rows << "," << train.n_cols << std::endl;
	arma::vec y = find_class(X);
	arma::mat W = fill_zero(train.n_cols,1);
	std::cout << "Initialization done" << std::endl;
	int train_size = (int)(0.8*train.n_rows);
	svn(train,y,W,train_size);
	return 0;
}
