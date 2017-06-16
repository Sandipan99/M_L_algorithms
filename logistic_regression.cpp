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

implements logistic regression
input: feature matrix and class
output: accuracy
@author Sandipan Sikdar

**/

int main(int argc, char *argv[]){
	arma::mat X = read_csv("test_data");
	arma::mat train = remove_col(X,-1);
	std::cout<< "split train matrix" << std::endl;
	arma::vec y = find_class(X);
	arma::mat W = fill_random(train.n_cols,1);
	double 
	return 0;
}
