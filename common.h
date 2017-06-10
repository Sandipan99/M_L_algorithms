#ifndef __COMMON_H_INCLUDED__
#define __COMMON_H_INCLUDED__

#include<vector>
#include<armadillo>
#include<string>
#include<map>

std::vector<double> split(char *line,std::vector<double>&);
arma::mat read_csv(std::string);
void display(arma::mat);
bool present(std::vector<int> vec, int a);
bool present(std::map<int,int>, int a);
arma::mat fill_random(int,int);
arma::vec fill_random(int);
arma::mat activate_tanh(arma::mat);
arma::mat activate_relu(arma::mat);
arma::mat element_exp(arma::mat);
arma::mat remove_col(arma::mat,int);
arma::vec find_class(arma::mat);
arma::vec col_sum(arma::mat,int);
#endif
