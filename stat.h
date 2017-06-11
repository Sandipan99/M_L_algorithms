#ifndef __STAT_H_INCLUDED__
#define __STAT_H_INCLUDED__

#include<vector>
#include<armadillo>

double mean(std::vector<int>);
double mean(std::vector<double>);
double std_dev(std::vector<int>);
double std_dev(std::vector<double>);
//double median(std::vector<T>);

arma::vec mean(arma::mat);
arma::vec std_dev(arma::mat);
double l1_norm(arma::mat);
double l2_norm(arma::mat);
//arma::vec median(arma::mat);


#endif
