#include "stat.h"
#include<math.h>

double mean(std::vector<int> A){
	double sum = 0.0;
	std::vector<int>::iterator it;
	for(it=A.begin();it!=A.end();it++){
		sum+=*it;
	}
	sum/=(double)(A.size());
	return sum;	
}

double mean(std::vector<double> A){
	double sum = 0.0;
	std::vector<double>::iterator it;
	for(it=A.begin();it!=A.end();it++){
		sum+=*it;
	}
	sum/=(double)(A.size());
	return sum;	
}


double std_dev(std::vector<int> A){
	double sum = 0.0;
	std::vector<int>::iterator it;
	double m = mean(A);
	for(it=A.begin();it!=A.end();it++){
		sum+= pow(*it-m,2);
	}
	sum/= (double)(A.size());
	return sqrt(sum);
}

double std_dev(std::vector<double> A){
	double sum = 0.0;
	std::vector<double>::iterator it;
	double m = mean(A);
	for(it=A.begin();it!=A.end();it++){
		sum+= pow(*it-m,2);
	}
	sum/= (double)(A.size());
	return sqrt(sum);
}

arma::vec mean(arma::mat A){
	int x = A.n_rows;
	int y = A.n_cols;
	arma::vec B(y);
	double sum;
	for(int i=0;i<y;i++){
		sum = 0.0;
		for(int j=0;j<x;j++){
			sum+=A(j,i);
		}	
		B(i) = sum/x;	
	}
	return B;
}

arma::vec std_dev(arma::mat A){
	int x = A.n_rows;
	int y = A.n_cols;
	arma::vec M(y);
	arma::vec B(y);
	M = mean(A);
	double sum,m_n;
	for(int i=0;i<y;i++){
		sum = 0.0;
		m_n = M(i);
		for(int j=0;j<x;j++){
			sum+= pow(A(j,i)-m_n,2);
		}
		sum/=x;
		B(i) = sqrt(sum);
	}
	return B;
}

/*
double median(std::vector<T> A){
	
}*/
