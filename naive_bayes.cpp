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
/**

implements basic naive base classification agorithm
input: feature matrix and class
output: accuracy
@author Sandipan Sikdar

**/

std::vector<double> split(char *line,std::vector<double> &con){
	char *ptr;
	ptr = strtok(line,",");
	while(ptr!=NULL){
		con.push_back(atof(ptr));
		ptr = strtok(NULL,",");
	}
	return con;		
}

arma::mat read_csv(std::string fname){
	int size;
	std::ifstream in;
	std::vector<double> con;
	std::vector<std::vector<double> > feat;
	in.open(fname.c_str());
	char line[1000];
	while(in >> line){
		con = split(line,con);
		feat.push_back(con);
		size = (int)(con.size());
		con.clear();
	}
	in.close();
	arma::mat A((int)(feat.size()),size);
	for(int i=0;i<(int)feat.size();i++)
		for(int j=0;j<size;j++)
			A(i,j) = feat[i][j];
	return A;
}

void display(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++){
			std::cout<<A(i,j)<<" ";
		}
		std::cout<< std::endl;
	}
}

bool present(std::vector<int> vec, int a){
	std::vector<int>::iterator it;
	it = find(vec.begin(),vec.end(),a);
	if(it==vec.end())
		return false;
	return true;	
}

bool present(std::map<int,int> m, int a){
	if(m.find(a)==m.end())
		return false;
	else	return true;
}

void find_vec(std::vector<int> &vec, int t_s, int r){
	float rnd;
	while(t_s>0){
		for(int i=0;i<r;i++){
			if(!present(vec,i)){
				rnd = (float) (rand())/RAND_MAX;
				if(rnd<0.5){
					t_s--;
					vec.push_back(i);
				}
			}
			if(t_s==0)
				break;
		}
	}
}

arma::mat fill(arma::mat A, arma::mat B, std::vector<int> v, int c){
	std::vector<int>::iterator it;
	int r;
	int te_c = 0;
	for(it=v.begin();it!=v.end();it++){
		r = *it;
		for(int j=0;j<c;j++)
			B(te_c,j) = A(r,j);
		te_c++;
	}
	return B;
}

arma::mat fill_complement(arma::mat A, arma::mat B, std::vector<int> v, int r , int c){
	int te_c = 0;
	for(int i=0;i<r;i++){
		if(!present(v,i)){
			for(int j=0;j<c;j++)
				B(te_c,j) = A(i,j);
			te_c++;
		}
	}
	return B;
}

std::map<int,int> FindClass(arma::mat A){
	std::map<int,int> cl;
	int c = A.n_cols;
	for(int i=0;i<A.n_rows;i++){
		if(!present(cl,A(i,c-1)))
			cl[A(i,c-1)]=1;
		else	cl[A(i,c-1)]++;
	}
	return cl;
}

std::map<int,arma::vec> ClassSummaryMean(std::map<int,int> c_l, arma::mat A){
	int a,b,cnt;
	int r = A.n_rows;
	int c = A.n_cols;
	std::map<int,arma::vec> M;
	std::map<int,int>::iterator it;
	arma::mat B;
	arma::vec V;
	for(it=c_l.begin();it!=c_l.end();it++){
		a = it->first;
		b = it->second;
		B.set_size(b,c-1);
		V.set_size(c-1);
		cnt = 0;
		for(int i=0;i<r;i++){
			if(A(i,c-1)==a){
				for(int j=0;j<c-1;j++)
					B(cnt,j) = A(i,j);
				cnt++;
			}
		}	
		V = mean(B);
		M[a] = V; 
	}
	return M;
}

std::map<int,arma::vec> ClassSummaryStd(std::map<int,int> c_l, arma::mat A){
	int a,b,cnt;
	int r = A.n_rows;
	int c = A.n_cols;
	std::map<int,arma::vec> M;
	std::map<int,int>::iterator it;
	for(it=c_l.begin();it!=c_l.end();it++){
		a = it->first;
		b = it->second;
		arma::mat B(b,c-1);
		arma::vec V(b);
		cnt = 0;
		for(int i=0;i<r;i++){
			if(A(i,c-1)==a){
				for(int j=0;j<c-1;j++)
					B(cnt,j) = A(i,j);
				cnt++;
			}
		}
		V = std_dev(B);
		M[a] = V; 
	}
	return M;
}

double CalculateProbability(double m,double s,double x){
	double prob = 1.0/(s*sqrt(2*3.14));
	double a = pow(x-m,2);
	a/=2*s*s;
	prob *= exp(a);
	return prob;
}

std::map<int,double> FindClassProbability(std::map<int,arma::vec> M, std::map<int,arma::vec> S,arma::vec B,std::map<int,double> cl_prior){
	std::map<int,double> class_prob;
	std::vector<int> C_l;
	int c = B.n_cols;
	std::map<int,arma::vec>::iterator it;
	for(it=M.begin();it!=M.end();it++)
		C_l.push_back(it->first);

	for(int i=0;i<(int)(C_l.size());i++){
		double m,s,x,prob=1.0;
		for(int j=0;j<c;j++){
			m = M[C_l[i]](j);
			s = S[C_l[i]](j);
			x = B(j);
			prob*=CalculateProbability(m,s,x);	
		}
		class_prob[C_l[i]] = prob*cl_prior[C_l[i]];
	}
	return class_prob;
}

std::map<int,double> CalculatePrior(std::map<int,int> A){
	std::map<int,double> B;
	std::map<int,int>::iterator it;
	double a,b,sum=0.0;
	for(it=A.begin();it!=A.end();it++){
		a = it->first;
		b = it->second;
		B[a] = (double)(b);
		sum+=(double)(b);
	}
	for(it=A.begin();it!=A.end();it++){
		a = it->first;
		B[a] /= sum; 
	}
	return B;
}

int FindMax(std::map<int,double> A){
	int c_l = 0;
	double p = 0.0;
	std::map<int,double>::iterator it;
	for(it=A.begin();it!=A.end();it++){
		if (it->second > p){
			p = it->second;
			c_l = it->first;
		}	
	}
	return c_l;
}

double CalculateAccuracy(arma::mat Test, std::map<int,arma::vec> M, std::map<int,arma::vec> S, std::map<int,double> cl_prior){
	double accr = 0.0;
	std::map<int,double> c_p;
	int c,c_t;
	int x = Test.n_rows;
	int y = Test.n_cols;
	for(int i=0;i<x;i++){
		c = Test(i,y-1);
		arma::vec A(y-1);
		for(int j=0;j<y-1;j++)
			A(j) = Test(i,j);
		c_p = FindClassProbability(M,S,A,cl_prior);
		c_t = FindMax(c_p);
		if(c_t==c)
			accr+=1;
	}
	return accr/x;
}


int main(int argc, char *argv[]){
	arma::mat feature_mat = read_csv("pima-indians-diabetes.data.txt");
	int data_p = feature_mat.n_rows;
	double accr;
	int dim = feature_mat.n_cols;
	float split_ratio = 0.6;
	int train_size = (int)(data_p*split_ratio);
	arma::mat train(train_size,dim);
	arma::mat test(data_p-train_size,dim);
	std::vector<int> vec;
	std::map<int,int> c_l;
	std::map<int,double> cl_prior;
	std::map<int,arma::vec> M_n_train;
	std::map<int,arma::vec> S_d_train;
	find_vec(vec,train_size,data_p);
	train = fill(feature_mat,train,vec,dim);
	test = fill_complement(feature_mat,test,vec,data_p,dim);
	c_l = FindClass(train);
	M_n_train = ClassSummaryMean(c_l,train);
	S_d_train = ClassSummaryStd(c_l,train);
	cl_prior = CalculatePrior(c_l);
	accr = CalculateAccuracy(test,M_n_train,S_d_train,cl_prior);
	std::cout << "accuracy: " << accr << std::endl;
	return 0;
}

