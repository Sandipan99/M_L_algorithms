/**
@author Sandipan Sikdar
**/
#include "common.h"
#include<stdlib.h>
#include<math.h>

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

void display_dimension(arma::mat A){
	std::cout<< A.n_rows << "," << A.n_cols << std::endl;
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

arma::mat fill_random(int r, int c){
	arma::mat A(r,c);
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			A(i,j) = (double)(rand())/RAND_MAX;	
		}
	}
	return A;
}

arma::vec fill_random(int s){
	arma::vec A(s);
	for(int i=0;i<s;i++)
		A(i) = (double)(rand())/RAND_MAX;
	return A;
}

arma::mat fill_zero(int r, int c){
	arma::mat A(r,c);
	for(int i=0;i<r;i++){
		for(int j=0;j<c;j++){
			A(i,j) = 0.0;	
		}
	}
	return A;
	
}

arma::vec fill_zero(int s){
	arma::vec A(s);
	for(int i=0;i<s;i++)
		A(i) = 0.0;
	return A;
}

arma::mat activate_tanh(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++){
			A(i,j) = tanh(A(i,j));
		}
	}
	return A;
}

arma::mat activate_relu(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++){
			if(A(i,j)<0)
				A(i,j) = 0.0;
			else	continue;
		}
	}
	return A;
}

double sigmoid(double val){
	double x = 1/(1+exp(-val));
	return x;
}

arma::mat activate_sigmoid(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++){
			A(i,j) = sigmoid(A(i,j));
		}
	}
	return A;
}


arma::mat element_exp(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			A(i,j) = exp(A(i,j));
	}
	return A;
}

arma::mat element_neglog(arma::mat A){
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			A(i,j) = -log(A(i,j));
	}
	return A;
}


arma::mat remove_col(arma::mat A,int c){	
	arma::mat B(A.n_rows,A.n_cols-1);
	if(c==-1){
		for(int i=0;i<A.n_rows;i++){
			for(int j=0;j<A.n_cols-1;j++){
				B(i,j) = A(i,j);
			}
		}
	} 
	else{
		for(int i=0;i<A.n_rows;i++){		
			int k = 0;
			for(int j=0;j<A.n_cols-1;j++){
				if(j!=c){
					B(i,k) = A(i,j);
					k++;
				}
			}
		}
	}
	return B;
}

arma::vec find_class(arma::mat A){	
	arma::vec Y(A.n_rows);
	for(int i=0;i<A.n_rows;i++){
		Y(i) = A(i,A.n_cols-1);
	}
	return Y;
}

arma::vec col_sum(arma::mat A,int axis){
	if(axis==0){ // row_wise addition
		arma::vec V(A.n_rows);
		for(int i=0;i<A.n_rows;i++){
			double sum = 0.0;
			for(int j=0;j<A.n_cols;j++)
				sum+=A(i,j);
			V(i) = sum;
		}
		return V;
	}
	else{ // column_wise addition
		arma::vec V(A.n_cols);
		for(int i=0;i<A.n_cols;i++){
			double sum = 0.0;
			for(int j=0;j<A.n_rows;j++)
				sum+=A(j,i);
			V(i) = sum;
		}
		return V;
	}
}
arma::mat div_op(arma::mat A,arma::vec V){
	arma::mat B(A.n_rows,A.n_cols);
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++){
			B(i,j) = A(i,j)/V(i);
		}
	}
	return B;
}

arma::mat mul_scalar(arma::mat A, double x){
	arma::mat B(A.n_rows,A.n_cols);
	for(int i=0;i<A.n_rows;i++){
		for(int j=0;j<A.n_cols;j++)
			B(i,j) = A(i,j)*x;
	}
	return B;
}

arma::vec mul_scalar(arma::vec A, double x){
	arma::vec B(A.n_rows);
	for(int i=0;i<A.n_rows;i++)
		B(i) = A(i)*x;
	return B;
}

arma::mat shuffle(arma::mat A){
	int s_a,s_b;
	double temp;
	int i = (int)(0.5*A.n_rows);
	int limit = A.n_rows;
	while(i>0){
		s_a = rand()%limit;
		s_b = rand()%limit;
		while(s_b==s_a)
			s_b = rand()%limit;
		for(int k=0;k<A.n_cols;k++){
			temp = A(s_a,k);
			A(s_a,k) = A(s_b,k);
			A(s_b,k) = temp;
		}	
		i--;	
	}
	return A;	
}

int max(int a, int b){
	if(a>b)
		return a;
	return b;
}

double vector_dot(arma::vec A, arma::vec B){
	int x = A.n_rows;
	int y = A.n_cols;
	double res = 0.0;
	for(int i=0;i<max(x,y);i++)
		res+=A(i)*B(i);
	return res;	
}

