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
