#include<stdlib.h>
#include<time.h>
#include<iostream>
#include<armadillo>
#include<vector>
#include<math.h>
#include<fstream>
#include<string>
#include<algorithm>

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
	in.open(fname);
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
	std::cout << A.n_rows << " , " << A.n_cols << std::endl;
	std::cout << B.n_rows << " , " << B.n_cols << std::endl;
	std::vector<int>::iterator it;
	int r;
	int te_c = 0;
	for(it=v.begin();it!=v.end();it++){
		r = *it;
		//std::cout << r << std::endl;
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

int main(int argc, char *argv[]){
	arma::mat feature_mat = read_csv("pima-indians-diabetes.data.txt");
	int data_p = feature_mat.n_rows;
	int dim = feature_mat.n_cols;
	int train_size = (int)(data_p*0.67);
	arma::mat train(train_size,dim);
	arma::mat test(data_p-train_size,dim);
	std::vector<int> vec;
	find_vec(vec,train_size,data_p);
	std::cout << vec.size() << std::endl;
	train = fill(feature_mat,train,vec,dim);
	std::cout << "obtained training matrix" << std::endl;
	test = fill_complement(feature_mat,test,vec,data_p,dim);
	std::cout << "obtined test matrix" << std::endl;
	return 0;
}
