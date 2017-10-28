#include<iostream>
#include<iostream>
#include<math.h>
#include<fstream>
#include<map>
#include "stat.h"
#include "common.h"

/*
Implements SMOTE
Input: feature matrix with class, N-> percentage of oversampling
Output: feature matrix with oversampled minority class
@author Sandipan Sikdar
*/

int find_max(std::map<int,int> c_l){
	int max_val = 0;
	int max_ind = 0;
	std::map<int,int>::iterator it;
	for(it=c_l.begin();it!=c_l.end();it++){
		if(max_val<it->second){
			max_val = it->second;
			max_ind = it->first;
		}
	}
	return max_ind;
}

arma::mat find_minority_class(arma::mat data){
	int c = data.n_cols;
	std::map<int,int> c_l;
	for(int i=0;i<data.n_rows;i++){
		if(!present(c_l,data(c-1)))
			c_l[data(c-1)] = 1;
		else	c_l[data(c-1)]++;
	}
	int m = find_max(c_l);
	arma::mat samp_frm = fill_zero(c_l[m],c);
	for(int i=0;i<data.n_rows;i++){
		if(data(i,c-1)==m){
			for(int j=0;j<c;j++)
				samp_frm(i,j) = data(i,j);
		}
	}
	return samp_frm;
}

int find_max_ind(int i, int* index, arma::mat d_mat,int k){
	double max_val = 0.0;
	int max_ind = 0;
	for(int j=0;j<k;j++){
		if(d_mat(i,index[j])>max_val){
			max_val = d_mat(i,index[j]);
			max_ind = j;
		}
	}
	return max_ind; 
}

std::vec find_nearest_neighbor(int i, arma::mat d_mat, int k, int *index){
	int flag = 0;
	int max_ind;
	int r = d_mat.n_rows;
	for(int j=0;j<r;j++){
		if(j<k)
			index[j] = j;
			flag = 1;
		else{
			if(flag==1){
				max_ind = find_max_ind(i,index,d_mat,k);
				flag = 0;
			}
			else{
				if(d_mat(i,j)<d_mat(i,index[max_ind])){
					index[max_ind] = j;
					max_ind = find_max_ind(i,index,d_mat,k)
				}
			}
		}
	}
	return index;
}

arma::mat Smote(arma::mat m_cl,int N, int c, int n_n){
	N = N/100;
	int cnt = 0;
	arma::mat distance_mat = pairwise_distance(m_cl);
	arma::mat o_s(m_cl.n_rows*N,m_cl.n_cols);
	int n_index[n_n];
	for(int i=0;i<m_cl.n_rows;i++){
		find_nearest_neighbor(i,distance_mat,n_n,n_index);
		for(int j=0;j<N;j++){
			int w = n_index[rand()%n_n];
			for(int k=0;k<m_cl.n_cols-1;k++){
				double diff = m_cl(w,k) - m_cl(i,k);
				double gap = (double)(rand())/RAND_MAX;
				o_s(cnt,k) = m_cl(i,k) + gap*diff;
				cnt++; 
			}
		}
	}
	return o_s;
}

void write_to_file(arma::mat data, arma::mat o_s){
	arma::mat C = join_rows(data,o_s);
	std::ofstream out;
	out.open("over_sampled_fetaure_matrix");
	for(int i=0;i<C.n_rows;i++){
		for(int j=0;j<C.n_cols;j++)
			out << C(i,j) << " ";
		out << "\n";
	}
	out.close();	
}

int main(int argc, char *argv[]){
	arma::mat data = read_csv("test_data");
	arma m_cl = find_minority_class(data);
	int N = 200;
	int n_n = 5;
	arma::mat oversample = Smote(m_cl,N,c,n_n);
	write_to_file(data,oversample);
	return 0;
}

