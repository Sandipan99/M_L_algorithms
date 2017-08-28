CC = g++ -std=gnu++11
objects = stat.o common.o distribution.o

naive_bayes : $(objects)
		 $(CC) logistic_regression.cpp -o log_reg $(objects) -O1 -larmadillo 
stat.o : stat.h stat.cpp 
	$(CC) -c stat.cpp

common.o : common.h common.cpp
	$(CC) -c common.cpp 

distribution.o : distribution.h distribution.cpp
	$(CC) -c distribution.cpp
		 
clean : 
	rm $(objects) 
