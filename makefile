CC = g++
objects = stat.o common.o 

naive_bayes : $(objects)
		 $(CC) neural_network.cpp -o neural_net $(objects) -O1 -larmadillo 
stat.o : stat.h stat.cpp 
	$(CC) -c stat.cpp

common.o : common.h common.cpp
	$(CC) -c common.cpp 
		 
clean : 
	rm $(objects) 
