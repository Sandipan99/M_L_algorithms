CC = g++
objects = stat.o 

naive_bayes : $(objects)
		 $(CC) -o naive_bayes naive_bayes.cpp $(objects)

stat.o : stat.h stat.cpp 
	$(CC) -c stat.cpp 
		 
clean : 
	rm $(objects) 
