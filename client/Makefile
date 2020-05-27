CC=g++
FLAGS= -Wall -std=c++11 -Isrc/ 
SRC=src/TCPClient.cpp src/client.cpp
OUT=-o client -fPIC -shared -lpthread

all:
	$(CC) $(FLAGS) $(SRC) $(OUT)
	$(CC) -c $(SRC)
	ar rvs libSimpleNetwork.a *.o

clean:
	rm *.o client libSimpleNetwork.*
