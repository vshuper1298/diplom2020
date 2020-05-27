CC=g++
FLAGS= -Wall -std=c++11 -Isrc/ 
SRC=src/TCPServer.cpp src/server.cpp
OUT=-o server -fPIC -shared -lpthread

all:
	$(CC) $(FLAGS) $(SRC) $(OUT)
	$(CC) -c $(SRC)
	ar rvs libSimpleNetwork.a *.o

clean:
	rm *.o server libSimpleNetwork.*
