CC=g++

CFLAGS=-c

INCDIR=-I/usr/local/include/opencv4

LIBDIR=-L/usr/local/lib

LIBFILE=-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

CST11=-std=c++11

all: main

main: movement_detection.o
	$(CC) movement_detection.o $(LIBDIR) $(LIBFILE) $(CST11) -o main

movement_detection.o: movement_detection.cpp
	$(CC) $(CFLAGS) $(INCDIR) $(CST11) movement_detection.cpp

clean:
	rm -f *o main
