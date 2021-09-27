CC=g++

CFLAGS=-c

INCDIR=-I/usr/local/include/opencv4

LIBDIR=-L/usr/local/lib

LIBFILE=-lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core

CST11=-std=c++11

all: main

main: AlphaCompositing.o
	$(CC) AlphaCompositing.o $(LIBDIR) $(LIBFILE) $(CST11) -o main

AlphaCompositing.o: AlphaCompositing.cpp
	$(CC) $(CFLAGS) $(INCDIR) $(CST11) AlphaCompositing.cpp

clean:
	rm -f *o main
