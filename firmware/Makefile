TARGET = main

CFLAGS = -Ofast -Wall -fopenmp -std=c++11 -fpermissive `pkg-config --cflags opencv`
LDFLAGS = -Ofast -Wall -fopenmp -std=c++11 `pkg-config --libs opencv`
CC = g++

all: clean $(TARGET)

$(TARGET): $(TARGET).o VideoMaterialDetector.o conv.o dense.o vgg9.o
	$(CC) $(LDFLAGS) -o $@ $^ `pkg-config --libs opencv` -lpthread

%.o: %.cpp
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) *.a *.o *~