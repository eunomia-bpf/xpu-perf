# Makefile for CPU minimal test program

CC = gcc
CFLAGS = -Wall -O2
TARGET = test_malloc

.PHONY: all clean

all: $(TARGET)

$(TARGET): test_malloc.c
	$(CC) $(CFLAGS) -o $(TARGET) test_malloc.c

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Running test_malloc..."
	./$(TARGET)
