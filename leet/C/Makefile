CC = gcc
CFLAGS = -I./test
SRC = $(wildcard *.c)
TEST_SRC = test/unity.c test/test_leet.c
OBJ = $(SRC:.c=.o)
TEST_OBJ = $(TEST_SRC:.c=.o)

all: $(OBJ)

test: $(OBJ) $(TEST_OBJ)
	$(CC) -o test_runner $(OBJ) $(TEST_OBJ)

clean:
	rm -f $(OBJ) test_runner $(TEST_OBJ)
