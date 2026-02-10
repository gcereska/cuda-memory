CC = gcc
CFLAGS = -Wall -Wextra -std=c11


COMMON = cpuPoolMalloc.c

TESTS = $(wildcard test*.c)

TARGETS = $(TESTS:.c=)


all: $(TARGETS)

$(TARGETS): %: %.c $(COMMON)
	$(CC) $(CFLAGS) -o $@ $^ 

clean:
	rm -f $(TARGETS) *.o

