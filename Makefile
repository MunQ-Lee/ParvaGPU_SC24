TARGET=parva_sched
CC=gcc
CXX=g++

C_SRCS=$(addprefix src/, parva_sched.c configurator.c data_load.c allocator.c queue.c)
C_OBJS=$(C_SRCS:.c=.o)
OBJS=$(C_OBJS)

CFLAGS=-Iinc
LDFLAGS=-lm

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)

%.o: %.c inc
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -f $(OBJS) parva_sched
