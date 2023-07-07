CC = g++
CFLAGS = -std=c++11 -Wall -Iinclude

SRCDIR = src
INCDIR = include
BINDIR = bin

SOURCES = $(wildcard $(SRCDIR)/*.cpp)
HEADERS = $(wildcard $(INCDIR)/*.h)
OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)
EXECUTABLE = $(BINDIR)/program

.PHONY: all clean

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJECTS): $(BINDIR)/%.o : $(SRCDIR)/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(BINDIR)/*


