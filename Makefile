# Compiler flags
CFLAGS = -lm -g -Wall -pedantic

# Source file
SOURCE = main.c

# Output file
OUTPUT = net.exe

all: $(OUTPUT)

$(OUTPUT): $(SOURCE)
	clear
	$(CC) $(SOURCE) -o $(OUTPUT) $(CFLAGS)
	./net.exe
clean:
	rm -f $(OUTPUT)
