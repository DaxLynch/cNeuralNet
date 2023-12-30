# Compiler flags
CFLAGS = -lm -g
LIBS=
# Source file
SOURCE = main.cu

# Output file
OUTPUT = net.exe

all: $(OUTPUT)

$(OUTPUT): $(SOURCE)
	clear
	nvcc $(SOURCE) -o $(OUTPUT) $(LIBS) $(CFLAGS)
clean:
	rm -f $(OUTPUT)
