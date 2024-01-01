# Compiler flags
CFLAGS = -lm -g -lcublas
LIBS=
# Source file
SOURCE = main.cu

# Output file
OUTPUT = prog

OBJECTS = main.o matrix.o dataLoader.o network.o

all: $(OUTPUT)

$(OUTPUT): $(OBJECTS)
	nvcc $(CFLAGS) -o $(OUTPUT) $(OBJECTS)

%.o: %.cu
	nvcc $(CFLAGS) -c $<

clean:
	rm -f $(OUTPUT)
