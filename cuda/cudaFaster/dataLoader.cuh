typedef struct{
	matrix matrix;
	int truth;
} data;

void displayChar(unsigned char input);
void dataLoader(data** dataPointer, char* images, char* labels, int dataLength);
void fileDataViewer(char* inputFile);
void structDataViewer(data* dataPointer);

