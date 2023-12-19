typedef struct {
	int* sizes;
	matrix* weights;
	matrix* biases;
	int num_layers;
} network;
void networkAllocate(network* net, int len, ...);
void networkSizeAllocate(network* net, network* src);
void networkFree(network* net);
void update_mini_batch(network* net, int batch_size, int* batch_list, data* data_set, double eta);
void shuffle(int *array, size_t n);
void networkSGD(network* net, data* dataset, int dataLength, data* testSet, int testLength, int test, int epochs, int batch_size, double eta);
int setEvaluate(network* net, data* testData, int dataLength);
int evaluate(network* net, data* datum, int print);

