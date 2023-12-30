#include "main.cuh"
#define Print 1
#define DontPrint 0
int main(){
	
	setlocale(LC_CTYPE, "");
	initMat();
		
//	matrix test1;
//	matrix test2;
//	matrix test3;
//	matrixAllocate(&test1,14,30);
//	matrixAllocate(&test2,30,10);
//	matrixAllocate(&test3,14,10);
//	matrixRandFill(&test2);
//	matrixRandFill(&test1);
//	matrixMult(&test1,&test2,&test3);
//	matrixPrint(&test3);

	data* trainingData;
	data* testingData;
	int trainingLength = 60000;
	int testingLength = 10000;
	dataLoader(&trainingData, "trainingData/train-images.idx3-ubyte", "trainingData/train-labels.idx1-ubyte", trainingLength);
	dataLoader(&testingData, "trainingData/t10k-images.idx3-ubyte", "trainingData/t10k-labels.idx1-ubyte", testingLength);
	//The above loades data in from the trainingData folder,
	network net;
	networkAllocate(&net,3, 784, 30, 10); //Initilizes a 3 layer net with sizes 784, 30, 100
	networkWeightsInit(&net); //Sets the weights and biases to gaussian distributed around 0
	networkSGD(&net, trainingData, trainingLength, testingData, testingLength, Print, 3, 50, 5, 5);
	evaluateSet(&net, testingData, testingLength, Print);
	evaluateSetManual(&net,testingData,testingLength);
	//Hyper parameter grid
//	for( int k = 1; k < 5; k++){

//	float hyperParameters[10][10];
//	int learningRates[] = {1,2,4,8,16,24,32,50,75,100};
//	int batchSize[] = {1,2,4,8,16,24,32,50,100,200};
//	for( int i = 0; i < 10; i++){
//		for(int j  = 0; j < 10; j++){
//			network net;
//			networkAllocate(&net,3, 784, 30, 10); //Initilizes a 3 layer net with sizes 784, 30, 100
//			networkWeightsInit(&net); //Sets the weights and biases to gaussian distributed around 0
//			networkSGD(&net, trainingData, trainingLength, testingData, testingLength, DontPrint, k, batchSize[j], learningRates[i]);
//	//The above does SGD on the training data, with it printing every epoch, and 10 epochs, and a batchsize of 10, with a learning rate of 5.
//			hyperParameters[i][j] = evaluateSet(&net, testingData, testingLength, DontPrint);
//			networkFree(&net);//These commands free the networks and arrays
//		}
//	
//	}
//	printf("Epochs: %d, \n", k);
//	printf("    |"); for(int i = 0; i < 10; i++){printf("%4d", batchSize[i]);};
//	printf("\n");
//	for(int i = 0; i < 10; i++){
//		printf("%4d|", learningRates[i]);
//		for (int j = 0; j < 10; j++){
///
//			displayChar((unsigned char)(hyperParameters[i][j] * 255.0f));
//			displayChar((unsigned char)(hyperParameters[i][j] * 255.0f));
//			displayChar((unsigned char)(hyperParameters[i][j] * 255.0f));
//			displayChar((unsigned char)(hyperParameters[i][j] * 255.0f));
//		}
//		printf("\n");
//	}

//	}



	dataFree(&trainingData, trainingLength);
	dataFree(&testingData, testingLength);

}
