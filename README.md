# MNIST Classifier in C

## Introduction

This project is a simple digit classifier for the MNIST dataset implemented in C. The primary goal is to revisit and reinforce the understanding of the backpropagation algorithm by implementing a neural network from scratch. In doing so I had to make a c matrix library, which I used as an opportunity to make optimized multithreaded matrix algorithms, utilizing either openMP, pthreads, or CUDA.

### Key Objectives

- Relearn and implement the backpropagation algorithm for neural network training.
- Develop a modular and efficient C matrix library to support matrix operations.
- Lay the groundwork for potential future enhancements, such as multithreaded matrix algorithms.

I followed this textbook, http://neuralnetworksanddeeplearning.com,
which had very good information regarding the algorithms for the implementation. 

## Results
cc main.c -o net.exe -lm -g -Wall -pedantic  
./net.exe  
Starting epoch: 0  
77.98 % correct, 7798/10000  
.....  
Starting epoch: 9  
95.73 % correct, 9573/10000  
Press q to exit, press any character to see the next evaluation  


      ▓▩░░▓
      ██████▩▩▩▩▩▩▩▩░▓
      ▓▒▓▒░██████████░
            ▓ ▓▓▓▓ ██▒
                  ▓█▩
                  ██▓
                 ░██▓
                ▓██▓
                ░█▩
                ▩█▓
               ▒█▩
              ▓██▓
              ██░
             ▩██
             ██▓
            ██▒
           ░██▓
          ▓███▓
          ▒███
          ▒█▩

returned 7 with value of 0.99, true value is 7  

In the above output, I ran the program and got 95% accuracy on the mnist dataset.   

## Conclusion
The conclusion I gained from all of this is to use BLAS and python as they greatly simplify and expediate the process. However I am thankful I learned lots regarding backproagation, and multithreaded matrix multiplication algorithms.

