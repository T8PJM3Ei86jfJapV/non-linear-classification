#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <exception>
#include <ctime>
#include <vector>
#include <iomanip>

#define ALPHA0 0.02// learning rate.
#define BETA 0.96
#define ZETA 1.01
#define INPUT_NUM 617
#define HIDDEN 251 // number of hidden perceptrons.
#define OUT_NUM 26
#define LAYER_NUM 3
#define MOMENTUM 0.1
#define ITER_NUM 200

#define LEARNING_NUM 6238
#define TRAIN_COLUMN 619
#define TESTING_NUM 1559
#define TEST_COLUMN 618
#define DIMENSION 617

double learning_examples[LEARNING_NUM][DIMENSION];
double test_values[TESTING_NUM][DIMENSION];
int train_label[LEARNING_NUM] = {-1};



class BPL {
public:
    // Initialize learning_examples, testing_examples from file.
    // Initialize weight randomly.
    BPL(const char* learning_dataset, const char* testing_dataset);

    // Release memory.
    ~BPL();

    // Training 
    void train();

    // Testing
    void test();

    // Get output.
    void FeedForward(double *examples);

    // Use output to adjust weight.
    void BackPropogation(double *examples, int index_row);

    void read_TrainValues();
    void read_TestValues();
  
private:
    // Active function.
    double sigmoid(double x);
    double calErr(double *examples, int index_row);
    int calCor();
    void saveWeight();
    void restoreWeight();

    const char *train_filename;
    const char *test_filename;
    
    double target[OUT_NUM][OUT_NUM];
    double delta[LAYER_NUM-1][HIDDEN]; // for back propogation.
    double hidden_weight[HIDDEN][INPUT_NUM];
    double output_weight[OUT_NUM][HIDDEN];
    double pre_hidden_weight[HIDDEN][INPUT_NUM];
    double pre_output_weight[OUT_NUM][HIDDEN];
    double output[LAYER_NUM-1][HIDDEN];
    double ALPHA;

};
