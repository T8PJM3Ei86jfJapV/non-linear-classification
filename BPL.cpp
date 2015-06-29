#include "BPL.h"
#include "Read.cpp"

using namespace std;

// Initialize learning_examples, testing_examples from file.
// Initialize weight randomly.
BPL::BPL(const char* learning_dataset, const char* testing_dataset)
    : train_filename(learning_dataset), test_filename(testing_dataset) {

    srand(time(NULL));
    double temp;
    for (int i = 0; i < HIDDEN; i++) {
		for (int j = 0; j < INPUT_NUM; j++) {
	    	temp = rand() % 100;
	    	if (((int)temp % 2) == 0) {
				hidden_weight[i][j] = temp / 1000;
	    	} else {
				hidden_weight[i][j] = -(temp / 1000);
	    	}

	    	pre_hidden_weight[i][j] = 0;
		}
    }

    for (int i = 0; i < OUT_NUM; i++) {
		for (int j = 0; j < HIDDEN; j++) {
	    	temp = rand() % 100;
	    	if (((int)temp % 2) == 0) {
				output_weight[i][j] = temp / 1000;
	    	} else {
				output_weight[i][j] = -(temp / 1000);
	    	}

	    	pre_output_weight[i][j] = 0;
		}
    }

    // initialize target
    for (int i = 0; i < OUT_NUM; ++i) {
		for (int j = 0; j < OUT_NUM; ++j) {
	    	if(i == j)
				target[i][j] = 1;
	    	else
				target[i][j] = 0;
		}
    }
}

// Release memory.
BPL::~BPL() {
}

// Get output.
void BPL:: FeedForward(double *examples) {
    
    double sum = 0.0;
    
    // hiden layer
    for (int i = 0; i < HIDDEN ; ++i) {
		sum = 0.0;
		for (int k = 0; k < INPUT_NUM; ++k) {
	    	sum += hidden_weight[i][k] * examples[k];
		}
		output[0][i] = sigmoid(sum);
    }

    // output layer
    for (int i = 0; i < OUT_NUM; ++i) {
		sum = 0.0;
		for (int j = 0; j < HIDDEN; ++j) {
	    	sum += output_weight[i][j] * output[0][j];
		}
		output[1][i] = sigmoid(sum);
    }
}

// Use output to adjust weight.
void BPL::BackPropogation(double *examples, int index_row) {

    double sum = 0.0;
    // output layer delta
    for (int i = 0; i < OUT_NUM; ++i) {
		delta[1][i] = (target[train_label[index_row]][i] - output[1][i])
						* output[1][i] * (1 - output[1][i]);
    }

    // hiden layer delta
    for (int j = 0; j < HIDDEN; ++j) {
		sum = 0.0;
		for (int i = 0; i < OUT_NUM; ++i) {
	    	sum += output_weight[i][j] * delta[1][i];
		}
		delta[0][j] = sum * output[0][j] * (1 - output[0][j]);
    }
    // output layer update weight
    for (int i = 0; i < OUT_NUM; ++i) {
		for (int j = 0; j < HIDDEN; ++j) {
		    pre_output_weight[i][j] = MOMENTUM * pre_output_weight[i][j] 
										+ ALPHA * output[0][j] * delta[1][i];
	    	output_weight[i][j] += pre_output_weight[i][j];
		}
    }
    
    // hiden layer update weight
    for (int j = 0; j < HIDDEN ; ++j) {
		for (int k = 0; k < INPUT_NUM; ++k) {
	    	pre_hidden_weight[j][k] = MOMENTUM * pre_hidden_weight[j][k]
										+ ALPHA * examples[k] * delta[0][j];
	    	hidden_weight[j][k] += pre_hidden_weight[j][k];
		}
    }
    
}

// Active function.
double BPL::sigmoid(double x) {
    return (1 / (1 + std::exp(-x)));
}

// Calculate minimal square error
double BPL::calErr(double *examples, int index_row) {
    double sum = 0.0;
    for (int i = 0; i < OUT_NUM; ++i) {
		sum += pow((target[train_label[index_row]][i] - output[1][i]), 2);
    }
    return sum / 2;
}

// Calculate wheter the result is correct
int BPL::calCor() {
    int max = 0;
    for (int j = 1; j < OUT_NUM; j++) {
		if (output[1][j] > output[1][max])
	    	max = j;
    }
    return max;
}

// restore weight
void BPL::restoreWeight() {

    char flag;
    cout << "restore weight from last time(y or n): ";
    cin >> flag;

    if (flag != 'y')
	return;

    int i = 0;
    char buffer[9600];
    char *p;
    const char *delim = ",\r";
    fstream i_h, h_o;

    i_h.open("hidden.txt",ios::out);
    h_o.open("output.txt", ios::out);
    
    if (i_h == NULL || h_o == NULL)
		throw exception();

    while (!i_h.eof() && i < HIDDEN) {
		int j = 0;

		i_h.getline(buffer, 9600, '\n');
		p = strtok(buffer, delim);

		hidden_weight[i][j] = atof(p);
		while (++j < INPUT_NUM && (p=strtok(NULL, delim))) {
	    	hidden_weight[i][j] = atof(p);
		}
		i++;
    }
    
    while (!h_o.eof() && i < OUT_NUM) {
		int j = 0;

		h_o.getline(buffer, 9600, '\n');
		p = strtok(buffer, delim);

		output_weight[i][j] = atof(p);
		while (++j < HIDDEN && (p=strtok(NULL, delim))) {
	    	output_weight[i][j] = atof(p);
		}
		i++;
    }

    i_h.close();
    h_o.close();
}

// save the train weight
void BPL::saveWeight() {

    fstream i_h, h_o;
    i_h.open("hidden.txt",ios::out);
    h_o.open("output.txt", ios::out);

    for (int i = 0; i < HIDDEN; i++) {
		for (int j = 0; j < INPUT_NUM; j++) {
	    	i_h << hidden_weight[i][j] << ",";
		}
		i_h << endl;
    }

    for (int i = 0; i < OUT_NUM; i++) {
		for (int j = 0; j < HIDDEN; j++) {
	    	h_o << output_weight[i][j] << ",";
		}
		h_o << endl;
    }

    i_h.close();
    h_o.close();
}
// Training the neural net
// max training times 10000
void BPL::train() {
    int counter = 0;
    int iter = 0;
    double preErr = 0.01;
    double Err;
    
    ALPHA = ALPHA0;
    while (iter < ITER_NUM) {
		cout << "iter " << iter << " " << counter << endl;
		counter = 0;
		for (int i = 0; i < LEARNING_NUM; i++) {
	    	FeedForward(learning_examples[i]);
	    	if (calCor() == train_label[i])
				counter++;
	    	BackPropogation(learning_examples[i], i);
		}
		if (counter > LEARNING_NUM - 100)
	    	break;

		ALPHA = BETA * ALPHA;
		iter++;
	
    }
    cout << "total iter " << iter << endl;
 }

// Testing the result
void BPL::test() {
    fstream result_file;
    result_file.open("result.csv", ios::out);
    result_file << "Id,label" << endl;
        
    int max = 0;
    for (int i = 0; i < TESTING_NUM; i++) {

        FeedForward(test_values[i]);
		max = calCor();

		result_file << i << "," << max+1 << endl;
    }
}

void BPL::read_TrainValues() {
	ifstream infile("train.csv");
	string line;
	getline(infile, line);
	int index1 = 0, index2 = 0;
	while (infile.good()) {
		getline(infile, line);
		Read r(TRAIN_COLUMN);
		double *data = new double[TRAIN_COLUMN];
		data = r.translate(line);
		train_label[index2++] = (int)data[TRAIN_COLUMN-1] - 1;
		for (int i = 1, j = 0; j < DIMENSION; i++, j++) 
			learning_examples[index1][j] = data[i];
		index1++;
		delete data;
	}
	infile.close();
}

void BPL::read_TestValues() {
	ifstream infile("test.csv");
	string line;
	getline(infile, line);
	int index = 0;
	while (infile.good()) {
		getline(infile, line);
		Read r(TEST_COLUMN);
		double *data = new double[TEST_COLUMN];
		data = r.translate(line);
		for (int i = 1, j = 0; j < DIMENSION; i++, j++) 
			test_values[index][j] = data[i];
		index++;
		delete data;
	}
	infile.close();
}