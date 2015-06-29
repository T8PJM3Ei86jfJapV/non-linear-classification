#include <iostream>
#include "BPL.cpp"
#include "Read.cpp"

using namespace std;

int main() {
  
  BPL bpl("train.csv", "test.csv");
  bpl.read_TrainValues();
  bpl.read_TestValues();

  cout << "training ...." << endl;
  bpl.train();
  
  cout << "testing...." << endl;
  bpl.test();

  return 0;
}
