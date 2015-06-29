#ifndef READ_H
#define READ_H

#include <cstring>
using namespace std;

class Read {
public:
	Read(int col);
	void split(char** arr, char* str, char* del);
	char* toChar(string str);
	double* translate(string line);
	~Read();
private:
	string line;
	int column;
};

#endif