#ifndef READ_CPP
#define READ_CPP

#include <stdlib.h>
#include "Read.h"

Read::Read(int col) { column = col; }
Read::~Read() {}

void Read::split(char** arr, char* str, char* del) {
	char *s = NULL; 
 	s = strtok(str, del);
 	while (s != NULL) {
  		*arr++ = s;
  		s = strtok(NULL, del);
 	}
}

char* Read::toChar(string str) {
	char* c = new char[str.size()];
	for (int i = 0; i < str.size(); i++)
		c[i] = str[i];
	return c;
}

double* Read::translate(string line) {
	char **result = new char*[column];
	char del[2] = ",";

	split(result, toChar(line), del);

	double* data = new double[column];
	for (int i = 0; i < column; i++)
		data[i] = atof(result[i]);
	return data;
}

#endif