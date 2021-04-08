#include <malloc.h>
#include <cstdio>
#include <fstream>

#ifndef Header_H
#define Header_H

float* read_bin_data(const char *);
void export_data(const char*, float*, int&);
void show_gpu_array(double*,size_t);

#endif