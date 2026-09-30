#ifndef __DENSE_H__
#define __DENSE_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>
#include <cmath>

// Macro dinh nghia so lop Dense (khop vs file dense.cpp)
#define NUM_DENSE_LAYER 2

// Khai bao cac bien ngoai vi de dung chung giua cac file
extern float ***wd;
extern float **bd;
extern int numthreads;
extern int dshape[NUM_DENSE_LAYER][2];
extern const char weight_dense[NUM_DENSE_LAYER][50];
extern const char bias_dense[NUM_DENSE_LAYER][50];

// Khai bao cac khoi nho cho lop Dense Flatten
extern int mem_block_dense_shape;
extern float *mem_block1_dense;
extern float *mem_block2_dense;

// Cac ham thao tac bo nho
void reset_mem_block_dense(float *mem);
void init_memory();
void free_memory();
void read_weights();

// Cac ham tinh toan mang No-ron
void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu);
void dense(float *in, float **weights, float *out, int sh_in, int sh_out);
void softmax(float *out, int sh_out);

// Ham ra quyet dinh cuoi cung
int get_VGG9_predict(float *base_flatten, float *acc);

#endif