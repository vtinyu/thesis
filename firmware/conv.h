#ifndef __CONV_H__
#define __CONV_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>
#include <unistd.h>

#define NUM_LAYER 7

extern int cshape[NUM_LAYER][4];
extern char weight_name[NUM_LAYER][50];
extern char bias_name[NUM_LAYER][50];

// Khai bao cac ham dieu khien IP Core tich chap, void write_img (const char *file_img, int insize, int nchannel, unsigned long *ddr_addr);
void write_start(unsigned long *low_cnn_addr);
void implement_a_layer(int pooling, int insize, int nkernel, int nchannel, unsigned long *low_cnn_addr, unsigned long *ddr_addr, int layer, uint32_t rbase, uint32_t wbase);

// Khai bao cac ham doc/ghi va khoi tao bo nho
void write_bias(FILE *file_bs, int nkernel, unsigned long *ddr_addr);
void write_weight(FILE *file_wei, int nchannel, int nkernel, unsigned long *ddr_addr);
void init_mem_conv(unsigned long *ddr_addr);

// Ham Flatten de chuyen du lieu tu phan cung FPGA len phan mem CPU
void flatten(int16_t *base_in, float *base_out);

// Ham trung tam dieu phoi toan bo qua trinh Tich chap 7 lop
void vgg9_convolution(unsigned long *low_cnn_addr, unsigned long *ddr_addr, float *flatten_base);

#endif