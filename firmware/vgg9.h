#ifndef __VGG9_H__
#define __VGG9_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdint.h>
#include "conv.h"
#include "dense.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// Ham tong hop ket noi xu ly anh, chay mach FPGA va chay Dense CPU
int vgg9_predict(cv::Mat &frame, unsigned long *ddr_mem, unsigned long *low_cnn_addr, float *flatten_base, float *acc);

#endif