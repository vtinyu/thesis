#include "dense.h"

#define NUM_DENSE_LAYER 2

// Cau truc lop Dense: [Input, Output]
int dshape[NUM_DENSE_LAYER][2] = {
	{ 2048, 128 }, 
	{ 128, 5 }  
};

float ***wd;
float **bd;
int numthreads = 2; 

const char weight_dense[NUM_DENSE_LAYER][50] = {
	"weights/dense1_W.txt",
	"weights/dense2_W.txt"
};
						
const char bias_dense[NUM_DENSE_LAYER][50] = {
	"weights/dense1_b.txt",
	"weights/dense2_b.txt"
};
									
int mem_block_dense_shape = 128 * 4 * 4; // 2048
float *mem_block1_dense;
float *mem_block2_dense;

void reset_mem_block_dense(float *mem) {
	int i;
	for (i = 0; i < mem_block_dense_shape; i++) {
		mem[i] = 0.0;
	}
}

void init_memory() {
	int i, l;

	wd = new float**[NUM_DENSE_LAYER];
	bd = new float*[NUM_DENSE_LAYER];
    
	for (l = 0; l < NUM_DENSE_LAYER; l++) {
		wd[l] = new float*[dshape[l][1]];
		for (i = 0; i < dshape[l][1]; i++) {
			wd[l][i] = new float[dshape[l][0]];
		}
		bd[l] = new float[dshape[l][1]];
	}
	
	mem_block1_dense = new float[mem_block_dense_shape];
	mem_block2_dense = new float[mem_block_dense_shape];
}

void free_memory() {
	int i, l;

	for (l = 0; l < NUM_DENSE_LAYER; l++) {
		for (i = 0; i < dshape[l][1]; i++) {
			delete[] wd[l][i];
		}
		delete[] wd[l];
		delete[] bd[l];
	}
	delete[] wd;
	delete[] bd;
	
	delete[] mem_block1_dense;
	delete[] mem_block2_dense;
}

void read_weights() {
	float dval;
	int i, j, z;
	FILE *in_w[NUM_DENSE_LAYER];
	FILE *in_b[NUM_DENSE_LAYER];

	for(i = 0; i < NUM_DENSE_LAYER; i++) {
		in_w[i] = fopen(weight_dense[i], "r");
		if (in_w[i] == NULL) {
			printf("ERROR: File %s absent\n", weight_dense[i]);
			exit(1);
		}
	}
	
	for(i = 0; i < NUM_DENSE_LAYER; i++) {
		in_b[i] = fopen(bias_dense[i], "r");
		if (in_b[i] == NULL) {
			printf("ERROR: File %s absent\n", bias_dense[i]);
			exit(1);
		}
	}
	
	for (z = 0; z < NUM_DENSE_LAYER; z++) {
		for (i = 0; i < dshape[z][1]; i++) {
			for (j = 0; j < dshape[z][0]; j++) {
				fscanf(in_w[z], "%f\n", &dval);
				wd[z][i][j] = dval;
			}
		}
		for (i = 0; i < dshape[z][1]; i++) {
			fscanf(in_b[z], "%f\n", &dval);
			bd[z][i] = dval;
		}
	}
	
	for(i = 0; i < NUM_DENSE_LAYER; i++) {
		fclose(in_w[i]);
		fclose(in_b[i]);
	}
}

void add_bias_and_relu_flatten(float *out, float *bs, int size, int relu) {
	int i;
	for (i = 0; i < size; i++) {
		out[i] += bs[i];
		if (relu == 1) {
			if (out[i] < 0) out[i] = 0.0;
		}
	}
}

void dense(float *in, float **weights, float *out, int sh_in, int sh_out) {
	int i, j;
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (i = 0; i < sh_out; i++) {
		float sum = 0.0;
		for (j = 0; j < sh_in; j++) {
			sum += in[j] * weights[i][j];
		}
		out[i] = sum;
	}
}

// Ham Softmax
void softmax(float *out, int sh_out) {
	int i;
	float max_val, sum;
	max_val = out[0];
	for (i = 1; i < sh_out; i++) {
		if (out[i] > max_val)
			max_val = out[i];
	}
	sum = 0.0;
	for (i = 0; i < sh_out; i++) {
		out[i] = exp(out[i]);
		sum += out[i];
	}
	for (i = 0; i < sh_out; i++) {
		out[i] /= sum;
	}
}

int get_VGG9_predict(float *base_flatten, float *acc) {
	int i, j = 0;
	int level;
	float max;
	
	reset_mem_block_dense(mem_block1_dense);
	reset_mem_block_dense(mem_block2_dense);
	
	level = 0;
	dense(base_flatten, wd[level], mem_block1_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block1_dense, bd[level], dshape[level][1], 1);

	level = 1;
	dense(mem_block1_dense, wd[level], mem_block2_dense, dshape[level][0], dshape[level][1]);
	add_bias_and_relu_flatten(mem_block2_dense, bd[level], dshape[level][1], 0);
	
	softmax(mem_block2_dense, dshape[level][1]);
	
	max = mem_block2_dense[0];
	for(i = 1; i < dshape[level][1]; i++) {
		if(mem_block2_dense[i] > max) {
			max = mem_block2_dense[i];
			j = i;
		}
	}
	
	*acc = max; 
	return j;   
}