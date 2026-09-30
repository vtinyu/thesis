#include "conv.h" 
#define NUM_LAYER 7

// Cau hinh mang VGG9: [So Kernel dau ra, Số Kênh dau vao, Kich thuoc, Pooling]
int cshape[NUM_LAYER][4] = { 
	{ 32, 3, 64, 0 },
	{ 32, 32, 64, 1 },
	{ 32, 32, 32, 0 },
	{ 32, 32, 32, 1 },
	{ 64, 32, 16, 0 },
	{ 64, 64, 16, 1 },
	{ 128, 64, 8, 1 }
};

char weight_name[NUM_LAYER][50] = {
	"weights/conv1_W.txt",
	"weights/conv2_W.txt",
	"weights/conv3_W.txt",
	"weights/conv4_W.txt",
	"weights/conv5_W.txt",
	"weights/conv6_W.txt",
	"weights/conv7_W.txt"
};
	
char bias_name[NUM_LAYER][50] = {
	"weights/conv1_b.txt",
	"weights/conv2_b.txt",
	"weights/conv3_b.txt",
	"weights/conv4_b.txt",
	"weights/conv5_b.txt",
	"weights/conv6_b.txt",
	"weights/conv7_b.txt"
};

void write_weight(FILE* file_wei, int nchannel, int nkernel, unsigned long *ddr_addr)
{
	int i;
	int wdata;
	int tmp0, tmp1;
	unsigned long k;
	
    // Cong thuc tinh so lan ghi dua tren kien truc Bus 256-bit cua phan cung
	k = 9 * 16 * nchannel * (nkernel / 32);
	
	for(i = 0; i < k; i++)
	{
		fscanf(file_wei, "%d %d\n", &tmp0, &tmp1);
		wdata = (tmp1 << 16) + tmp0;
		ddr_addr[i] = wdata;
	}
}

void write_bias(FILE *file_bs, int nkernel, unsigned long *ddr_addr)
{
	int i;
	int wdata;
	int tmp0, tmp1;
	
	for(i = 0; i < nkernel / 2; i++)
	{
		fscanf(file_bs, "%d %d\n", &tmp0, &tmp1);
		wdata = (tmp1 << 16) + tmp0;
		ddr_addr[i] = wdata;
	}
}

void write_start(unsigned long *low_cnn_addr)
{
	low_cnn_addr[0] = 1;
}

void write_layer_status(int conv, int pooling, int insize, int nkernel, int nchannel, unsigned long *low_cnn_addr)
{
	int wdata = (conv << 31) + (pooling << 30) + (insize << 20) + (nkernel << 10) + nchannel;
	low_cnn_addr[2] = wdata;
}

// Ham kich hoat FPGA chay 1 lop Tich chap
void implement_a_layer(int pooling, int insize, int nkernel, int nchannel, unsigned long *low_cnn_addr, unsigned long *ddr_addr, int layer, uint32_t rbase, uint32_t wbase)
{
	int a = 0;
	
	write_layer_status(1, pooling, insize, nkernel, nchannel, low_cnn_addr);
	
	low_cnn_addr[4] = rbase;
	low_cnn_addr[5] = wbase;
	usleep(1);
	
    write_start(low_cnn_addr);
	
  // Doi co bao hieu xong (done) tu FPGA
	while (!a)
	{
		a = low_cnn_addr[3];
		usleep(1);
	}
}

// Khoi tao bo nho va day Trong so/Bias xuong SDRAM
void init_mem_conv(unsigned long *ddr_addr)
{
	FILE *w[NUM_LAYER];
	FILE *b[NUM_LAYER];
	
	for(int i = 0; i < NUM_LAYER; i++)
	{
		w[i] = fopen(weight_name[i], "r");
		if(w[i] == NULL)
		{
			printf("ERROR: Cannot open weight file %d\n", i);
			return;
		}
		
		b[i] = fopen(bias_name[i], "r");
		if(b[i] == NULL)
		{
			printf("ERROR: Cannot open bias file %d\n", i);
			return;
		}
	}
	
	for(int i = 0; i < NUM_LAYER; i++)
	{
		write_weight(w[i], cshape[i][1], cshape[i][0], ddr_addr + i * 0x140000 + 0x200000);
		printf("Loaded Weights for Layer %d\n", i);
        
		write_bias(b[i], cshape[i][0], ddr_addr + i * 0x100 + 0x1380000);
		printf("Loaded Bias for Layer %d\n", i);
        
        // Dong file sau khi doc xong
        fclose(w[i]);
        fclose(b[i]);
	}
}

// Dao nguoc luong tu hoa va ep phang mang (Flatten)
void flatten(int16_t *base_in, float *base_out)
{
	int i;
	for (i = 0; i < 2048; i++)
	{
		base_out[i] = (float)base_in[i] / 128.0f;
	}
}

// Vong lap dieu phoi du lieu Ping-Pong qua 7 lop Conv
void vgg9_convolution(unsigned long *low_cnn_addr, unsigned long *ddr_addr, float *flatten_base)
{
	int i;
	unsigned long rbase = 0x0, wbase = 0x5000000;
	unsigned long tmp;
	
	for(i = 0; i < NUM_LAYER; i++)
	{
		implement_a_layer(cshape[i][3], cshape[i][2], cshape[i][0], cshape[i][1], low_cnn_addr, ddr_addr, i, rbase, wbase);
		
        // Dao dia chi doc/ghi (Ping-Pong Buffer)
		tmp = wbase;
		wbase = rbase;
		rbase = tmp;
	}
	
	flatten((int16_t*)(ddr_addr + rbase / 4), flatten_base);
}