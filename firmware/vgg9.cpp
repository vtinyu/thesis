#include "vgg9.h"

int vgg9_predict(cv::Mat &frame, unsigned long *ddr_mem, unsigned long *low_cnn_addr, float *flatten_base, float *acc)
{
    int x, y, k = 0;
    unsigned long addr = 0;
    
  // Vong lap tien xu ly, them vien (padding) va kep cha 16-bit vao DDR memory
    for(y = 0; y < frame.rows + 2; y += 2)
    {
        for(x = 0; x < frame.cols + 2; x++)
        {
            if(x == 0)
            {
                // Vien trai: don 3 gia tri 0
                ddr_mem[addr++] = 0;
                ddr_mem[addr++] = 0;
                ddr_mem[addr++] = 0;
            }
            else if(x == (frame.cols + 1))
            {
                // Vien phai: don 45 so 0 de can chinh theo chuan Avalon Bus (240)
                for(int i = 0; i < 45; i++)
                    ddr_mem[addr++] = 0;
            }
            else
            {
                if(y == 0)
                {
                    int color = (int)frame.at<uchar>(y, x - 1);
                    int a = color - 128; 
                    
                    // Nhan ban 3 lan cho 3 kenh dau vao cua IP
                    ddr_mem[addr++] = a & 0xffff;
                    ddr_mem[addr++] = a & 0xffff;
                    ddr_mem[addr++] = a & 0xffff;
                }
                else if(y == frame.rows)
                {
                    int color = (int)frame.at<uchar>(y - 1, x - 1);
                    int a = color - 128;
                    
                    // Kep cha 16-bit cao va nhan ban 3 kenh
                    ddr_mem[addr++] = a << 16;
                    ddr_mem[addr++] = a << 16;
                    ddr_mem[addr++] = a << 16;
                }
                else
                {
                    // Ghep cap pixel tren va duoi vao chung 1 thanh ghi 32-bit (Bit-packing)
                    int color1 = (int)frame.at<uchar>(y - 1, x - 1);
                    int color2 = (int)frame.at<uchar>(y, x - 1);
                    int a = color1 - 128;
                    int b = color2 - 128;
                    
                    // Day du lieu kep cha vao 3 kenh lien tiep
                    ddr_mem[addr++] = a << 16 | (b & 0x0000ffff);
                    ddr_mem[addr++] = a << 16 | (b & 0x0000ffff);
                    ddr_mem[addr++] = a << 16 | (b & 0x0000ffff);
                }
            }
        }
    }
    
    // 1. Kich hoat khoi Tich chap tren phan cung FPGA
    vgg9_convolution(low_cnn_addr, ddr_mem, flatten_base);
    
    // 2. Chay cac lop Dense tren CPU HPS de ra quyet dinh cuoi cung
    k = get_VGG9_predict(flatten_base, acc);
    
    return k;
}