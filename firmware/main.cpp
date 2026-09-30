#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "VideoMaterialDetector.h" 
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "hwlib.h"
#include "socal/socal.h"
#include "socal/hps.h"
#include "socal/alt_gpio.h"
#include "hps_0.h"
#include <omp.h>
#include <sys/time.h>
#include <time.h>

#include "vgg9.h" 
#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#define HW_REGS_BASE ( ALT_STM_OFST )
#define HW_REGS_SPAN ( 0x04000000 )
#define HW_REGS_MASK ( HW_REGS_SPAN - 1 )

using namespace std;

// Ham dem thoi gian thuc thi (Microsecond)
static long get_tick_count(void)
{
    int k;
    struct timespec now;
    k = clock_gettime(CLOCK_MONOTONIC, &now);
    return now.tv_sec*1000000 + now.tv_nsec/1000;
}

const cv::String WINDOW_NAME("Surface Defect Detection - VGG9");

int main(int argc, char** argv)
{
    // Cap phat con tro giao tiep phan cung (Avalon Bus)
    void *virtual_base;
    void *h2p_low_cnn_addr;
    unsigned long *mem_addr;
    unsigned long *mem;
    
    // Cau hinh mang No-ron
    float *flatten_base;
    float acc;
    int fd;
    string s;
    stringstream stream;
    
    // Cap phat mang Flatten cung (128 * 4 * 4 = 2048)
    flatten_base = new float [128*4*4];
    if(flatten_base == NULL)
    {
        printf("ERROR: Cannot allocate flatten memory\n");
        return 0;
    }
    
    // === MAPPING MEMORY CHO HPS-FPGA ===
    if( ( fd = open( "/dev/mem", ( O_RDWR | O_SYNC ) ) ) == -1 ) {
        printf( "ERROR: could not open \"/dev/mem\"...\n" );
        return( 1 );
    }
    
    virtual_base = mmap( NULL, HW_REGS_SPAN, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, HW_REGS_BASE );
    if( virtual_base == MAP_FAILED ) {
        printf( "ERROR: virtual_base mmap() failed...\n" );
        close( fd );
        return( 1 );
    }
    
    h2p_low_cnn_addr = virtual_base + ( ( unsigned long )( ALT_LWFPGASLVS_OFST + 0x00 ) & ( unsigned long)( HW_REGS_MASK ) );
    
    mem_addr = ( unsigned long *)mmap( NULL, 0x8000000, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, 0x30000000 );    
    if( mem_addr == MAP_FAILED ) {
        printf( "ERROR: mem_addr mmap() failed...\n" );
        close( fd );
        return( 1 );
    }
    
    mem = ( unsigned long *)mmap( NULL, 0x1000, ( PROT_READ | PROT_WRITE ), MAP_SHARED, fd, 0xffc25000 );    
    if( mem == MAP_FAILED ) {
        printf( "ERROR: mem mmap() failed...\n" );
        close( fd );
        return( 1 );
    }
    
    // Reset/Khoi tao phan cung
    mem[0x80>>2] = 0x3fff;
    printf("Initialize memory map done\n");
    
    init_mem_conv(mem_addr);
    printf("Initialize convolution memory done\n");
    
    init_memory();
    printf("Initialize memory dense done\n");
    
    read_weights();
    printf("Read weight dense done\n");
    
    // === CAU HINH CAMERA ===
    cv::Size target_size(64,64); // VGG9 nhan anh goc 64x64
    cv::VideoCapture camera(0);
    camera.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    
    if (!camera.isOpened()) {
        fprintf(stderr, "ERROR: Getting camera...\n");
        exit(1);
    }
    
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_KEEPRATIO | cv::WINDOW_AUTOSIZE);
    
    // Khoi tao Object Detector (Da an logic cat ROI 480x480 ben trong)
    VideoMaterialDetector detector(camera);
    cv::Mat frame;
    cv::Mat greyMat;
    double fps = 0, time_per_frame;

    // === VONG LAP NHAN DIEN CHINH ===
    while (true)
    {
        int start = cv::getCPUTickCount();
        detector >> frame; // Lay frame 640x480 tu camera
        
        // Gia đinh ham isMaterialFound() luon true
        if (detector.isMaterialFound())
        {
            // Lay khung toa do da duoc fix cung (480x480) tu Class 
            cv::Rect myROI = detector.getROI(); 
            
            // Ve 1 khung hinh chu nhat Xanh quanh vat lieu tren man hinh
            cv::rectangle(frame, myROI, cv::Scalar(0, 255, 0), 2);
            
            // Cat anh, chuyen sang xam va nen ve 64x64
            greyMat = frame(myROI);
            cv::cvtColor(greyMat, greyMat, CV_BGR2GRAY);
            cv::resize(greyMat, greyMat, target_size);
            
            // Goi ham AI du doan tren FPGA
            int result = vgg9_predict(greyMat, mem_addr, (unsigned long *)h2p_low_cnn_addr, flatten_base, &acc);
            
            // Tinh toan FPS
            int end = cv::getCPUTickCount();
            time_per_frame = (end - start) / cv::getTickFrequency();
            fps = (15 * fps + (1 / time_per_frame)) / 16;
            
            // In nhan loi theo chuan NEU Surface Defect Database
            switch (result)
            {
                case 0: s = "Crack"; break;
                case 1: s = "Hole"; break;
                case 2: s = "Normal"; break;
                case 3: s = "Rust"; break;
                case 4: s = "Scratch"; break;
                default: s = "Unknown"; break;
            }
            
            cout << "Defect: " << s << " | Accuracy: " << acc << " | FPS: " << fps << endl;
            
            // Hien thi ket qua len man hinh camera
            cv::putText(frame, s, cv::Point2f(myROI.x + 10, myROI.y + 40), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 0, 255), 2);
            
            // In do tin cay (Accuracy) % len man hinh
            stringstream acc_stream;
            acc_stream << fixed << setprecision(2) << acc * 100 << "%";
            cv::putText(frame, acc_stream.str(), cv::Point2f(myROI.x + 10, myROI.y + 80), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 0, 0), 2);
        }
    
        cv::imshow(WINDOW_NAME, frame);
        // Nhan ESC de thoat
        if (cv::waitKey(25) == 27) break;
    }
    
    // === DON DEP BO NHO ===
    free_memory();
    delete[] flatten_base;
    
    if( munmap( virtual_base, HW_REGS_SPAN ) != 0 ) printf( "ERROR: munmap virtual_base failed...\n" );
    if( munmap( mem_addr, 0x8000000 ) != 0 ) printf( "ERROR: munmap mem_addr failed...\n" );
    if( munmap( mem, 0x1000 ) != 0 ) printf( "ERROR: munmap mem failed...\n" );
    
    close( fd );
    return 0;
}