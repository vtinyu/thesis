#include "VideoMaterialDetector.h"
#include <iostream>

VideoMaterialDetector::VideoMaterialDetector(cv::VideoCapture &videoCapture)
{
    setVideoCapture(videoCapture);
    m_foundMaterial = true; // Luon san sang nhan dien khi camera mo
}

void VideoMaterialDetector::setVideoCapture(cv::VideoCapture &videoCapture)
{
    m_videoCapture = &videoCapture;
}

cv::VideoCapture *VideoMaterialDetector::videoCapture() const
{
    return m_videoCapture;
}

bool VideoMaterialDetector::isMaterialFound() const
{
    return m_foundMaterial;
}

// Tinh toan va tra ve khung ROI 480x480 o chinh giua
cv::Rect VideoMaterialDetector::getROI() const
{
    // Kich thuoc Crop
    int crop_width = 480;
    int crop_height = 480;
    
    // Gia dinh camera đang set ở 640x480, nhung van dung CT tong quat 
    // de phong ban doi do phan giai camera sau nay
    int frame_width = 640; 
    int frame_height = 480;
    
    // Neu co videoCapture, lay kich thuoc that cua camera
    if (m_videoCapture != NULL && m_videoCapture->isOpened()) {
        frame_width = (int)m_videoCapture->get(CV_CAP_PROP_FRAME_WIDTH);
        frame_height = (int)m_videoCapture->get(CV_CAP_PROP_FRAME_HEIGHT);
    }
    
    // Tinh toa do X, Y sao cho khung 480x480 nam ngay tam man hinh
    int start_x = (frame_width - crop_width) / 2;
    int start_y = (frame_height - crop_height) / 2;
    
    // Dam bao khong am neu camera nho hơn 480
    start_x = std::max(start_x, 0);
    start_y = std::max(start_y, 0);

    return cv::Rect(start_x, start_y, crop_width, crop_height);
}

// Toan tu lay frame moi tu camera
cv::Point VideoMaterialDetector::getFrameAndDetect(cv::Mat &frame)
{
    if (m_videoCapture != NULL && m_videoCapture->isOpened()) {
        *m_videoCapture >> frame;
        m_foundMaterial = !frame.empty();
    } else {
        m_foundMaterial = false;
    }
    
    // Tra ve toa do (0,0) vì khong can tracking tam di dong
    return cv::Point(0, 0); 
}

// Goi tat toan tu >>
cv::Point VideoMaterialDetector::operator>>(cv::Mat &frame)
{
    return this->getFrameAndDetect(frame);
}

VideoMaterialDetector::~VideoMaterialDetector()
{
   
}