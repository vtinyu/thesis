#ifndef VIDEOMATERIALDETECTOR_H
#define VIDEOMATERIALDETECTOR_H

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

class VideoMaterialDetector
{
public:
    VideoMaterialDetector(cv::VideoCapture &videoCapture);
    ~VideoMaterialDetector();

    void setVideoCapture(cv::VideoCapture &videoCapture);
    cv::VideoCapture *videoCapture() const;
    
    bool isMaterialFound() const;
    cv::Rect getROI() const;
    
    cv::Point operator>>(cv::Mat &frame);

private:
    cv::Point getFrameAndDetect(cv::Mat &frame);
    
    cv::VideoCapture *m_videoCapture = NULL;
    bool m_foundMaterial = false;
};

#endif