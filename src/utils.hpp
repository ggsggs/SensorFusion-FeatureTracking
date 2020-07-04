#ifndef utils
#define utils

#include <iostream>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void visKeypoints(cv::Mat img, std::vector<cv::KeyPoint> keypoints,
                  std::string windowName);

void nms(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double maxOverlap,
         int apertureSize, int minResponse);

#endif