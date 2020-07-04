#include "utils.hpp"
#include <opencv2/features2d.hpp>

void visKeypoints(cv::Mat img, std::vector<cv::KeyPoint> keypoints,
                  std::string windowName) {
  cv::Mat visImage = img.clone();
  cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::namedWindow(windowName, 6);
  imshow(windowName, visImage);
  cv::waitKey(0);
}

void nms(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, double maxOverlap,
         int apertureSize, int minResponse) {
  // keypoint selection
  for (size_t j = 0; j < img.rows; j++) {
    for (size_t i = 0; i < img.cols; i++) {
      int response = (int)img.at<float>(j, i);
      if (response > minResponse) { // only store points above a threshold

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f(i, j);
        newKeyPoint.size = 2 * apertureSize;
        newKeyPoint.response = response;

        // perform non-maximum suppression (NMS) in local neighbourhood around
        // new key point
        bool bOverlap = false;
        for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
          double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
          if (kptOverlap > maxOverlap) {
            bOverlap = true;
            if (newKeyPoint.response >
                (*it).response) { // if overlap is >t AND response is higher for
                                  // new kpt
              *it = newKeyPoint;  // replace old key point with new one
              break;              // quit loop over keypoints
            }
          }
        }
        if (!bOverlap) { // only add new key point if no overlap has been found
                         // in previous NMS
          keypoints.push_back(
              newKeyPoint); // store new keypoint in dynamic list
        }
      }
    } // eof loop over cols
  }   // eof loop over rows
}