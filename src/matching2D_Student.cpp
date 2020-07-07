#include "matching2D.hpp"
#include <numeric>

#include "utils.hpp"

using std::vector;
using std::string;
// Find best matches for keypoints in two camera images based on several
// matching methods
void matchDescriptors(vector<cv::KeyPoint> &kPtsSource,
                      vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, vector<cv::DMatch> &matches,
                      string descriptorType, string matcherType,
                      string selectorType) {
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType == "MAT_BF") {
    // with SIFT
    int normType =
        descriptorType == "DES_HOG" ? cv::NORM_L2 : cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
  } else if (matcherType == "MAT_FLANN") {
    if (descSource.type() != CV_32F ||
        descRef.type() != CV_32F) { // OpenCV bug workaround : convert binary
                                    // descriptors to floating point due to a
                                    // bug in current OpenCV implementation
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }

    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  }

  // perform matching task
  // nearest neighbor (best match)
  if (selectorType == "SEL_NN") {
    // Finds the best match for each descriptor in desc1
    matcher->match(descSource, descRef, matches);

    // k nearest neighbors (k=2)
  } else if (selectorType == "SEL_KNN") {
    vector<vector<cv::DMatch>> knn_matches;
    double t = (double)cv::getTickCount();
    matcher->knnMatch(descSource, descRef, knn_matches, 2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " (KNN) with n=" << knn_matches.size() << " matches in "
         << 1000 * t / 1.0 << " ms" << std::endl;

    // filter matches using descriptor distance ratio test
    double minDescDistRatio = 0.8;
    for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {
      if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
        matches.push_back((*it)[0]);
      }
    }
    std::cout << "# keypoints removed = " << knn_matches.size() - matches.size()
         << std::endl;
  }
}

// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, string descriptorType) {
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor; // BRIEF, ORB, FREAK, AKAZE, SIFT
  if (descriptorType == "BRISK") {

    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for
                               // sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (descriptorType == "BRIEF") { // binary
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    std::cout << "Selected Brief \n";
  } else if (descriptorType == "ORB") {
    extractor = cv::ORB::create();
    std::cout << "Selected ORB \n";
  } else if (descriptorType == "FREAK") {
    extractor = cv::xfeatures2d::FREAK::create();
    std::cout << "Selected FREAK \n";
  } else if (descriptorType == "AKAZE") {
    extractor = cv::AKAZE::create();
    std::cout << "Selected AKAZE \n";
  } else if (descriptorType == "SIFT") {
    extractor = cv::xfeatures2d::SIFT::create();
    std::cout << "Selected SIFT \n";
  } else {
    std::cout << "\"" << descriptorType
              << "\" not found, using BRISK descriptor.";
    descriptorType = "BRISK";
    extractor = cv::FastFeatureDetector::create();
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0
            << " ms" << std::endl;
  return t*1000;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                           bool bVis) {
  // compute detector parameters based on image size
  int blockSize = 4; //  size of an average block for computing a derivative
                     //  covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners =
      img.rows * img.cols / std::max(1.0, minDistance); // max. num. of keypoints

  double qualityLevel = 0.01; // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  double t = (double)cv::getTickCount();
  vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                          cv::Mat(), blockSize, false, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it) {

    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << "Shi-Tomasi detection with n=" << keypoints.size()
            << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;

  // visualize results
  if (bVis)
    visKeypoints(img, keypoints, "Shi-Tomasi Corner Detector Results");
  return t * 1000.0;
}

double detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        bool bVis) {
  // solution from cornerness_harris.cpp, lesson 4.2
  // Detector parameters
  int blockSize =
      2; // for every pixel, a blockSize × blockSize neighborhood is considered
  int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
  int minResponse =
      100; // minimum value for a corner in the 8bit scaled response matrix
  double k = 0.04; // Harris parameter (see equation for details)

  cv::Mat dst_norm;
  cv::Mat dst_norm_scaled;
  cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);

  double t = (double)cv::getTickCount();
  cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  nms(keypoints, dst_norm, 0, apertureSize, minResponse);

  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << "Harris detection with n=" << keypoints.size()
            << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
    
  // visualize results
  if (bVis)
    visKeypoints(img, keypoints, "Harris keypoint detector results");

  return t*1000.0;
}

double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        std::string detectorType, bool bVis) {
  // FAST, BRISK, ORB, AKAZE, SIFT
  cv::Ptr<cv::FeatureDetector> detector;

  if (detectorType == "FAST") {
    detector = cv::FastFeatureDetector::create();
  } else if (detectorType == "BRISK") {
    detector = cv::BRISK::create();
  } else if (detectorType == "ORB") {
    detector = cv::ORB::create();
  } else if (detectorType == "AKAZE") {
    detector = cv::AKAZE::create();
  } else if (detectorType == "SIFT") {
    detector = cv::xfeatures2d::SIFT::create();
  } else {
    std::cout << "\"" << detectorType << "\" not found, using FAST detector.";
    detectorType = "FAST";
    detector = cv::FastFeatureDetector::create();
  }

  double t = (double)cv::getTickCount();
  detector->detect(img, keypoints);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << detectorType << " detector with n= " << keypoints.size()
            << " keypoints in " << 1000 * t / 1.0 << " ms" << std::endl;
  // visualize results
  if (bVis)
    visKeypoints(img, keypoints, detectorType + " keypoint detector results");
  return t*1000.0;
}