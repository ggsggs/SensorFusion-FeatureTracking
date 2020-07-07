#include "matching2D.hpp"
#include <numeric>

#include "utils.hpp"
using namespace std;

// Find best matches for keypoints in two camera images based on several
// matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcherType.compare("MAT_BF") == 0) {
    int normType = cv::NORM_HAMMING;
    matcher = cv::BFMatcher::create(normType, crossCheck);
  } else if (matcherType.compare("MAT_FLANN") == 0) {
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
  if (selectorType.compare("SEL_NN") == 0) {
    // Finds the best match for each descriptor in desc1
    matcher->match(descSource, descRef, matches);

    // k nearest neighbors (k=2)
  } else if (selectorType.compare("SEL_KNN") == 0) {
    vector<vector<cv::DMatch>> knn_matches;
    double t = (double)cv::getTickCount();
    matcher->knnMatch(descSource, descRef, knn_matches, 2);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << " (KNN) with n=" << knn_matches.size() << " matches in "
         << 1000 * t / 1.0 << " ms" << endl;

    // filter matches using descriptor distance ratio test
    double minDescDistRatio = 0.8;
    for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it) {
      if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance) {
        matches.push_back((*it)[0]);
      }
    }
    cout << "# keypoints removed = " << knn_matches.size() - matches.size()
         << endl;
  }
}

// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, string descriptorType) {
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor; // BRIEF, ORB, FREAK, AKAZE, SIFT
  if (descriptorType.compare("BRISK") == 0) {

    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for
                               // sampling the neighbourhood of a keypoint.

    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (descriptorType.compare("BRIEF")) { // binary
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
  } else if (descriptorType.compare("ORB")) {
    extractor = cv::ORB::create();
  } else if (descriptorType.compare("FREAK")) {
    extractor = cv::xfeatures2d::FREAK::create();
  } else if (descriptorType.compare("AKAZE")) {
    extractor = cv::AKAZE::create();
  } else if (descriptorType.compare("SIFT")) {
    extractor = cv::xfeatures2d::SIFT::create();
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
            << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                           bool bVis) {
  // compute detector parameters based on image size
  int blockSize = 4; //  size of an average block for computing a derivative
                     //  covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners =
      img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

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
            << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis)
    visKeypoints(img, keypoints, "Shi-Tomasi Corner Detector Results");
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
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
            << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  // visualize results
  if (bVis)
    visKeypoints(img, keypoints, "Harris keypoint detector results");
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                        std::string detectorType, bool bVis) {
  // FAST, BRISK, ORB, AKAZE, SIFT
  cv::Ptr<cv::FeatureDetector> detector;

  if (detectorType.compare("FAST")) {
    detector = cv::FastFeatureDetector::create();
  } else if (detectorType.compare("BRISK")) {
    detector = cv::BRISK::create();
  } else if (detectorType.compare("ORB")) {
    detector = cv::ORB::create();
  } else if (detectorType.compare("AKAZE")) {
    detector = cv::AKAZE::create();
  } else if (detectorType.compare("SIFT")) {
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
            << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
  // visualize results
  if (bVis)
    visKeypoints(img, keypoints, detectorType + " keypoint detector results");
}