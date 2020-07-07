/* INCLUDES FOR THIS PROJECT */
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>

#include "dataStructures.h"
#include "logger.cpp"
#include "matching2D.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {

  /* INIT VARIABLES AND DATA STRUCTURES */

  // data location
  string dataPath = "../";

  // camera
  string imgBasePath = dataPath + "images/";
  string imgPrefix =
      "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
  string imgFileType = ".png";
  int imgStartIndex = 0; // first file index to load (assumes Lidar and camera
                         // names have identical naming convention)
  int imgEndIndex = 9;   // last file index to load
  int imgFillWidth =
      4; // no. of digits which make up the file index (e.g. img-0001.png)

  // misc
  int dataBufferSize = 2;       // no. of images which are held in memory (ring
                                // buffer) at the same time
  vector<DataFrame> dataBuffer; // list of data frames which are held in memory
                                // at the same time
  bool bVis = false;            // visualize results
  bool bFocusOnVehicle = true;
  bool bLimitKpts = false;

  string detectorType =
      "BRISK"; // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
  string descriptorType = "ORB"; // BRIEF, ORB, FREAK, AKAZE, SIFT
  string matcherType = "MAT_BF"; // MAT_BF, MAT_FLANN
  string descriptorCat =
      descriptorType == "SIFT" ? "DES_HOG" : "DES_BINARY";
  string selectorType = "SEL_KNN"; // SEL_NN, SEL_KNN

  Logger logger("summary", detectorType, descriptorType);
  /* MAIN LOOP OVER ALL IMAGES */
  for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex;
       imgIndex++) {
    /* LOAD IMAGE INTO BUFFER */
    // assemble filenames for current index
    ostringstream imgNumber;
    imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
    string imgFullFilename =
        imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

    // load image from file and convert to grayscale
    cv::Mat img, imgGray;
    img = cv::imread(imgFullFilename);
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    //// TASK MP.1
    // push image into data frame buffer
    // WHY NOT USE deque??
    DataFrame frame;
    frame.cameraImg = imgGray;
    if (dataBuffer.size() > dataBufferSize) {
      dataBuffer.erase(dataBuffer.begin());
      dataBuffer.push_back(frame);
    } else {
      dataBuffer.push_back(frame);
    }

    cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

    /* DETECT IMAGE KEYPOINTS */
    // extract 2D keypoints from current image
    vector<cv::KeyPoint> keypoints;
    //// TASK MP.2 - DONE
    double tDet;
    if (detectorType == "SHITOMASI") {
      tDet = detKeypointsShiTomasi(keypoints, imgGray, bVis);
    } else if (detectorType == "HARRIS") {
      tDet = detKeypointsHarris(keypoints, imgGray, bVis);
    } else {
      tDet = detKeypointsModern(keypoints, imgGray, detectorType, bVis);
    }

    //// TASK MP.3 - DONE
    // only keep keypoints on the preceding vehicle
    cv::Rect vehicleRect(535, 180, 180, 150);
    std::vector<cv::KeyPoint> filteredKeypoints;
    if (bFocusOnVehicle) {
      for (auto &kp : keypoints) {
        auto x = kp.pt.x;
        auto y = kp.pt.y;

        if (x >= vehicleRect.x && x <= vehicleRect.x + vehicleRect.width &&
            y >= vehicleRect.y && y <= vehicleRect.y + vehicleRect.height)
          filteredKeypoints.push_back(kp);
      }
    }
    keypoints = std::move(filteredKeypoints);

    // optional : limit number of keypoints (helpful for debugging and learning)
    if (bLimitKpts) {
      int maxKeypoints = 50;

      if (detectorType == "SHITOMASI") {
        // there is no response info, so keep the first 50 as they are
        // sorted in descending quality order
        keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
      }
      cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
      cout << " NOTE: Keypoints have been limited!" << endl;
    }

    // push keypoints and descriptor for current frame to end of data buffer
    (dataBuffer.end() - 1)->keypoints = keypoints;
    cout << "#2 : DETECT KEYPOINTS done" << endl;

    /* EXTRACT KEYPOINT DESCRIPTORS */
    //// TASK MP.4 - DONE
    cv::Mat descriptors;
    double tDes = descKeypoints((dataBuffer.end() - 1)->keypoints,
                                (dataBuffer.end() - 1)->cameraImg, descriptors,
                                descriptorType);

    // push descriptors for current frame to end of data buffer
    (dataBuffer.end() - 1)->descriptors = descriptors;

    cout << "#3 : EXTRACT DESCRIPTORS done" << endl;

    // TASK MP.9 - DONE
    logger.addTimes(tDet, tDes);
    // wait until at least two images have been processed
    if (dataBuffer.size() > 1) {
      /* MATCH KEYPOINT DESCRIPTORS */
      vector<cv::DMatch> matches;

      //// TASK MP.5 - DONE
      //// TASK MP.6 - DONE
      matchDescriptors((dataBuffer.end() - 2)->keypoints,
                       (dataBuffer.end() - 1)->keypoints,
                       (dataBuffer.end() - 2)->descriptors,
                       (dataBuffer.end() - 1)->descriptors, matches,
                       descriptorCat, matcherType, selectorType);
      // store matches in current data frame
      (dataBuffer.end() - 1)->kptMatches = matches;
      cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

      //// TASK MP.7 - DONE
      logger.analyzeKeypoints(keypoints);
      //// TASK MP.8 - DONE
      logger.countMatchedKeypoints(matches);
      std::cout << "Num matched KPs: " << matches.size() << std::endl;
      // visualize matches between current and previous image
      if (bVis) {
        cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
        cv::drawMatches((dataBuffer.end() - 2)->cameraImg,
                        (dataBuffer.end() - 2)->keypoints,
                        (dataBuffer.end() - 1)->cameraImg,
                        (dataBuffer.end() - 1)->keypoints, matches, matchImg,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        vector<char>(),
                        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        string windowName = "Matching keypoints between two camera images";
        cv::namedWindow(windowName, 7);
        cv::imshow(windowName, matchImg);
        cout << "Press key to continue to next image" << endl;
        cv::waitKey(0); // wait for key to be pressed
      }
    }
  } // eof loop over all images
  logger.writeCSV();

  return 0;
}
