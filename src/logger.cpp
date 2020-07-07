#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>

using std::string;
using std::vector;


class Logger {
public:
  Logger(string fn, string detType, string desType)
      : filename(fn), detector_type(detType), descriptor_type(desType) {}
  void addTimes(float time_kp, float time_descriptor) {
    times_kp.push_back(time_kp);
    times_descriptor.push_back(time_descriptor);
  }

  void analyzeKeypoints(const vector<cv::KeyPoint> &keypoints) {
    int num_keypoints = keypoints.size();
    double mean = std::accumulate(keypoints.begin(), keypoints.end(), 0.0,
                                  [](double t, const cv::KeyPoint &kp) {
                                    return t + kp.size;
                                  }) /
                  num_keypoints;
    double variance =
        std::accumulate(keypoints.begin(), keypoints.end(), 0.0,
                        [mean](double sum, const cv::KeyPoint &kp) {
                          return sum + pow(kp.size - mean, 2);
                        }) /
        (num_keypoints - 1);

    num_kp.push_back(num_keypoints);
    mean_neigh.push_back(mean);
    var_neigh.push_back(variance);

    std::cout << "Num KPs: " << num_keypoints
              << " - Neighborhood size with mean=" << mean
              << " and var=" << variance << std::endl;
  }

  void countMatchedKeypoints(const vector<cv::DMatch> &matches) {
    num_matched_kp.push_back(matches.size());
  }

  void writeCSV() {
    std::ofstream f;
    f.open(folder + "/" + filename + ".csv", std::ios_base::app);
    vector<double> row_values{getMean(times_kp), getMean(times_descriptor),
                              getMean(num_kp), getMean(num_matched_kp), getMean(mean_neigh),
                              getMean(var_neigh)};

    auto vectorToRow = [](vector<double>& v){
      return std::accumulate(
        v.begin(), v.end(),
        string{}, [](std::string t, double e) {
          return std::move(t) + ", " + std::to_string(e);
        });
    };
    f << detector_type + ", " + descriptor_type + vectorToRow(row_values) << std::endl;
    f.close();

    f.open(folder + "/" + detector_type + "_" + descriptor_type + ".csv", std::ios_base::trunc);
    f << "Det [ms], Descr [ms], num kp [#], num matched kp [#], mean neigh size [px], var neigh size [px]" << std::endl;
    f << times_kp[0] << ", " <<  times_descriptor[0] << std::endl;
    for (size_t i = 0; i < num_kp.size(); i++) {
      row_values = {times_kp[i+1], times_descriptor[i+1], static_cast<double>(num_kp[i]),static_cast<double>(num_matched_kp[i]), mean_neigh[i], var_neigh[i]};
      f << vectorToRow(row_values).substr(2) << std::endl;
    }
    f.close();
  }

private:
  const string folder{"../logs"};

  string filename;
  string detector_type;
  string descriptor_type;

  vector<float> times_kp;         // milliseconds
  vector<float> times_descriptor; // milliseconds
  vector<uint> num_kp;
  vector<uint> num_matched_kp;
  vector<double> mean_neigh;
  vector<double> var_neigh;

  template <typename T, typename A> double getMean(const vector<T, A> &v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  }

  template <typename T, typename A, typename F>
  double getMean(const vector<T, A> &v, F &f) {
    return std::accumulate(v.begin(), v.end(), 0.0, f) / v.size();
  }

  template <typename T> double getVariance(double mean, const vector<T> &v) {
    return std::accumulate(v.begin(), v.end(), 0.0,
                           [mean](double total, const T &el) {
                             return total + pow(el - mean, 2);
                           }) /
           (v.size() - 1);
  }

  template <typename T, typename F>
  double getVariance(double mean, const vector<T> v, F &f) {
    std::accumulate(v.begin(), v.end(), 0.0, f) / (v.size());
  }
};
