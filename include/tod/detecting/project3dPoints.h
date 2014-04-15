#ifndef PROJECT_3D_POINTS_H
#define PROJECT_3D_POINTS_H

#include <opencv2/opencv.hpp>

void project3dPoints(const std::vector<cv::Point3f>& points, const cv::Mat& rvec, const cv::Mat& tvec,
                     std::vector<cv::Point3f>& modif_points);
#endif