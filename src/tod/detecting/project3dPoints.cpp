#include <tod/detecting/project3dPoints.h>
#include <vector>
#include <iostream>

using namespace std;
/*
*TODO fixed BeS 04.11.2012
*/
void project3dPoints(const std::vector<cv::Point3f>& points, const cv::Mat& rvec, const cv::Mat& tvec, vector<cv::Point3f>& modif_points)
{
	modif_points.clear();
	modif_points.resize(points.size());
	cv::Mat R(3, 3, CV_64FC1);
	Rodrigues(rvec, R);
	for (size_t i = 0; i < points.size(); i++)
	{
		modif_points[i].x = R.at<double> (0, 0) * points[i].x + R.at<double> (0, 1) * points[i].y + R.at<double> (0, 2)
			* points[i].z + tvec.at<double> (0, 0);
		modif_points[i].y = R.at<double> (1, 0) * points[i].x + R.at<double> (1, 1) * points[i].y + R.at<double> (1, 2)
			* points[i].z + tvec.at<double> (1, 0);
		modif_points[i].z = R.at<double> (2, 0) * points[i].x + R.at<double> (2, 1) * points[i].y + R.at<double> (2, 2)
			* points[i].z + tvec.at<double> (2, 0);
	}
}
/*********************/
