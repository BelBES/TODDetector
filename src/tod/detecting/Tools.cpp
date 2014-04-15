/*
* Tools.cpp
*
*  Created on: Jan 15, 2011
*      Author: Alexander Shishkov
*/
#include <tod/detecting/Tools.h>
#include <opencv2/calib3d/calib3d.hpp>
using namespace tod;
using namespace std;
using namespace cv;
float Tools::calcQuantileValue(vector<float> dists, float quantile)
{
	sort(dists.begin(), dists.end());
	return dists[int(floor(dists.size()*quantile))];
}

PoseRT Tools::invert(const PoseRT& pose)
{
	PoseRT inverted;
	Mat R, RInv;
	Rodrigues(pose.rvec,R);
	RInv = R.inv();
	Rodrigues(RInv, inverted.rvec);
	inverted.tvec = RInv*pose.tvec;
	inverted.tvec = inverted.tvec*(-1);
	return inverted;
}

Point3f Tools::calcMean(const vector<Point3f>& points)
{
	Point3f mean;
	for(size_t i = 0; i < points.size(); i++)
	{
		mean += points[i];
	}

	mean = mean*(1.0/points.size());
	return mean;
}

void Tools::filterOutlierPoints(vector<Point3f>& points, float quantile)
{
	Point3f mean = calcMean(points);
	vector<float> dists;
	for(size_t i = 0; i < points.size(); i++)
	{
		dists.push_back(norm(points[i] - mean));
	}
	float quantileValue = calcQuantileValue(dists, 0.9);

	vector<Point3f> filtered;
	for(size_t i = 0; i < points.size(); i++)
	{
		if(norm(points[i] - mean) < quantileValue)
		{
			filtered.push_back(points[i]);
		}
	}

	points = filtered;
}

float Tools::computeStdDev(const vector<Point3f>& points)
{
	Point3f sum(0, 0, 0);
	float sum2 = 0;
	for(size_t j = 0; j < points.size(); j++)
	{
		sum += points[j];
		sum2 += points[j].dot(points[j]);
	}
	float stddev = sqrt(sum2/points.size() - sum.dot(sum)/(points.size()*points.size()));

	return stddev;
}
