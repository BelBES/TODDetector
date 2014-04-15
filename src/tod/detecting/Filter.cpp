/*
* Filter.cpp
*
*  Created on: Jan 15, 2011
*      Author: Alexander Shishkov
*/

#include <tod/detecting/Filter.h>
#include <tod/detecting/Tools.h>
#include <tod/detecting/HCluster.h>
#include <tod/detecting/project3dPoints.h>
using namespace std;
using namespace cv;

tod::StdDevFilter::StdDevFilter(const GuessGeneratorParameters& generatorParams):
params(generatorParams)
{
}

void tod::StdDevFilter::filterGuesses(::vector<Guess>& guesses)
{
	for (vector<Guess>::iterator guessIt = guesses.begin(); guessIt != guesses.end();)
	{
		double objectStddev = guessIt->getObject()->stddev;
		cout << "Factor: " << guessIt->stddev / objectStddev << endl << "Stddev: " << objectStddev << endl
			<< "GuessStd: " << guessIt->stddev << endl;
		if ((int)guessIt->inliers.size() < params.minInliersCount)
		{
			guessIt = guesses.erase(guessIt);
		}
		else if (guessIt->stddev < objectStddev  * params.minStddevFactor)
		{
			guessIt = guesses.erase(guessIt);
		}
		else
		{
			guessIt++;
		}
	}
}

void tod::OverlappingFilter::filterGuesses(std::vector<Guess>& guesses)
{
	std::vector<HCluster> clusters;
	for (std::vector<Guess>::const_iterator it = guesses.begin(); it != guesses.end(); it++)
	{
		HCluster cluster(*it, it->projected_points(), it->inliers.size());
		clusters.push_back(cluster);
	}

	const float minCover = 0.5;
	HCluster::filterClusters(clusters, minCover);

	guesses.clear();
	for (vector<HCluster>::const_iterator it = clusters.begin(); it != clusters.end(); it++)
	{
		guesses.push_back(it->guess);
	}
}

tod::CloudFilter::CloudFilter(int width, int height): width(width), height(height)
{
}

void tod::CloudFilter::filterGuesses(vector<Guess>& guesses)
{
	if (guesses.size() < 1)
		return;
	for (vector<Guess>::iterator guessIt = guesses.begin(); guessIt != guesses.end();)
	{
		vector<Point3f> cloudPoints = guessIt->getObject()->observations[0].cloud(), cloud;
		PoseRT inverted = Tools::invert(guessIt->getObject()->observations[0].camera().pose);
		project3dPoints(cloudPoints, inverted.rvec, inverted.tvec, cloud);
		Tools::filterOutlierPoints(cloud, 0.9);
		vector<Point2f> projectedPoints;

		Mat rvec;
		Rodrigues(guessIt->aligned_pose().r<cv::Mat> (), rvec);
		projectPoints(Mat(cloud), rvec, guessIt->aligned_pose().t<cv::Mat> (),
			guessIt->getK(), guessIt->getD(), projectedPoints);
		vector<Point2f>::iterator point;
		for (point = projectedPoints.begin(); point != projectedPoints.end(); )
		{
			if (point->x > 0 && point->y > 0 && point->x < width && point->y < height)
			{
				point++;
			}
			else
			{
				point = projectedPoints.erase(point);
			}
		}

		vector<Point2f> hull;
		int count = 0;
		if (projectedPoints.size() > 0)
		{
			convexHull(Mat(projectedPoints), hull);
			vector<Point> ihull;
			for (size_t i = 0; i < hull.size(); i++)
			{
				ihull.push_back(Point(hull[i].x, hull[i].y));
			}
			Mat binMask(height, width, CV_8UC1, Scalar(0));
			Point* points = (Point*)&ihull[0];
			int npoints = ihull.size();
			fillPoly(binMask, const_cast<const Point**>(&points), &npoints, 1, Scalar(255));
			for (size_t m = 0; m < guessIt->inliers.size(); m++)
			{
				Point p = guessIt->projected_points()[guessIt->inliers[m]];
				if (p.x < binMask.cols && p.y < binMask.rows)
				{
					if (binMask.at<uint8_t> (p.y, p.x) == 255)
					{
						count++;
					}
				}
			}

			cout <<"Cloud filter: " << guessIt->inliers.size() << " " << count << endl;
		}
		if (count > guessIt->inliers.size() * 0.5)
		{
			guessIt++;
		}
		else
			guessIt = guesses.erase(guessIt);
	}
}


