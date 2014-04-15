/*
* HCluster.cpp
*
*  Created on: Jan 15, 2011
*      Author: Alexander Shishkov
*/

#include <tod/detecting/HCluster.h>

using namespace tod;
using namespace std;
using namespace cv;

HCluster::HCluster(const HCluster& cluster): guess(cluster.guess), inlierCount(cluster.inlierCount)
{
	setProjectedPoints(cluster.projectedPoints);
}

HCluster::HCluster(const Guess& _guess, const vector<Point2f>& points, int _inlierCount):
guess(_guess), inlierCount(_inlierCount)
{
	setProjectedPoints(points);
}

float HCluster::oneWayCover(const HCluster& cluster) const
{
	cv::Mat _hull(hull);
	int sum = 0;
	for(size_t i = 0; i < cluster.projectedPoints.size(); i++)
	{
		float dist = cv::pointPolygonTest(_hull, cluster.projectedPoints[i], true);
		if(dist > 0)
		{
			sum++;
		}
	}
	cout << "pointPolygonTest returns " << sum << " points, total points " << cluster.projectedPoints.size() << endl;
#if 0
	cv::Mat drawImg = cv::Mat::zeros(1500, 1500, CV_8UC3);

	for(size_t k = 0; k < hull.size(); k++)
	{
		int idx1 = k;
		int idx2 = (k+1)%hull.size();

		cv::line(drawImg, cv::Point(hull[idx1].x, hull[idx1].y), cv::Point(hull[idx2].x, hull[idx2].y), cv::Scalar(0, 255, 0));
	}

	for(size_t k = 0; k < projectedPoints.size(); k++)
	{
		cv::Point2f p = projectedPoints[k];
		cv::circle(drawImg, cv::Point(p.x, p.y), 5, cv::Scalar(0, 255, 0));
	}

	for(size_t k = 0; k < cluster.projectedPoints.size(); k++)
	{
		cv::Point2f p = cluster.projectedPoints[k];
		cv::circle(drawImg, cv::Point(p.x, p.y), 5, cv::Scalar(255, 0, 0));
	}

	cv::namedWindow("oneWayCover", 1);
	cv::imshow("oneWayCover", drawImg);
	cv::waitKey(0);
#endif

	return float(sum);
}

float HCluster::cover(const HCluster& cluster) const
{
	float dist1 = oneWayCover(cluster);
	float dist2 = cluster.oneWayCover(*this);
	return (dist1 + dist2)/(projectedPoints.size() + cluster.projectedPoints.size());
}

void HCluster::updateHull()
{
	cv::convexHull(cv::Mat(projectedPoints), hull);
	std::vector<cv::Point2f> hullApprox;
	cv::approxPolyDP(cv::Mat(hull), hullApprox, 1.0, true);
	hull = hullApprox;
}

void HCluster::setProjectedPoints(const std::vector<cv::Point2f>& points)
{
	//apply found R and T to point cloud
	projectedPoints = points;
	updateHull();
}

bool HCluster::pred(const HCluster& c1, const HCluster& c2)
{
	return c1.inlierCount > c2.inlierCount;
}

void HCluster::filterClusters(std::vector<HCluster>& clusters, float minCover)
{
	std::sort(clusters.begin(), clusters.end(), HCluster::pred);
	std::vector<int> flags;
	flags.assign(clusters.size(), 1);

	for(size_t i = 0; i < clusters.size(); i++)
	{
		if(flags[i] == 0) continue;

		for(size_t j = i + 1; j < clusters.size(); j++)
		{
			if(flags[j] == 0) continue;

			cout << "Overlap between " << i << " and " << j << " is " <<
				clusters[i].oneWayCover(clusters[j])/clusters[j].projectedPoints.size() << endl;
			if(clusters[i].oneWayCover(clusters[j]) > minCover*clusters[j].projectedPoints.size())
			{
				flags[j] = 0;
				cout << "Removing cluster " << j << " inliers " << clusters[j].inlierCount <<
					", in favor of cluster " << i << " inliers " << clusters[i].inlierCount << endl;
			}
		}
	}

	std::vector<HCluster> filtered;
	for(size_t i = 0; i < clusters.size(); i++)
	{
		if(flags[i] == 1)
		{
			filtered.push_back(clusters[i]);
			cout << "Cluster " << i << " -> " << filtered.size() - 1 << endl;
		}
	}

	clusters = filtered;
}
