/*
* Cluster.cpp
*
*  Created on: Dec 15, 2010
*      Author: alex
*/

#include <tod/detecting/Cluster.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace cv;
using namespace std;
using namespace tod;

void Cluster::init(cv::Point2f center_, int index)
{
	center = center_;
	sum = center_;
	pointsIndices.push_back(index);
}

Cluster::Cluster(Point2f center_, int index, int imgInd)
{
	init(center_, index);
	imageIndex = imgInd;
}

Cluster::Cluster(cv::Point2f center_, int index)
{
	init(center_, index);
}

void Cluster::merge(const Cluster& cluster)
{
	pointsIndices.insert(pointsIndices.end(), cluster.pointsIndices.begin(), cluster.pointsIndices.end());
	sum += cluster.sum;
	center = sum * (1.0 / pointsIndices.size());
}

double Cluster::distance(const Cluster& cluster)
{
	double distance = std::numeric_limits<double>::max();
	if (imageIndex == cluster.imageIndex)
		distance = norm(center - cluster.center);
	return distance;
}

ClusterBuilder::~ClusterBuilder() {}

ClusterBuilder::ClusterBuilder(float maxDistance_)
{
	maxDistance = maxDistance_;
}

void ClusterBuilder::initClusters(const vector<Point2f>& imagePoints, const vector<int>& imageIndices)
{
	for (size_t i = 0; i < imagePoints.size(); i++)
	{
		clusters.push_back(Cluster(imagePoints[i], i, imageIndices[i]));
	}
}

void ClusterBuilder::convertToIndices(vector<vector<int> >& clusterIndices)
{
	clusterIndices.reserve(clusters.size());
	foreach(Cluster& cluster, clusters)
	{
		clusterIndices.push_back(cluster.pointsIndices);
	}
}

void ClusterBuilder::clusterize()
{
	bool isMergedAnyClusters;
	do
	{
		isMergedAnyClusters = false;
		for (list<Cluster>::iterator cluster = clusters.begin(); cluster != clusters.end(); cluster++)
		{
			double minDistance = std::numeric_limits<double>::max();
			list<Cluster>::iterator nearestCluster = cluster;

			list<Cluster>::iterator next = cluster;
			for (++next; next != clusters.end(); next++)
			{
				double distance = cluster->distance(*next);

				if (distance < minDistance)
				{
					minDistance = distance;
					nearestCluster = next;
				}
			}

			if (nearestCluster != cluster && minDistance < maxDistance)
			{
				isMergedAnyClusters = true;
				cluster->merge(*(nearestCluster));
				clusters.erase(nearestCluster);
			}
		}
	} while (isMergedAnyClusters);
}

void ClusterBuilder::clusterPoints(const vector<Point2f>& imagePoints, const vector<int>& imageIndices,
	vector<vector<int> >& clusterIndices)
{
	clusters.clear();
	initClusters(imagePoints, imageIndices);

	clusterize();

	convertToIndices(clusterIndices);
}

void ClusterBuilder::clusterPoints(const vector<Point2f>& imagePoints, vector<vector<int> >& clusterIndices)
{
	vector<int> imageIndexes(imagePoints.size(), -1);
	clusterPoints(imagePoints, imageIndexes, clusterIndices);
}
