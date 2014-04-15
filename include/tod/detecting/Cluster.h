/*
 * Cluster.h
 *
 *  Created on: Dec 19, 2010
 *      Author: alex
 */

#ifndef CLUSTER_H_
#define CLUSTER_H_

#include <list>
#include <opencv2/core/core.hpp>

namespace tod
{

class Cluster
{
public:
  cv::Point2f center;
  cv::Point2f sum;
  int imageIndex;
  std::vector<int> pointsIndices;

  Cluster(cv::Point2f center_, int index, int imgInd);

  Cluster(cv::Point2f center_, int index);

  void init(cv::Point2f center_, int index);

  void merge(const Cluster& cluster);
  double distance(const Cluster& cluster);
};

//hierarchical clustering
class ClusterBuilder
{
public:
  ClusterBuilder(float maxDistance_);
  virtual ~ClusterBuilder();
  virtual void clusterPoints(const std::vector<cv::Point2f>& imagePoints, const std::vector<int>& imageIndices, std::vector<
      std::vector<int> >& clusterIndices);
  virtual void clusterPoints(const std::vector<cv::Point2f>& imagePoints, std::vector<std::vector<int> >& clusterIndices);

private:
  std::list<Cluster> clusters;
  float maxDistance;

  void initClusters(const std::vector<cv::Point2f>& imagePoints, const std::vector<int>& imageIndices);
  void clusterize();
  void convertToIndices(std::vector<std::vector<int> >& clusterIndices);
};

}

#endif /* CLUSTER_H_ */
