/*
 * HCluster.h
 *
 *  Created on: Jan 15, 2011
 *      Author: Alexander Shishkov
 */

#ifndef HCLUSTER_H_
#define HCLUSTER_H_

#include <opencv2/core/core.hpp>
#include <tod/detecting/GuessGenerator.h>

namespace tod
{
  struct HCluster
  {
    Guess guess;
    std::vector<cv::Point2f> projectedPoints;
    std::vector<cv::Point2f> hull;
    int inlierCount;

    HCluster(const HCluster& cluster);

    HCluster(const Guess& _guess, const std::vector<cv::Point2f>& points, int _inlierCount = -1);

    float oneWayCover(const HCluster& cluster) const;
    float cover(const HCluster& cluster) const;
    void updateHull();
    void setProjectedPoints(const std::vector<cv::Point2f>& points);
    static bool pred(const HCluster& c1, const HCluster& c2);
    static void filterClusters(std::vector<HCluster>& clusters, float minCover);
  };
}

#endif /* HCLUSTER_H_ */

