/*
 * Tools.h
 *
 *  Created on: Jan 15, 2011
 *      Author: Alexander Shishkov
 */

#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv_candidate/PoseRT.h>
namespace tod
{
  using opencv_candidate::PoseRT;
  class Tools
  {
  public:
    static PoseRT invert(const PoseRT& pose);
    static float calcQuantileValue(std::vector<float> dists, float quantile);
    static cv::Point3f calcMean(const std::vector<cv::Point3f>& points);
    static void filterOutlierPoints(std::vector<cv::Point3f>& points, float quantile);
    static float computeStdDev(const std::vector<cv::Point3f>& points);
  };
}
#endif /* TOOLS_H_ */
