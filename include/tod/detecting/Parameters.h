/*
 * Parameters.h
 *
 *  Created on: Dec 19, 2010
 *      Author: alex
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <tod/core/common.h>
#include <tod/detecting/GuessGenerator.h>
#include <tod/training/feature_extraction.h>

#include <opencv2/opencv.hpp>
#include <string>

namespace tod
{
class MatcherParameters: public Serializable
{
public:
  MatcherParameters() {}
  ~MatcherParameters() {}

  void write(cv::FileStorage& fs) const;
  void read(const cv::FileNode& fn);

  std::string type; //FLANN or BF
  float ratioThreshold;
  int knn;
  bool doRatioTest;

  static const std::string YAML_NODE_NAME;
};

class ClusterParameters: public Serializable
{
public:
  ClusterParameters() {}
  ~ClusterParameters() {}

  void write(cv::FileStorage& fs) const;
  void read(const cv::FileNode& fn);

  int maxDistance;
  static const std::string YAML_NODE_NAME;
};

class TODParameters: public Serializable
{
public:
  TODParameters() {}
  ~TODParameters() {}

  void write(cv::FileStorage& fs) const;
  void read(const cv::FileNode& fn);

  GuessGeneratorParameters guessParams;
  FeatureExtractionParams feParams;
  MatcherParameters matcherParams;
  ClusterParameters clusterParams;

  static const std::string YAML_NODE_NAME;
  static TODParameters CreateSampleParams();
};
}
#endif /* PARAMETERS_H_ */
