/*
 * Recognizer.h
 *
 *  Created on: Feb 3, 2011
 *      Author: Alexander Shishkov
 */

//TODO fixed BeS 04.11.2012 removed the reference to the ROS

#ifndef RECOGNIZER_H_
#define RECOGNIZER_H_

#include <opencv2/core/core.hpp>
#include <tod/core/TrainingBase.h>
#include <tod/core/Features2d.h>
#include <tod/detecting/GuessGenerator.h>
#include <tod/detecting/Matcher.h>
//#include "opencv_candidate/ros/msgs.h"

namespace tod
{

class Recognizer
{
public:
  Recognizer();
  Recognizer(TrainingBase* base, cv::Ptr<Matcher> matcher, GuessGeneratorParameters* params, int verbose);
  virtual ~Recognizer();
  virtual void match(const Features2d& test, std::vector<Guess>& objects) = 0;
  virtual void match(const Features2d& test, const PointCloud &query_cloud, std::vector<Guess>& objects)
  {
  }

protected:
  TrainingBase* base;
  cv::Ptr<Matcher> matcher;
  GuessGeneratorParameters* params;
  int verbose;
};

class KinectRecognizer : public Recognizer
{
public:
  //TODO: fix matcher
  KinectRecognizer(TrainingBase* base, cv::Ptr<Matcher> matcher, GuessGeneratorParameters* params, int verbose,
                   std::string baseDirectory);
  ~KinectRecognizer();
  virtual void match(const Features2d& features_2d, std::vector<Guess>& objects);
  virtual void match(const Features2d& features_2d, const PointCloud &query_cloud, std::vector<Guess>& objects);
private:
//  void match_common(const Features2d& features_2d, const PointCloud &query_cloud, ros::Time n,
//                    std::vector<Guess>& in_guesses);
  void match_common(const Features2d& features_2d, const PointCloud &query_cloud,
                    std::vector<Guess>& in_guesses);
  std::string baseDirectory;
};

class TODRecognizer : public Recognizer
{
public:
  TODRecognizer(TrainingBase* base, cv::Ptr<Matcher> matcher, GuessGeneratorParameters* params, int verbose,
                std::string baseDirectory, float maxDistance);
  ~TODRecognizer();
  virtual void match(const Features2d& test2d, std::vector<Guess>& objects);
private:
  std::string baseDirectory;
  float maxDistance;
  void printMatches(std::vector<int>& objectIds);
  void drawProjections(cv::Mat& image, int id, const std::vector<Guess>& guesses, const std::string& baseDirectory);
  void drawClusters(const cv::Mat img, const std::vector<cv::Point2f>& imagePoints,
                    const std::vector<std::vector<int> >& indices);
};
}

#endif /* RECOGNIZER_H_ */
