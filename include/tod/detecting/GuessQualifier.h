/*
 * GuessQualifier.h
 *
 *  Created on: Jan 15, 2011
 *      Author: Alexander Shishkov
 */

#ifndef GUESSQUALIFIER_H_
#define GUESSQUALIFIER_H_

#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <tod/core/TexturedObject.h>
#include <tod/core/Features2d.h>
#include <tod/detecting/GuessGenerator.h>

namespace tod
{
  class GuessQualifier
  {
  public:
    GuessQualifier(const cv::Ptr<TexturedObject>& texturedObject, const std::vector<cv::DMatch>& objectMatches,
                   const Features2d& test, const GuessGeneratorParameters& generatorParams);
    virtual ~GuessQualifier();
    void clarify(std::vector<Guess>& guesses);
  private:
    int prevImageIdx;
    const cv::Ptr<TexturedObject>& object;
    const std::vector<cv::DMatch>& matches;
    const Features2d& test;
    const GuessGeneratorParameters& params;

    void initPointsVector(std::vector<cv::Point2f>& imagePoints,
    		std::vector<cv::Point3f>& objectPoints, std::vector<unsigned int>& imageIndices);
    void clarifyGuess(Guess& guess, std::vector<cv::Point2f>& imagePoints,
    		std::vector<cv::Point3f>& objectPoints, std::vector<unsigned int>& imageIndices);
  };
}

#endif /* GUESSQUALIFIER_H_ */
