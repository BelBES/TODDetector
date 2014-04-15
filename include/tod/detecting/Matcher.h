/*
 * Matcher.h
 *
 *  Created on: Dec 15, 2010
 *      Author: alex
 */

#ifndef MATCHER_H_
#define MATCHER_H_

#include <tod/core/common.h>
#include <tod/core/TrainingBase.h>
#include <tod/detecting/Parameters.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>

namespace tod
{
  typedef std::vector<cv::DMatch> Matches;

  class Matcher
  {
  public:
    /** Set of matches, usually for a given object
     */
    virtual ~Matcher();
    Matcher(const cv::Ptr<cv::DescriptorMatcher> matcher, int knn = 1);

    /**Add the TrainingBase to the matcher, this copies all descriptors into the matcher
     */
    virtual void add(const TrainingBase& base);
    virtual void match(const cv::Mat& queryDescriptors);
    void getObjectMatches(int objectId, tod::Matches& matches) const;

    /** \brief Get all of the labels, sorted by the number of matches per label.
     *  \param labels_sizes [out] a vector, where each element is (label,size )where size is the
     */
    void getLabelSizes(std::vector<std::pair<int,int> >& labels_sizes) const;
    void getImageMatches(int objectId, int imageId, tod::Matches& matches) const;

    static Matcher* create(MatcherParameters& params);
    static void drawMatches(const TrainingBase& base, const cv::Ptr<Matcher> matcher,
      const cv::Mat& testImage, const KeypointVector& testKeypoints, std::string directory);
    template<typename T>
    static bool pair_second_greater(const T& lhs,const T& rhs){
      return lhs.second > rhs.second;
    }

  protected:
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<tod::Matches > objectMatches;
    std::vector<int> objectSizes; //count of fetures3d elements in each TexturedObject in TrainingBase
    std::vector<int> objectIds;
    int knn_;

    Matcher() {};
    virtual void privateMatch(const cv::Mat& queryDescriptors, tod::Matches& matches);
    virtual void convertMatches(const tod::Matches& matches);
    virtual int getObjectIndex(const cv::DMatch& match);
  };

  class RatioTestMatcher : public Matcher
  {
  public:
    RatioTestMatcher(const cv::Ptr<cv::DescriptorMatcher> matcher_, int knnNum_, double ratioThreshold_);
    virtual ~RatioTestMatcher();

    virtual void match(const cv::Mat& queryDescriptors);
    /** Go over the different matches and remove the ones involving features matching to many.
     * E.g., we have a test image with a checkerboard where all corners match one unique feature in the training image.
     * Those a re obviously bad matches for geometrical constraints and need to be removed
     * @param maximum_match_number maximum allowed number of matches from a feature to the original image
     */
    void matchPruner(unsigned int maximum_match_number = 1);

  private:
    int knnNum;
    double ratioThreshold;
  };

  class FlannRatioTestMatcher : public Matcher
    {
    public:
      FlannRatioTestMatcher(int knnNum_, double ratioThreshold_);
      virtual ~FlannRatioTestMatcher();
      void add(const TrainingBase& base);
      virtual void match(const cv::Mat& queryDescriptors);

    private:
      int knnNum;
      double ratioThreshold;
      cv::Mat descriptors;
      std::vector<int> imageIndices;
      std::vector<int> objectIndices;
      std::vector<int> keypointIndices;
      cv::flann::Index* flann_index;
    };
}
#endif /* MATCHER_H_ */
