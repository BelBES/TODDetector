/*
 * Features2d.h
 *
 *  Created on: Dec 7, 2010
 *      Author: erublee
 */

#ifndef FEATURES2D_H_
#define FEATURES2D_H_

#include <tod/core/common.h>

#include <opencv2/features2d/features2d.hpp>
#include <opencv_candidate/PoseRT.h>
#include <opencv_candidate/Camera.h>

namespace tod
{
using opencv_candidate::Camera;
using opencv_candidate::PoseRT;
/** store list of key points from 'feature detection'
 */
typedef std::vector<cv::KeyPoint> KeypointVector;
/** \brief Gloms together all relevant data that has to do with features2d detection and extraction.
 */
class Features2d : public Serializable, public Drawable
{
public:
  Features2d();
  Features2d(const Camera& camera, const cv::Mat& image);
  virtual ~Features2d();

  /** \brief Will draw the image and features.
   */
  virtual void draw(cv::Mat& out, int flags = 0) const;
  //serialization
  virtual void write(cv::FileStorage& fs) const;
  virtual void read(const cv::FileNode& fn);

  /** \brief use default copy constructor on all members except for keypoints and descriptors.
   */
  void lightCopy(Features2d& f2d) const;
  cv::Mat image; //!< image for doing detection and extraction of features
  cv::Mat mask; //!< mask, if non empty should be the same size as image - this indicates where to detect features
  KeypointVector keypoints; //!< the detected keypoints
  cv::Mat descriptors; //!<the detected descriptors
  std::string image_name; //!< image file name
  std::string mask_name; //!< mask file name
  Camera camera; //!< camera used to take the image
  static const std::string YAML_NODE_NAME;
};

}

#endif /* FEATURES2D_H_ */
