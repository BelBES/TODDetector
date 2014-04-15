/*
 * Features3d.h
 *
 *  Created on: Dec 7, 2010
 *      Author: erublee
 */

#ifndef TOD_FEATURES3D_H_
#define TOD_FEATURES3D_H_

#include <tod/core/common.h>
#include <tod/core/Features2d.h>
#include <opencv_candidate/Camera.h>

namespace tod
{
using opencv_candidate::Camera;
/** \brief A Point Cloud, opencv style...
 */
typedef std::vector<cv::Point3f> Cloud;

/** \brief Encapsulates the relationship between a cloud and a 2d keypoints
 * The cloud should have a one to one correspondence with the features.
 */
class Features3d : public Serializable, public Drawable
{
public:
  Features3d();
  virtual ~Features3d();

  /**\brief Create's a features3d object, with a cloud of unit rays in the direction of each feature
   */
  Features3d(const Features2d& f2d);
  Features3d(const Features2d& f2d, const Cloud& cloud);

  /** get the camera object, that projects the cloud into the frame that the features are in
   */
  const Camera& camera() const;

  /* \brief get access to the underlying Features2d. const version
   */
  const Features2d& features() const;
  /* \brief get access to the underlying Features2d. mutable version
    */
  Features2d& features();

  /* \brief get access to the underlying Cloud. const version
   */
  const Cloud& cloud() const;
  /* \brief get access to the underlying Cloud. mutable version
   */
  Cloud& cloud();


  //drawing
  virtual void draw(cv::Mat& out, int flags = 0) const;
  //serialization
  virtual void write(cv::FileStorage& fs) const;
  virtual void read(const cv::FileNode& fn);
  const static std::string YAML_NODE_NAME;
private:
  Features2d features_;
  Cloud cloud_;
};

}

#endif /* TOD_FEATURES3D_H_ */
