/*
 * common.h
 *
 *  Created on: Dec 7, 2010
 *      Author: erublee
 */

#ifndef TOD_COMMON_H_
#define TOD_COMMON_H_
#include <opencv2/core/core.hpp>

namespace tod
{
/** \brief simple opencv serialization interface.
 *
 * Maybe could be more elaborate.
 * Motivation for this is to enforce the ability to serialize tod data structures.
 */
class Serializable
{
public:
  virtual ~Serializable()
  {
  }

  Serializable()
  {
  }
  /** Opencv serialization, use like:
   \code
   fs << "myobj";
   myobj.write(fs);
   \endcode
   */
  virtual void write(cv::FileStorage& fs) const = 0;
  /** Opencv deserialization, use like:
   \code
   myobj.read(fn["myobj"]);
   \endcode
   */
  virtual void read(const cv::FileNode& fn) = 0;
};

/** \brief Drawable interface, this object will draw stuff to a cv::Mat, draw whatever...
 */
class Drawable
{
public:
  virtual ~Drawable()
  {
  }
  /** \brief Draw something to a matrix...
   * \param out [out] The matrix to draw to.
   * \param flags Optional flags for doing different draws
   */
  virtual void draw(cv::Mat& out, int flags = 0) const = 0;
};
}

#endif /* TOD_COMMON_H_ */
