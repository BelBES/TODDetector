/*
 * Features2d.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: erublee
 */

#include <tod/core/Features2d.h>
#include <opencv2/core/core_c.h>
namespace tod
{
const std::string Features2d::YAML_NODE_NAME = "features";
Features2d::Features2d()
{
}

Features2d:: Features2d(const Camera& camera, const cv::Mat& image) :image(image),camera(camera)
{
}

Features2d::~Features2d()
{
}

void Features2d::lightCopy(Features2d& f2d) const{
	f2d = Features2d(camera,image);
	f2d.image_name = image_name;
	f2d.mask_name = mask_name;
	f2d.mask = mask;
}

//drawing
void Features2d::draw(cv::Mat& out, int flags) const
{
  if (!image.empty())
  {
    image.copyTo(out);
    cv::drawKeypoints(out, keypoints, out, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
  }
}
//serialization
void Features2d::write(cv::FileStorage& fs) const
{
  cvWriteComment(*fs, "Features2d", 0);
  fs << "{";
  fs << "image_name" << image_name;
  fs << "mask_name" << mask_name;
  fs << "camera";
  camera.write(fs);
  if (keypoints.size())
     cv::write(fs, "keypoints", keypoints);
   if (!descriptors.empty())
     fs << "descriptors" << descriptors;
  fs << "}";
}
void Features2d::read(const cv::FileNode& fn)
{
  //TODO, should this read in image?
  cv::read(fn["keypoints"], keypoints);
  fn["descriptors"] >> descriptors;
  image_name = (std::string)fn["image_name"];
  mask_name = (std::string)fn["mask_name"];
  camera.read(fn["camera"]);
}

}
