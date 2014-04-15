/*
 * clouds.h
 *
 *  Created on: Nov 4, 2010
 *      Author: ethan
 */

#ifndef CLOUDS_H_TOD_
#define CLOUDS_H_TOD_

#include <tod/core/Features3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/registration.h>
#include <boost/foreach.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <limits>

#include <opencv2/core/eigen.hpp>
namespace tod
{

Cloud createOneToOneCloud(const Features2d& f2d);

inline void Cloud2Image3d(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, cv::Mat& image3d)
{
  image3d.create(cv::Size(cloud.width, cloud.height), CV_32FC3);
  for (size_t y = 0; y < cloud.height; y++)
    for (size_t x = 0; x < cloud.width; x++)
    {
      cv::Point3f& point = image3d.at<cv::Point3f> (y, x);
      point.x = cloud.at(x, y).x;
      point.y = cloud.at(x, y).y;
      point.z = cloud.at(x, y).z;
    }
}

inline void transformCloud(const pcl::PointCloud<pcl::PointXYZRGB>& in, pcl::PointCloud<pcl::PointXYZRGB>& out,
                           opencv_candidate::PoseRT& pose)
{
  //convert the tranform from our fiducial markers to
  //the Eigen
  if (!pose.estimated)
    return;
  Eigen::Matrix<float, 3, 3> R;
  Eigen::Vector3f T;
  cv::Mat cvR;
  cv::Rodrigues(pose.rvec, cvR);
  cv::cv2eigen(pose.tvec, T);
  cv::cv2eigen(cvR, R);

  Eigen::Affine3f transform;
  transform = Eigen::AngleAxisf(R);
  transform *= Eigen::Translation3f(T);
  //transform the cloud
  pcl::transformPointCloud(in, out, transform);
}

inline void Cloud2Image(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, cv::Mat& image)
{
  image.create(cv::Size(cloud.width, cloud.height), CV_8UC3);
  for (size_t y = 0; y < cloud.height; y++)
    for (size_t x = 0; x < cloud.width; x++)
    {
      cv::Vec3b& pixel = image.at<cv::Vec3b> (y, x);
      float rgb = cloud.at(x, y).rgb;
      const cv::Vec4b* rgba = reinterpret_cast<const cv::Vec4b*> (&rgb);
      pixel[0] = (*rgba)[0];
      pixel[1] = (*rgba)[1];
      pixel[2] = (*rgba)[2];
    }
}
inline void Cloud2Cv(const pcl::PointCloud<pcl::PointXYZRGB>& cloud, cv::Mat& image)
{
  image.create(cv::Size(cloud.width, cloud.height), CV_8UC3);
  for (size_t y = 0; y < cloud.height; y++)
    for (size_t x = 0; x < cloud.width; x++)
    {
      cv::Vec3b& pixel = image.at<cv::Vec3b> (y, x);
      float rgb = cloud.at(x, y).rgb;
      const cv::Vec4b* rgba = reinterpret_cast<const cv::Vec4b*> (&rgb);
      pixel[0] = (*rgba)[0];
      pixel[1] = (*rgba)[1];
      pixel[2] = (*rgba)[2];
    }
}
template<typename T>
  void cloudToPoints(std::vector<T>&v, const Cloud& cloud, const std::vector<cv::DMatch>& m)
  {
    v.clear();
    v.reserve(cloud.size());
    BOOST_FOREACH(const cv::DMatch& match, m)
          {
            int idx = match.trainIdx;
            const cv::Point3f& x = cloud[idx];
            v.push_back(T(x.x, x.y, x.z));
          }
  }
template<typename T, typename PointT>
  void PCLToPoints(std::vector<T>&v, const pcl::PointCloud<PointT>& cloud, const Features2d& f2d = Features2d())
  {
    v.clear();
    //todo make this work!!!!
    // std::cout << "cloud dense? " << (int)cloud.is_dense << endl;
    // std::cout << cloud.width << " , " << cloud.height << endl;
    if (f2d.keypoints.size()/* && cloud.is_dense*/)
    {
      float scale_factor = float(cloud.width) / f2d.camera.image_size.width; //use this to scale points by
      //  cout << scale_factor << endl;
      //  std::cout << "using cloud.at(u,v)" << endl;
      v.reserve(f2d.keypoints.size());
      BOOST_FOREACH(const cv::KeyPoint& x, f2d.keypoints)
            {
              cv::Point p(x.pt.x * scale_factor, x.pt.y * scale_factor);
              if (p.y < (int)cloud.height)
              {
                PointT xyz = cloud.at(p.x, p.y);
                v.push_back(T(xyz.data[0], xyz.data[1], xyz.data[2]));
              }
              else
                //truncate height, 640x480 does not provide depth for the bottom part of the high resolution image
                //v.push_back(T(NAN, NAN, NAN));
				//TODO fixed BeS 04.11.2012
				v.push_back(T(std::numeric_limits<int>::quiet_NaN(), std::numeric_limits<int>::quiet_NaN(), std::numeric_limits<int>::quiet_NaN()));
            }
    }
    else
    {
      v.reserve(cloud.size());
      BOOST_FOREACH(const PointT& x, cloud)
            {
              v.push_back(T(x.data[0], x.data[1], x.data[2]));
            }
    }
  }

/** Given a set of 2d points, get their corresponding input_cloud points and store them in the output_cloud
 * @param input_points
 * @param input_cloud
 * @param output_cloud
 */
template<typename OpenCvPoint, typename PclPoint>
  void cloud_to_sub_cloud(const std::vector<OpenCvPoint>&input_points, const pcl::PointCloud<PclPoint>& input_cloud,
                   unsigned int image_width, pcl::PointCloud<PclPoint>& output_cloud)
  {
    output_cloud.points.clear();
    output_cloud.is_dense = false;
    float scale_factor = float(input_cloud.width) / image_width; //use this to scale points by
    BOOST_FOREACH(const OpenCvPoint& pt, input_points)
          {
            float x = pt.x * scale_factor;
            float y = pt.y * scale_factor;
            if (y < (int)input_cloud.height)
              output_cloud.push_back(input_cloud(x, y));
            else
              //truncate height, 640x480 does not provide depth for the bottom part of the high resolution image
              output_cloud.push_back(PclPoint());
          }
  }

void filterCloudNan(Cloud& cloud, Features2d& f2d);
void filterCloudNan(Cloud& cloud);
/** \brief Interface for projecting a cloud into and image and creating an index look up table.
 *  \ingroup clouds
 */
class CloudProjector
{
public:
  virtual ~CloudProjector()
  {
  }

  /** \brief Project the given cloud into the image plane that is defined by the camera
   *  \param cloud the cloud that will be mapped into the image plane
   *  \return a matrix that is the same size as the image given by camera,
   *    and at each cell contains the index into the cloud that
   *    is closest to the ray associated with that cell.
   */
  virtual cv::Mat_<int> project(const Cloud& cloud) const = 0;
};

class CameraProjector : public CloudProjector
{
public:
  CameraProjector(const Camera& camera);
  virtual cv::Mat_<int> project(const Cloud& cloud) const;
private:
  Camera camera_; //!< camera that is observing the cloud
};

/** \brief this generates a mapping between the cloud and the features
 *  observed.
 *  Two cases are easy to identify -
 *      1. there is a one to one mapping between the point cloud and the features
 *      2. there is a projective mapping from a dense point cloud to a sparse set of features.
 *  \ingroup clouds
 */
class CloudFeaturesMapper
{
public:
  virtual ~CloudFeaturesMapper()
  {
  }
  /** \brief this should generate and index mapping between features and points in a point cloud
   * \param [in] features features to map
   * \param [in] cloud To map the 2d features to these 3d points
   * \param [out] cloud_out the cloud now one to one correspondence with the features
   */
  virtual void map(const Features2d& features, const Cloud& cloud, Cloud& cloud_out) const =0;
};

/** \brief A CloudFeaturesMapper where the observed cloud
 *      and the features already have a one to one mapping.
 *  \ingroup clouds
 */
class OneToOneMapper : public CloudFeaturesMapper
{
public:
  virtual void map(const Features2d& features, const Cloud& cloud, Cloud& cloud_out) const
  {
    cloud_out = cloud;
  }
};
/** \brief A CloudFeaturesMapper where the observed cloud
 *      and the features are mapped using a projective transformation
 *  \ingroup clouds
 */
class ProjectiveAppoximateMapper : public CloudFeaturesMapper
{
public:
  /** \brief constructor, that will copy the given projector and use it for the map operation
   *  \param projector an instance of an class derived from CloudProjector
   *  that will be used to generate the mapping.
   */
  //  template<typename ConcreteCloudProjector>
  //    ProjectiveAppoximateMapper(const ConcreteCloudProjector& projector) :
  //      projector_(new ConcreteCloudProjector(projector))
  //    {
  //
  //    }

  ProjectiveAppoximateMapper(const cv::Ptr<CloudProjector>& projector);

  virtual void map(const Features2d& features, const Cloud& cloud, Cloud& cloud_out) const;
private:
  cv::Ptr<CloudProjector> projector_;
  //TODO handle copying properly of the projector - use templatized polymorphic copier
};

void drawProjectedPoints(const Camera& camera, const Cloud& cloud_m, cv::Mat& projImg);
void drawProjectedPoints(const Camera& camera, const Cloud& cloud_m, cv::Mat& projImg, int radius, int thickness);

}//namespace tod

#endif /* CLOUDS_H_ */

