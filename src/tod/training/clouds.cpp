/*
* clouds.cpp
*
*  Created on: Dec 9, 2010
*      Author: erublee
*/

/*
* clouds.cpp
*
*  Created on: Nov 4, 2010
*      Author: ethan
*/

#include <tod/training/clouds.h>

#include <opencv2/opencv.hpp>
#include <boost/foreach.hpp>
#include <utility>
#include <cmath>
#include <math.h>
#include <limits>
#include <iostream>

//using namespace cv;

namespace tod
{

	void drawProjectedPoints(const Camera& camera, const Cloud& cloud, cv::Mat& projImg, int radius, int thickness)
	{

		vector<cv::Point2f> pts;
		camera.projectPoints(cloud, pts, false);
		vector<cv::Point2f>::const_iterator it = pts.begin();
		for (; it != pts.end(); ++it)
		{
			circle(projImg, *it, radius, cv::Scalar(255, 0, 255), thickness);
		}

	}

	void drawProjectedPoints(const Camera& camera, const Cloud& cloud_m, cv::Mat& projImg){
		drawProjectedPoints(camera,cloud_m,projImg,3,2);
	}

	Cloud createOneToOneCloud(const Features2d& f2d)
	{
		Cloud cloud;
		cloud.resize(f2d.keypoints.size());
		for (size_t i = 0; i < f2d.keypoints.size(); i++)
		{
			cloud[i] = cv::Point3f(f2d.keypoints[i].pt.x, f2d.keypoints[i].pt.y, 0);
		}
		return cloud;
	}

	ProjectiveAppoximateMapper::ProjectiveAppoximateMapper(const cv::Ptr<CloudProjector>& projector) :
	projector_(projector)
	{

	}
	void ProjectiveAppoximateMapper::map(const Features2d& features, const Cloud& cloud, Cloud& cloud_out) const
	{
		cv::Mat_<int> dense_map = projector_->project(cloud);
		Cloud _wcloud;
		Cloud * wcloud = &cloud_out;
		if (&cloud == wcloud)
			wcloud = &_wcloud;

		wcloud->clear();
		wcloud->reserve(features.keypoints.size());
		for (size_t i = 0; i < features.keypoints.size(); i++)
		{
			int cloud_idx = dense_map(features.keypoints[i].pt);
			if (cloud_idx > 0)
				wcloud->push_back(cloud[cloud_idx]);
			else
				//wcloud->push_back(Point3f(NAN, NaN, NaN));
				//fixed 25.10.2012
				wcloud->push_back(cv::Point3f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));

		}
		if (wcloud != &cloud_out)
			cloud_out = *wcloud;
	}

	CameraProjector::CameraProjector(const Camera& camera) :
	camera_(camera)
	{

	}
	bool checkNan(const cv::Point3f& p)
	{
		return _isnan(p.x) || _isnan(p.y) || _isnan(p.z);
	}
	bool checkNan(const cv::Point2f& p)
	{
		return _isnan(p.x) || _isnan(p.y);
	}
	void filterCloudNan(Cloud& cloud)
	{
		for (size_t i = 0; i < cloud.size(); i++)
		{
			if (_isnan(cloud[i].x) || _isnan(cloud[i].z) || _isnan(cloud[i].y))
			{
				std::swap(cloud[i], cloud.back());
				cloud.pop_back();
				i--;
			}
		}
	}

	void filterCloudNan(Cloud& cloud, Features2d& f2d)
	{
		CV_Assert(cloud.size() == f2d.keypoints.size());
		for (size_t i = 0; i < cloud.size(); i++)
		{
			if (checkNan(cloud[i]))
			{
				std::swap(f2d.keypoints[i],f2d.keypoints.back());
				cv::Mat row = f2d.descriptors.row(i);
				f2d.descriptors.row(f2d.descriptors.rows - 1).copyTo(row);
				std::swap(cloud[i], cloud.back());
				f2d.descriptors = f2d.descriptors.rowRange(0,f2d.descriptors.rows-1);
				f2d.keypoints.pop_back();
				cloud.pop_back();
				i--;
			}
		}
	}

	cv::Mat_<int> CameraProjector::project(const Cloud& cloud) const
	{
		// Cloud cloud = _cloud;
		//filterCloudNan(cloud);
		Cloud::const_iterator it = cloud.begin();

		float resolution = 2;
		cv::Mat_<int> dense_map = cv::Mat_<int>::ones(camera_.image_size) * -1;
		cv::Mat_<float> z_map(camera_.image_size);

		cv::Rect zrect = cv::Rect(cv::Point(resolution, resolution), cv::Size(z_map.size().width - 2 * resolution, z_map.size().height - 2
			* resolution));

		vector<cv::Point2f> points(cloud.size());
		camera_.projectPoints(cloud, points, false);

		//rotate the cloud
		//May need to be inverted
		//transform(pcloud, pcloud, camera_.rotationMatrix());
		//TODO transform the cloud before passing it here
		for (size_t i = 0; i < points.size(); i++)
		{
			const cv::Point2f& p = points[i];
			const cv::Point3f& pc = cloud[i];

			if (checkNan(pc) || checkNan(p))
				continue;
			cv::Point pi(p.x + 0.5, p.y + 0.5);

			cv::Rect roi(p.x - resolution / 2.0f, p.y - resolution / 2.0f, resolution, resolution);
			if (!zrect.contains(pi))
				continue;

			float zed = z_map(pi);
			if (zed > pc.z || dense_map(pi) == -1)
			{
				cv::Mat z_r = z_map(roi);
				z_r = cv::Scalar(pc.z);
				cv::Mat d_r = dense_map(roi);
				d_r = cv::Scalar(i);
			}

		}
		return dense_map;
	}

}
