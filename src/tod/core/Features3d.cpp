/*
* Features3d.cpp
*
*  Created on: Dec 7, 2010
*      Author: erublee
*/

#include <tod/core/Features3d.h>
#include <opencv2/core/core_c.h>
namespace tod
{

	Features3d::Features3d()
	{
	}

	Features3d::~Features3d()
	{

	}
	Features3d::Features3d(const Features2d& f2d):features_(f2d),cloud_(Cloud(f2d.keypoints.size(),cv::Point3f(0))){

	}
	Features3d::Features3d(const Features2d& f2d, const Cloud& cloud) :
	features_(f2d), cloud_(cloud)
	{

	}
	const Camera& Features3d::camera() const
	{
		return features_.camera;
	}

	const Features2d& Features3d::features() const
	{
		return features_;
	}

	const Cloud& Features3d::cloud() const
	{
		return cloud_;
	}
	Features2d& Features3d::features()
	{
		return features_;
	}

	Cloud& Features3d::cloud()
	{
		return cloud_;
	}

	//drawing
	void Features3d::draw(cv::Mat& out, int flags) const
	{
		//todo implement features3d drawing

	}
	const std::string Features3d::YAML_NODE_NAME = "features3d";
	//serialization
	void Features3d::write(cv::FileStorage& fs) const
	{
		cvWriteComment(*fs, "Features3d", 0);
		fs << "{";
		if (cloud_.size())
		{
			//TODO fixed BeS
			//fs << "cloud" << "[:" << cloud_ << "]";
			fs << "cloud" << cloud_ ;
		}
		fs << "features";
		features_.write(fs);
		fs << "}";
	}
	void Features3d::read(const cv::FileNode& fn)
	{
		//TODO, should this read in image?
		fn["cloud"] >> cloud_;
		features_.read(fn["features"]);
	}

}
