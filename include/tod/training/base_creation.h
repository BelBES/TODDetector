#ifndef __BASE_CREATION_H__
#define __BASE_CREATION_H__

#include <iostream>
#include <string>

#include <tod/training/feature_extraction.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

namespace tod
{
	class BaseCreator
	{
	public:
		virtual ~BaseCreator(){}
		BaseCreator(const std::string& directory_, cv::Ptr<tod::FeatureExtractor> extractor_);
		int createBase(int scale);
	private:
		static std::string CONFIG_NAME;
		static std::string IMAGES_LIST;
		static std::string CLOUDS_LIST;
		std::string directory;
		cv::Ptr<tod::FeatureExtractor> extractor;
	};

}/*namespace tod*/

#endif