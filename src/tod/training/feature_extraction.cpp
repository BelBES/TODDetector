/*
* feature_extraction.cpp
*
*  Created on: Oct 28, 2012
*      Author: erublee
*      fixed: BeS
*	TODO fixed all map<strind, float> m From m.at("name") To m["name"]
*/

#include <tod/training/feature_extraction.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>

#include <iostream>

using namespace cv;
using std::cout;

namespace tod
{
	void FeatureExtractionParams::write(cv::FileStorage& fs) const
	{
		cvWriteComment(*fs, "FeatureExtractionParams", 0);
		fs << "{";
		fs << "detector_type" << detector_type << "extractor_type" << extractor_type << "descriptor_type" << descriptor_type;
		fs << "detector_params" << "{";
		std::map<std::string, double>::const_iterator it = detector_params.begin();
		for (; it != detector_params.end(); ++it)
		{
			fs << it->first << it->second;
		}
		fs << "}";
		fs << "extractor_params" << "{";
		it = extractor_params.begin();
		for (; it != extractor_params.end(); ++it)
		{
			fs << it->first << it->second;
		}
		fs << "}";
		fs << "}";

	}
	void FeatureExtractionParams::read(const cv::FileNode& fn)
	{
		detector_type = (std::string)fn["detector_type"];
		extractor_type = (std::string)fn["extractor_type"];
		descriptor_type = (std::string)fn["descriptor_type"];

		cv::FileNode params = fn["detector_params"];
		CV_Assert(params.type() == cv::FileNode::MAP)
			;
		cv::FileNodeIterator it = params.begin();
		for (; it != params.end(); ++it)
		{
			detector_params[(*it).name()] = (double)(*it);
			//cout << "read:" << (*it).name() << " = " << detector_params[(*it).name()] << endl;
		}
		params = fn["extractor_params"];
		CV_Assert(params.type() == cv::FileNode::MAP)
			;

		it = params.begin();
		for (; it != params.end(); ++it)
		{
			extractor_params[(*it).name()] = (double)(*it);
			//cout << "read:" << (*it).name() << " = " << extractor_params[(*it).name()] << endl;
		}

	}

	FeatureExtractionParams FeatureExtractionParams::CreateSampleParams()
	{
		FeatureExtractionParams params;
		params.descriptor_type = "BRIEF";
		params.detector_type = "FAST";
		params.extractor_type = "multi-scale";
		params.extractor_params["octaves"] = 3;
		params.extractor_params["scale_factor"] = 1;
		params.detector_params["min_features"] = 500;
		params.detector_params["max_features"] = 700;
		params.detector_params["threshold"] = 200;
		params.detector_params["scale_factor"] = 1;
		return params;
	}
	const std::string FeatureExtractionParams::YAML_NODE_NAME = "feature_extraction_params";

	void KeyPointsToPoints(const KeypointVector& keypts, std::vector<cv::Point2f>& pts)
	{
		pts.clear();
		pts.reserve(keypts.size());
		for (size_t i = 0; i < keypts.size(); i++)
		{
			pts.push_back(keypts[i].pt);
		}
	}

	void PointsToKeyPoints(const std::vector<cv::Point2f>& pts, KeypointVector& kpts)
	{
		kpts.clear();
		kpts.reserve(pts.size());
		for (size_t i = 0; i < pts.size(); i++)
		{
			kpts.push_back(KeyPoint(pts[i], 6.0));
		}
	}

	/*
	FeatureExtractor
	*/
	FeatureDetector* FeatureExtractor::createDetector(const std::string& extractor_type, const std::map<std::string, double> &params)
	{
		FeatureDetector* fd = 0;
		if (!extractor_type.compare("FAST"))
		{
			if (params.end() != params.find("threshold"))
				fd = new FastFeatureDetector(params.at("threshold")/*threshold*/, true/*nonmax_suppression*/);
			else
				fd = new FastFeatureDetector();
		}
		else if (!extractor_type.compare("STAR"))
		{
			if (params.end() != params.find("threshold"))
				fd = new StarFeatureDetector(16/*max_size*/, (int)params.at("threshold")/*response_threshold*/, 10/*line_threshold_projected*/,
				8/*line_threshold_binarized*/, 5/*suppress_nonmax_size*/);
			else
				fd = new StarFeatureDetector();
		}
		else if (!extractor_type.compare("SIFT"))
		{
			//TODO fixed 28.10.2012 BeS
			//fd = new SiftFeatureDetector(SIFT::DetectorParams::GET_DEFAULT_THRESHOLD(), SIFT::DetectorParams::GET_DEFAULT_EDGE_THRESHOLD());
			fd =  new SiftFeatureDetector();
			//fd = new SiftFeatureDetector(params.at("max_features"), params.at("octaves"));
		}
		else if (!extractor_type.compare("SURF"))
		{
			float threshold = params.at("threshold");
			fd = new SurfFeatureDetector(threshold/*hessian_threshold*/, 5/*octaves*/, 4/*octave_layers*/);
		}
		else if (!extractor_type.compare("MSER"))
		{
			float threshold = params.at("threshold");

			fd = new MserFeatureDetector(5/*delta*/, 60/*min_area*/, 14400/*_max_area*/, 0.25f/*max_variation*/,
				0.2/*min_diversity*/, 200/*max_evolution*/, threshold/*area_threshold*/,
				0.003/*min_margin*/, 5/*edge_blur_size*/);
		}
		else if (!extractor_type.compare("GFTT"))
		{
			float threshold = params.at("threshold");

			fd = new GoodFeaturesToTrackDetector(1000/*maxCorners*/, threshold/*qualityLevel*/, 1./*minDistance*/,
				3/*int _blockSize*/, true/*useHarrisDetector*/, 0.04/*k*/);
		}
		else if (!extractor_type.compare("ORB"))
		{
			//fixed BeS
			//ORB::CommonParams orb_params;
			//orb_params.n_levels_ = params.at("octaves");
			//orb_params.scale_factor_ = params.at("scale_factor");

			//fd = new cv::OrbFeatureDetector(params.at("n_features"), orb_params);
			fd = new cv::OrbFeatureDetector(params.at("max_features"), params.at("scale_factor"), params.at("octaves"));
			//fd = new cv::OrbFeatureDetector();
		}
		else
			assert(0);
		return fd;
	}

	FeatureExtractor* FeatureExtractor::create(const FeatureExtractionParams &params)
	{
		FeatureExtractor* fe = 0;
		cv::Ptr<FeatureDetector> detector;

		if (params.detector_type == "DynamicSTAR")
		{
			detector = new DynamicAdaptedFeatureDetector(new StarAdjuster(),
				params.detector_params.at("min_features"),
				params.detector_params.at("max_features"), 200);
		}
		else if (params.detector_type == "DynamicSURF")
		{
			detector = new DynamicAdaptedFeatureDetector(new SurfAdjuster(),
				params.detector_params.at("min_features"),
				params.detector_params.at("max_features"), 200);
		}
		else 
		{
			FeatureExtractionParams new_params = params;
			if ((params.descriptor_type == "ORB") && (params.extractor_type == "multi-scale")) 
			{
				//fixed BeS 2013
				//new_params.detector_params.at("octaves") = 1;
				//new_params.detector_params.at("scale_factor") = 1;

				new_params.detector_params["octaves"]=1;
				new_params.detector_params["scale_factor"] = 1;
			}

			detector = FeatureExtractor::createDetector(params.detector_type, new_params.detector_params);
		}

		// Define the extractor
		//TODO fixed 28.10.2012
		cv::Ptr<DescriptorExtractor> extractor;
		if (params.descriptor_type == "ORB")
		{
			//ORB::CommonParams orb_params;
			if (params.extractor_type == "multi-scale")
			{
				//orb_params.n_levels_ = 1;
				//orb_params.scale_factor_ = 1;
				//extractor = new cv::OrbDescriptorExtractor(500,1,1);
				extractor = new cv::OrbDescriptorExtractor(params.detector_params.at("max_features"),params.extractor_params.at("scale_factor"),params.extractor_params.at("octaves"));
			}
			else
			{
				//orb_params.n_levels_ = params.extractor_params.at("octaves");
				//orb_params.scale_factor_ = params.extractor_params.at("scale_factor");
				extractor = new cv::OrbDescriptorExtractor(params.detector_params.at("max_features"),params.extractor_params.at("scale_factor"), params.extractor_params.at("octaves"));
			}	
			//extractor = new cv::OrbDescriptorExtractor(orb_params);
		} 
		else 
		{
			extractor = DescriptorExtractor::create(params.descriptor_type);
			if (extractor.empty())
			{
				throw std::runtime_error("bad extractor");
			}
		}

		if (params.extractor_type == "multi-scale")
		{
			fe = new MultiscaleExtractor(detector, extractor, params.extractor_params.at("octaves"));
		}
		else if (params.extractor_type == "sequential")
		{
			fe = new SequentialExtractor(detector, extractor);
		}
		else if (params.extractor_type == "ORB")
		{
			OrbCommonParams orb_params;
			orb_params.scale_factor_ = params.extractor_params.at("scale_factor");
			orb_params.n_levels_ = params.extractor_params.at("octaves");
			//TODO fixed BeS params.detector_params.at("n_features") to params.detector_params.at("nfeatures")
			fe = new OrbExtractor(orb_params, params.detector_params.at("max_features"));
		}

		return fe;
	}

	/*
	*MultiscaelExtractor
	*/
	MultiscaleExtractor::MultiscaleExtractor(const cv::Ptr<cv::FeatureDetector>& d,const cv::Ptr<cv::DescriptorExtractor>& e, int n_octaves):detector_(d), extractor_(e), n_octaves_(n_octaves)
	{
	}

	void MultiscaleExtractor::detectAndExtract(Features2d& features) const
	{
		int octaves = n_octaves_;
		Mat image = features.image.clone();

		float scale_factor = std::sqrt(2.);
		Mat mask = features.mask.empty() ? Mat() : features.mask.clone();

		float scale_x = 1.0f;
		float scale_y = 1.0f;
		for (int i = 0; i < octaves; i++)
		{
			std::vector<cv::KeyPoint> kpts;
			Mat descriptors;
			detector_->detect(image, kpts, mask);
			extractor_->compute(image, kpts, descriptors);

			for (size_t j = 0; j < kpts.size(); j++)
			{
				kpts[j].pt.x *= scale_x;
				kpts[j].pt.y *= scale_y;
			}

			if (i < octaves - 1)
			{
				scale_x = features.image.cols / (image.cols / scale_factor);
				scale_y = features.image.rows / (image.rows / scale_factor);

				Size n_size(image.cols / scale_factor, image.rows / scale_factor);
				resize(features.image, image, n_size);
				if (!features.mask.empty())
				{
					resize(features.mask, mask, n_size);
				}
			}
			features.keypoints.insert(features.keypoints.end(), kpts.begin(), kpts.end());
			Mat n_desc(features.descriptors.rows + descriptors.rows, extractor_->descriptorSize(), extractor_->descriptorType());
			Mat top_desc(n_desc.rowRange(Range(0, features.descriptors.rows)));
			Mat bottom_desc(n_desc.rowRange(Range(features.descriptors.rows, features.descriptors.rows + descriptors.rows)));

			features.descriptors.copyTo(top_desc);
			descriptors.copyTo(bottom_desc);
			features.descriptors = n_desc;
#if 0
			imshow("octave", image);
			imshow("scaled mask", mask);
			waitKey(0);
#endif
		}
	}

	//OrbCommonParams
	const float OrbCommonParams::DEFAULT_SCALE_FACTOR=1.2;

	//OrbExtractor
	OrbExtractor::OrbExtractor(OrbCommonParams params, int n_desired_features) : params_(params), n_desired_features_(n_desired_features)
	{
		//TODO fixed BeS 28.10.2012
		//orb_ = cv::ORB(n_desired_features_, params);
		orb_ = cv::ORB(n_desired_features_, params.scale_factor_, params.n_levels_, 31, params.first_level_, 2, 0, params.patch_size_);
	}

	cv::Mat GetAffineRotationMat(float a, cv::Point t)
	{
		a = a * CV_PI / 180;
		cv::Mat ar = (cv::Mat_<float>(3, 3) << cos(a), -sin(a), t.x, sin(a), cos(a), t.y, 0, 0, 1);
		return ar;
	}

	void GetPatches(cv::Size patch_size, const cv::Mat& image, const std::vector<cv::KeyPoint>& kpts, std::vector<cv::Mat>& patches)
	{
		patches.clear();
		cv::Rect roi(0, 0, patch_size.width + 10, patch_size.height + 10);
		cv::Rect image_roi(0, 0, image.size().width, image.size().height);

		//TODO fixed BeS 28.10.2012 
		//BOOST_FOREACH(const cv::KeyPoint& kpt, kpts)
		cv::KeyPoint kpt;
		int kpts_size=kpts.size();
		for(int i=0; i<kpts_size; i++)
		{
			kpt=kpts[i];
			roi.x = kpt.pt.x - patch_size.width / 2 - 5;
			roi.y = kpt.pt.y - patch_size.height / 2 - 5;
			if ((image_roi & roi).area() != roi.area())
				continue;
			cv::Point t(roi.width / 2, roi.height / 2);
			cv::Point t2(patch_size.width / 2, patch_size.height / 2);
			cv::Mat w = GetAffineRotationMat(0, t2) * GetAffineRotationMat(-kpt.angle, Point())* GetAffineRotationMat(0, -t);
			cv::Mat patch;
			cv::warpPerspective(image(roi), patch, w, patch_size);
			patches.push_back(patch);
		}
	}

	void OrbExtractor::detectAndExtract(Features2d& features) const
	{
		detectAndExtract(features.image, features.keypoints, features.descriptors, features.mask);
	}

	void OrbExtractor::detectAndExtract(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, const cv::Mat& mask) const
	{
		orb_(image, mask, keypoints, descriptors);
	}

	//SequentialExtractor
	SequentialExtractor::SequentialExtractor(const cv::Ptr<cv::FeatureDetector>& d, const cv::Ptr<cv::DescriptorExtractor>& e):detector_(d), extractor_(e)
	{
	}

	void SequentialExtractor::detectAndExtract(Features2d& features) const
	{
		detector_->detect(features.image, features.keypoints, features.mask);
		extractor_->compute(features.image, features.keypoints, features.descriptors);
	}

	//FileExtractor
	FileExtractor::FileExtractor(const std::string& f2dname) : f2dname_(f2dname)
	{
	}

	void FileExtractor::detectAndExtract(Features2d& features) const
	{
		cv::FileStorage fs(f2dname_, cv::FileStorage::READ);
		cv::read(fs["keypoints"], features.keypoints);
		//    fs["keypoints"] >> features.keypoints;
		fs["descriptors"] >> features.descriptors;
	}

}/*namespace tod*/