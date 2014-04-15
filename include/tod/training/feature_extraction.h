/*
* features.h
*
*  Created on: Oct 28, 2012
*      Author: BeS
*/

#ifndef _FEATURES_EXTRATION_H_TOD_
#define _FEATURES_EXTRATION_H_TOD_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <tod/core/Features2d.h>

//#include <rbrief/StopWatch.h>

#include <map>

namespace tod
{
	struct FeatureExtractionParams : public Serializable
	{
		std::string detector_type;
		std::string descriptor_type;
		std::string extractor_type;
		std::map<std::string, double> detector_params;
		std::map<std::string, double> extractor_params;
		//serialization
		virtual void write(cv::FileStorage& fs) const;
		virtual void read(const cv::FileNode& fn);

		static FeatureExtractionParams CreateSampleParams();
		static const std::string YAML_NODE_NAME;
	};

	/** convert from a vector of 'keypoints' with N-d data to 2d  xy points
	*/
	void KeyPointsToPoints(const KeypointVector& keypts, std::vector<cv::Point2f>& pts);
	void PointsToKeyPoints(const std::vector<cv::Point2f>& keypts, KeypointVector& pts);

	/* 
	*brief interface to fill out a Features2d object with keypoints and descriptors
	*/
	class FeatureExtractor
	{
	public:
		virtual ~FeatureExtractor()
		{
		}
		virtual void detectAndExtract(Features2d& features) const = 0;
		static FeatureExtractor* create(const FeatureExtractionParams &params);
		static cv::FeatureDetector* createDetector(const std::string& detectorType, const std::map<std::string, double> &params);
	};

	//MultiscaleExtractor
	/** \brief Given feature detector and extractor, compute features and descriptors simultaneously
	*   at different scales
	*   \code
	#Pseudo code for multi scale features + descriptors
	N_OCTAVES = 3
	scale = 1.0
	eps = scale / pow(2, N_OCTAVES)
	do:
	detect(img,keypoints)
	extract(img,descriptions)
	for x in keypoints:
	x.scale = scale
	x.pt = x.pt * ( 1 / scale) #rescale the point so that its relative to the original image
	resize( img, img.size/2 )
	scale = scale / 2
	while(scale > eps)
	*   \endcode
	*/
	class MultiscaleExtractor : public FeatureExtractor
	{
	public:
		MultiscaleExtractor(const cv::Ptr<cv::FeatureDetector>& d, const cv::Ptr<cv::DescriptorExtractor>& e, int n_octaves);
		MultiscaleExtractor()
		{
		}
		template<typename Detector, typename Extractor>
		MultiscaleExtractor(const Detector& d, const Extractor& e, int n_octaves) :detector_(new Detector(d)), extractor_(new Extractor(e)), n_octaves_(n_octaves)
		{
		}

		virtual void detectAndExtract(Features2d& features) const;
	private:
		cv::Ptr<cv::FeatureDetector> detector_;
		cv::Ptr<cv::DescriptorExtractor> extractor_;
		int n_octaves_;
	};

	//OrbCommonParams
	enum PatchSize
	{
		PATCH_LEARNED_31 = 31
	};

	struct OrbCommonParams
	{
		static const unsigned int DEFAULT_N_LEVELS = 3;
		static const float DEFAULT_SCALE_FACTOR;
		static const unsigned int DEFAULT_FIRST_LEVEL = 0;
		static const PatchSize DEFAULT_PATCH_SIZE = PATCH_LEARNED_31;

		/** default constructor */
		OrbCommonParams(float scale_factor = DEFAULT_SCALE_FACTOR, unsigned int n_levels = DEFAULT_N_LEVELS,
			unsigned int first_level = DEFAULT_FIRST_LEVEL, PatchSize patch_size = DEFAULT_PATCH_SIZE) :
		scale_factor_(scale_factor), n_levels_(n_levels), first_level_(first_level >= n_levels ? 0 : first_level),
			patch_size_(patch_size)
		{
		}
		void read(const cv::FileNode& fn)
		{
			scale_factor_ = fn["scaleFactor"];
			n_levels_ = int(fn["nLevels"]);
			first_level_ = int(fn["firsLevel"]);
			int patch_size = fn["patchSize"];
			patch_size_ = PatchSize(patch_size);
		}
		void write(cv::FileStorage& fs) const
		{
			fs << "scaleFactor" << scale_factor_;
			fs << "nLevels" << int(n_levels_);
			fs << "firsLevel" << int(first_level_);
			fs << "patchSize" << int(patch_size_);
		}

		/* Coefficient by which we divide the dimensions from one scale pyramid level to the next */
		float scale_factor_;
		/* The number of levels in the scale pyramid */
		unsigned int n_levels_;
		/*
		The level at which the image is given
		* if 1, that means we will also look at the image scale_factor_ times bigger
		*/
		unsigned int first_level_;
		/** The size of the patch that will be used for orientation and comparisons */
		PatchSize patch_size_;
	};


	//OrbExtractor
	class OrbExtractor : public FeatureExtractor
	{
	public:
		OrbExtractor(OrbCommonParams params, int n_desired_features);
		void detectAndExtract(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
			const cv::Mat& mask = cv::Mat()) const;
		virtual void detectAndExtract(Features2d& features) const;
	private:
		//	mutable rbrief::StopWatch watch_;
		OrbCommonParams params_;
		int n_desired_features_;
		mutable cv::ORB orb_;
	};

	/*
	*SequentialExtractor
	*/
	class SequentialExtractor : public FeatureExtractor
	{
	public:
		SequentialExtractor(const cv::Ptr<cv::FeatureDetector>& d, const cv::Ptr<cv::DescriptorExtractor>& e);
		SequentialExtractor()
		{
		}

		template<typename Detector, typename Extractor>
		SequentialExtractor(const Detector& d, const Extractor& e) :
		detector_(new Detector(d)), extractor_(new Extractor(e))
		{

		}
		virtual ~SequentialExtractor()
		{
		}

		virtual void detectAndExtract(Features2d& features) const;

	private:
		cv::Ptr<cv::FeatureDetector> detector_;
		cv::Ptr<cv::DescriptorExtractor> extractor_;

	};

	/*
	*brief a nop extractor, for when you don't want to extract or detect.
	*/
	class NOPExtractor : public FeatureExtractor
	{
	public:
		virtual ~NOPExtractor()
		{
		}
		virtual void detectAndExtract(Features2d& features) const
		{
		}
	};

	/*
	*\brief a file based extractor, for when you have a file of detected features
	*/
	class FileExtractor : public FeatureExtractor
	{
	public:
		FileExtractor(const std::string& f2dname);
		virtual ~FileExtractor()
		{
		}
		virtual void detectAndExtract(Features2d& features) const;
	private:
		std::string f2dname_;
	};


}/*namespace tod*/

#endif