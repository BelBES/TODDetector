/*
* PoseGenerator.h
*
*  Created on: Dec 15, 2010
*      Author: alex
*/

#ifndef POSEGENERATOR_H_
#define POSEGENERATOR_H_

#include <fstream>
#include <iostream>
#include <list>
#include <string>

#include <boost/dynamic_bitset.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <opencv2/opencv.hpp>
//#include <posest/pnp_ransac.h>
#include <tod/core/TexturedObject.h>
#include <opencv_candidate/PoseRT.h>

namespace tod
{
	typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;

	enum RecognitionMode
	{
		TOD = 0, KINECT = 1
	};

	class Guess
	{
	public:

		virtual ~Guess();

		Guess()
		{
		}
		Guess(const cv::Ptr<TexturedObject>& object, const opencv_candidate::Pose& rt, const cv::Mat & k, const cv::Mat & d,
			const cv::Mat& queryImage);

		std::vector<int> inliers;
		std::vector<cv::Point2f> image_points_;

		std::vector<unsigned int> image_indices_;
		float stddev;

		const cv::Ptr<TexturedObject> getObject() const;

		const opencv_candidate::Pose& aligned_pose() const;
		void set_aligned_pose(const opencv_candidate::Pose& pose);

		/** \brief draw the guess -
		*    0 draws the reprojection
		*    1 the correspondence;
		*
		*/
		void draw(cv::Mat& out, int flags = 0, const std::string& directory = ".") const;

		/** Original 3D points from the training data but to which the inverted camera transform has been applied
		* to be in an absolute frame
		*/
		std::vector<cv::Point3f> aligned_points_;

		/**
		* @return
		*/
		const std::vector<cv::Point2f> & projected_points() const;

		const cv::Mat& getK() const;
		const cv::Mat& getD() const;

	private:
		std::vector<cv::Point2f> projected_points_;
		cv::Ptr<TexturedObject> object_;
		opencv_candidate::Pose aligned_pose_;
		/** The calibration matrix
		*/
		cv::Mat k_;
		/** The distortion matrix
		*/
		cv::Mat d_;
		cv::Mat query_; //!< the image that generated the imagePoints
		static const cv::Scalar colors[6];
		static int colorId;
	};

	struct GuessGeneratorParameters : public Serializable
	{
		void write(cv::FileStorage& fs) const;

		void read(const cv::FileNode& fn);

		int minClusterSize;
		float minStddevFactor;
		int minInliersCount;
		int ransacIterationsCount;
		float maxProjectionError;
		float descriptorDistanceThreshold;

		static const std::string YAML_NODE_NAME;
	};

	class GuessGenerator
	{
	public:
		GuessGenerator(GuessGeneratorParameters params_);
		void calculateGuesses(const cv::Ptr<TexturedObject>& object, const std::vector<std::vector<int> >& clusterIndices,
			const std::vector<cv::DMatch>& matches, const Features2d& test, std::vector<Guess>& guesses);
		void calculateGuesses(const Features2d& features_2d, const PointCloud &query_cloud,
			const cv::Ptr<TexturedObject>& object, const std::vector<cv::DMatch>& matches,
			std::vector<Guess>& guesses);
	private:
		void calculateGuesses(const Features2d& features_2d, const PointCloud &query_cloud,
			const cv::Ptr<TexturedObject>& object, const std::vector<cv::DMatch>& matches,
			std::vector<Guess>& guesses, boost::dynamic_bitset<> & is_interesting);
		GuessGeneratorParameters params;
	};

}

#endif /* POSEGENERATOR_H_ */
