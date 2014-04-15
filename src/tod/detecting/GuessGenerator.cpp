/*
* PoseGenerator.cpp
*
*  Created on: Dec 15, 2010
*      Author: alex
*/
#include <opencv_candidate/PoseRT.h>
#include <tod/detecting/GuessGenerator.h>
#include <tod/detecting/Tools.h>
#include <tod/detecting/project3dPoints.h>
#include <boost/foreach.hpp>
#include <fiducial/fiducial.h>
#include "pcl/filters/filter.h"
#include "pcl/sample_consensus/prosac.h"
#include "pcl/sample_consensus/ransac.h"
#include "pcl/sample_consensus/sac_model_registration.h"
#include "tod/training/clouds.h"
#include <cmath>
//#include <Eigen/src/StlSupport/StdVector.h>

#include <opencv2/imgproc/types_c.h>

#define foreach BOOST_FOREACH

//using namespace std;
//using namespace cv;
using namespace tod;

const string GuessGeneratorParameters::YAML_NODE_NAME = "GuessParameters";

void GuessGeneratorParameters::write(cv::FileStorage& fs) const
{
	cvWriteComment(*fs, "GuessParameters", 0);
	fs << "{";
	fs << "min_cluster_size" << minClusterSize;
	fs << "min_inliers_count" << minInliersCount;
	fs << "ransac_iterations_count" << ransacIterationsCount;
	fs << "max_projection_error" << maxProjectionError;
	fs << "descriptor_distance_threshold" << descriptorDistanceThreshold;
	fs << "min_stddev_factor" << minStddevFactor;
	fs << "}";
}

void GuessGeneratorParameters::read(const cv::FileNode& fn)
{
	minClusterSize = (int)fn["min_cluster_size"];
	minInliersCount = (int)fn["min_inliers_count"];
	ransacIterationsCount = (int)fn["ransac_iterations_count"];
	maxProjectionError = (float)fn["max_projection_error"];
	descriptorDistanceThreshold = (float)fn["descriptor_distance_threshold"];
	minStddevFactor = (float)fn["min_stddev_factor"];
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const cv::Scalar Guess::colors[6] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255)};
int Guess::colorId = 0;

Guess::~Guess()
{
}
Guess::Guess(const cv::Ptr<TexturedObject>& object, const opencv_candidate::Pose& rt, const cv::Mat & k,
	const cv::Mat & d, const cv::Mat& queryImage) :
object_(object), aligned_pose_(rt), k_(k), d_(d), query_(queryImage)
{
}

const opencv_candidate::Pose& Guess::aligned_pose() const
{
	return aligned_pose_;
}

const std::vector<cv::Point2f> & Guess::projected_points() const
{
	if (projected_points_.empty())
	{
		unsigned int image_index = 0;
		if (!image_indices_.empty())
			image_index = image_indices_[0];
		cv::Mat K = k_, D = d_;
		if (K.empty())
			K = object_->observations[image_index].camera().K;
		if (D.empty())
			D = object_->observations[image_index].camera().D;

		vector<cv::Point3f> points;
		std::vector<cv::Point2f> & projected_points = const_cast<std::vector<cv::Point2f> &> (projected_points_);
		if (!image_indices_.empty())
		{
			cv::Mat rvec;
			cv::Rodrigues(aligned_pose_.r<cv::Mat> (), rvec);
			cv::projectPoints(cv::Mat(aligned_points_), rvec, aligned_pose_.t<cv::Mat> (), K, D,
				projected_points);
		}
		else
		{
			int cloudIndex = 0;
			PoseRT pose = Tools::invert(object_->observations[cloudIndex].camera().pose);
			vector<cv::Point3f> rotatedPoints;
			project3dPoints(object_->observations[cloudIndex].cloud(), pose.rvec, pose.tvec,
				rotatedPoints);
			cv::Mat rvec;
			cv::Rodrigues(aligned_pose_.r<cv::Mat> (), rvec);
			cv::projectPoints(cv::Mat(rotatedPoints), rvec, aligned_pose_.t<cv::Mat> (), K, D, projected_points);
		}

		/* TODO
		for (std::vector<Point2f>::iterator point = projected_points.begin(); point != projected_points.end();)
		{
		if (point->x < 0 || point->y < 0 || point->x >= size.width || point->y >= size.height)
		{
		point = projected_points.erase(point);
		}
		else
		{
		point++;
		}
		}*/
	}
	return projected_points_;
}

const cv::Ptr<TexturedObject> Guess::getObject() const
{
	return object_;
}

void Guess::set_aligned_pose(const opencv_candidate::Pose& pose)
{
	aligned_pose_ = pose;
}

void Guess::draw(cv::Mat& out, int flags, const std::string& directory) const
{
	if (flags == 0)
	{
		if (query_.channels() != 3)
		{
			if (out.empty())
				cvtColor(query_, out, CV_GRAY2RGB);
		}
		else if (out.empty())
			query_.copyTo(out);

		vector<cv::Point2f> projectedPoints;

		if (!image_indices_.empty())
			projectedPoints = projected_points();
		else
		{
			int cloudIndex = 0;
			cv::Mat K = k_, D = d_;
			if (K.empty())
				K = object_->observations[cloudIndex].camera().K;
			if (D.empty())
				D = object_->observations[cloudIndex].camera().D;

			vector<cv::Point3f> cloud, rotatedPoints;

			cv::Mat rvec(3, 1, CV_32F), tvec = aligned_pose_.t<cv::Mat>();
			cv::Rodrigues(aligned_pose_.r<cv::Mat>(), rvec);
			project3dPoints(cv::Mat(aligned_points_), rvec, tvec, rotatedPoints);
			Tools::filterOutlierPoints(rotatedPoints, 0.9);
			if (rotatedPoints.size() < 7)
				return;
			projectPoints(cv::Mat(rotatedPoints), cv::Mat(3, 1, CV_64F, cv::Scalar(0.0)), cv::Mat(3, 1, CV_64F, cv::Scalar(0.0)),
				K, D, projectedPoints);
		}

		//draw cloud
		int cloudIndex = 0;
		if (!image_indices_.empty())
			cloudIndex = image_indices_[0];
		cv::Mat K = k_, D = d_;
		if (K.empty())
			K = object_->observations[cloudIndex].camera().K;
		if (D.empty())
			D = object_->observations[cloudIndex].camera().D;

		vector<cv::Point3f> points;
		vector<cv::Point2f> projected_points;

		PoseRT pose = Tools::invert(object_->observations[cloudIndex].camera().pose);
		vector<cv::Point3f> rotatedPoints;
		project3dPoints(object_->observations[cloudIndex].cloud(), pose.rvec, pose.tvec,
			rotatedPoints);
		cv::Mat rvec;
		cv::Rodrigues(aligned_pose_.r<cv::Mat> (), rvec);
		cv::projectPoints(cv::Mat(rotatedPoints), rvec, aligned_pose_.t<cv::Mat> (), K, D, projected_points);

		cv::Scalar color = colors[(colorId++)%6];
		int size = projected_points.size();
		std::vector<cv::Point> tmp;
		std::vector<std::vector<cv::Point>> hull;
		hull.push_back(std::vector<cv::Point>());
		for(int i=0; i<size; i++)
		{
			cv::circle(out, cv::Point(projected_points[i].x, projected_points[i].y), 1, color);
			tmp.push_back(cv::Point(projected_points[i].x, projected_points[i].y));
		}
		cv::convexHull(tmp, hull[0]);
		cv::drawContours(out, hull, 0, color, 3);


		// Remove points out of the image but draw the ones inside the image
		cv::Point2f topright(0, 0);
		vector<cv::Point2f>::iterator point;
		for (point = projectedPoints.begin(); point != projectedPoints.end();)
		{
			if (point->x < 0 || point->y < 0 || point->x >= out.cols || point->y >= out.rows)
			{
				point = projectedPoints.erase(point);
			}
			else
			{
				circle(out, *point, 3, cv::Scalar(255, 0, 0), 1);
				topright += *point;
				point++;
			}
		}

#ifdef TOD_MATCHER
		// Add a convex hull around the matched points
		if (!projectedPoints.empty())
		{
			vector<Point2f> hull;
			convexHull(Mat(projectedPoints), hull);

			vector<Point> ihull;
			for (size_t i = 0; i < hull.size(); i++)
			{
				ihull.push_back(Point(hull[i].x, hull[i].y));
			}

			vector<vector<Point> > _hull;
			_hull.push_back(ihull);

			drawContours(out, _hull, -1, Scalar(255, 0, 0), 1);
		}
#endif

		// Add the name of the object
		topright.x /= projectedPoints.size();
		topright.y /= projectedPoints.size();
		putText(out, object_->name, topright, CV_FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 255), 2, CV_AA, false);

		/*
		if (!image_indices_.empty())
		{
		fiducial::PoseDrawer(out, k_, aligned_pose_);
		}
		else
		{
		int cloudIndex = 0;
		PoseRT frame_pose, camera_pose = object_->observations[cloudIndex].camera().pose;
		cv::composeRT(camera_pose.rvec, camera_pose.tvec, aligned_pose_.rvec, aligned_pose_.tvec, frame_pose.rvec,
		frame_pose.tvec);
		fiducial::PoseDrawer(out, object_->observations[cloudIndex].camera().K, frame_pose);
		}
		*/
	}
	else if (flags == 1)
	{
		if (!image_indices_.empty())
		{
			Features3d f3d = object_->observations[image_indices_[0]];
			Features2d f2d = f3d.features();
			if (f2d.image.empty())
			{
				string filename = directory + "/" + object_->name + "/" + f2d.image_name;
				f2d.image = cv::imread(filename, 0);
			}

			vector<cv::Point3f> cloud, rotatedPoints;
			int trainImgIdx = image_indices_[0];
			opencv_candidate::PoseRT pose = object_->observations[trainImgIdx].camera().pose;
			project3dPoints(cv::Mat(aligned_points_), pose.rvec, pose.tvec, rotatedPoints);

			cv::Mat K = k_, D = d_;
			if (K.empty())
				K = object_->observations[trainImgIdx].camera().K;
			if (D.empty())
				D = object_->observations[trainImgIdx].camera().D;
			vector<cv::Point2f> oppoints;
			projectPoints(cv::Mat(rotatedPoints), cv::Mat(3, 1, CV_64F, cv::Scalar(0.0)), cv::Mat(3, 1, CV_64F, cv::Scalar(0.0)),
				K, D,
				oppoints);

			vector<cv::DMatch> matches;
			vector<char> mask(aligned_points_.size(), 0);
			KeypointVector kpts_train, kpts_query;
			for (size_t i = 0; i < aligned_points_.size(); i++)
			{
				cv::DMatch m(i, i, 0);
				matches.push_back(m);
				kpts_train.push_back(cv::KeyPoint(oppoints[i].x, oppoints[i].y, 0, 0, 0, 0));
				kpts_query.push_back(cv::KeyPoint(image_points_[i].x, image_points_[i].y, 0, 0, 0, 0));
			}
			foreach(int inlier,inliers) {
				mask[inlier] = 1;
			}

			cv::drawMatches(f2d.image, kpts_train, query_, kpts_query, matches, out,
				cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), mask);
		}
	}
}


const cv::Mat& Guess::getK() const
{
	return k_;
}

const cv::Mat& Guess::getD() const
{
	return d_;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

GuessGenerator::GuessGenerator(GuessGeneratorParameters params_) :
params(params_)
{
}

struct GuessMatch
{
public:
	GuessMatch(const cv::DMatch* match, unsigned int index, const pcl::PointXYZ & point) :
	  match_(match), index_(index), point_(point)
	  {
	  }
	  const cv::DMatch* match_;
	  unsigned int index_;
	  pcl::PointXYZ point_;
};

/** Class used to compare pointers of cv::DMatch
*/
struct GuessMatchComparator
{
	bool operator()(const GuessMatch & match_1, const GuessMatch & match_2) const
	{
		return match_1.match_->distance < match_2.match_->distance;
	}
};

void GuessGenerator::calculateGuesses(const Features2d& features_2d, const PointCloud &query_cloud,
	const cv::Ptr<TexturedObject>& object, const std::vector<cv::DMatch>& in_matches,
	std::vector<Guess>& guesses, boost::dynamic_bitset<> & is_interesting)
{
	const KeypointVector& keypoints = features_2d.keypoints;
	bool do_3d_to_3d = !query_cloud.points.empty();

	// Proceed until we find some features matching a 3d configuration
	while (true)
	{
		// Those vector points are used for PnP stuff
		vector<cv::Point2f> image_points;
		vector<cv::Point3f> object_points;

		// Those clouds are use for 3d to 3d matching
		pcl::PointCloud<pcl::PointXYZ>::Ptr image_cloud(new pcl::PointCloud<pcl::PointXYZ>());
		image_cloud->is_dense = false;
		pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud(new pcl::PointCloud<pcl::PointXYZ>());
		object_cloud->is_dense = false;

		// Figure out which matches we are going to look at
		//TODO fixed BeS 19.04.2013 added Eigen::aligned_allocator<GuessMatch>
		std::vector<GuessMatch, Eigen::aligned_allocator<GuessMatch>> ordered_matches;
		for (unsigned int i = 0; i < in_matches.size(); ++i)
		{
			const cv::DMatch & match = in_matches[i];
			pcl::PointXYZRGB pcl_point;

			if (!is_interesting.test(i))
				continue;
			if (do_3d_to_3d)
			{
				// Make sure the query point cloud point is valid
				float scale_factor = float(query_cloud.width) / features_2d.camera.image_size.width; //use this to scale points by
				const cv::Point2f &point = keypoints[match.queryIdx].pt;
				float x = point.x * scale_factor;
				float y = point.y * scale_factor;
				if (y >= (int)query_cloud.height)
					continue;

				pcl_point = query_cloud(x, y);

				// Make sure it's not a NaN

				if ((boost::math::isnan(pcl_point.x)) || (boost::math::isnan(pcl_point.y)) || (boost::math::isnan(pcl_point.z)))
					continue;
				// Make sure it's within range
				if ((std::abs(pcl_point.x) >= 10) || (std::abs(pcl_point.y) >= 10) || (std::abs(pcl_point.z) >= 10)
					|| (std::abs(pcl_point.z) <= 0.01))
					continue;
			}
			ordered_matches.push_back(GuessMatch(&match, i, pcl::PointXYZ(pcl_point.x, pcl_point.y, pcl_point.z)));
		}

		// Don't waste your time if too few points are found
		if ((int)ordered_matches.size() < params.minInliersCount)
			break; //break out of look

		if (do_3d_to_3d)
			// Sort those matches (for PROSAC)
			std::sort(ordered_matches.begin(), ordered_matches.end(), GuessMatchComparator());

		// Get extra information concerning those matches
		std::vector<unsigned int> image_indices;
		BOOST_FOREACH(const GuessMatch & match_index, ordered_matches)
		{
			const cv::DMatch & match = *(match_index.match_);

			// add the 2d point to the list of points
			image_points.push_back(keypoints[match.queryIdx].pt);

			if (do_3d_to_3d)
				image_cloud->points.push_back(match_index.point_);

			// Add the 3d training point to the list of points
			{
				PoseRT pose = object->observations[match.imgIdx].camera().pose;

				cv::Point3d pt = object->observations[match.imgIdx].cloud()[match.trainIdx];
				std::vector<cv::Point3d> op(1, pt);
				cv::Mat mop(op);
				cv::Mat R;

				PoseRT inverted = Tools::invert(pose);

				cv::Rodrigues(inverted.rvec, R);

				//rotate the points in place
				cv::transform(mop, mop, R);

				//translate the points and add to pcl style point vector
				pt = op[0];
				//TODO fixed BeS 08.04.2013
				//cv::Point3d tmpP(inverted.tvec);
				//pt += inverted.tvec.at<cv::Point3d>();
				pt += cv::Point3d(inverted.tvec);

				object_points.push_back(pt);
				if (do_3d_to_3d)
					object_cloud->push_back(pcl::PointXYZ(pt.x, pt.y, pt.z));
			}

			// Add the index of the image
			image_indices.push_back(match.imgIdx);
		}

		// perform 2d to 3d or 3d to 3d matching
		vector<int> inliers;
		opencv_candidate::Pose pose;

		if (do_3d_to_3d)
		{
			// Remove bad points in the cloud
			std::vector<int> good_indices;
			for (unsigned int i = 0; i < image_cloud->points.size(); ++i)
				good_indices.push_back(good_indices.size());

			pcl::SampleConsensusModelRegistration<pcl::PointXYZ>::Ptr model(
				new pcl::SampleConsensusModelRegistration<
				pcl::PointXYZ>(object_cloud, good_indices));
			//pcl:RandomSampleConsensus<pcl::PointXYZ> sample_consensus(model);
			pcl::ProgressiveSampleConsensus<pcl::PointXYZ> sample_consensus(model);

			model->setInputTarget(image_cloud, good_indices);
			sample_consensus.setDistanceThreshold(0.01);
			sample_consensus.setMaxIterations(params.ransacIterationsCount);
			sample_consensus.computeModel();
			sample_consensus.getInliers(inliers);
			params.minInliersCount = 30;
			if ((int)inliers.size() >= params.minInliersCount)
			{
				Eigen::VectorXf coefficients;
				sample_consensus.getModelCoefficients(coefficients);

				cv::Mat_<float> R_mat(3, 3), tvec(3, 1);
				for (unsigned int j = 0; j < 3; ++j)
				{
					for (unsigned int i = 0; i < 3; ++i)
						R_mat(j, i) = coefficients[4 * j + i];
					tvec(j, 0) = coefficients[4 * j + 3];
				}
				pose.setR(R_mat);
				pose.setT(tvec);
			}
		}
		else
		{
			// Just use PnP
			cv::Mat rvec, tvec;
			//TODO fixed BeS 04.11.2012
			solvePnPRansac(object_points, image_points,
				features_2d.camera.K, features_2d.camera.D,
				rvec, tvec, false, params.ransacIterationsCount, params.maxProjectionError, -1, inliers);
			pose.setR(rvec);
			pose.setT(tvec);
		}

		if (((int)inliers.size() >= params.minInliersCount))
		{
			Guess guess(object, pose, features_2d.camera.K, features_2d.camera.D, features_2d.image);
			guess.inliers = inliers;
			guess.image_points_ = image_points;
			guess.aligned_points_ = object_points;
			guess.image_indices_ = image_indices;
			guesses.push_back(guess);

			// Make sure the inliers should not be checked for another object
			BOOST_FOREACH(unsigned int inlier_index, inliers)
				is_interesting.reset(ordered_matches[inlier_index].index_);
		}
		else
			break;
	}
}

void GuessGenerator::calculateGuesses(const Features2d& features_2d, const PointCloud &query_cloud,
	const cv::Ptr<TexturedObject>& object, const std::vector<cv::DMatch>& matches,
	std::vector<Guess>& guesses)
{
	guesses.clear();

	// Make sure we look at all the bits
	boost::dynamic_bitset<> is_interesting(matches.size());
	for (unsigned int i = 0; i < matches.size(); ++i)
		is_interesting.set(i);
	calculateGuesses(features_2d, query_cloud, object, matches, guesses, is_interesting);
}

void GuessGenerator::calculateGuesses(const cv::Ptr<TexturedObject>& object, const vector<vector<int> >& clusterIndices,
	const vector<cv::DMatch>& matches, const Features2d& test, vector<Guess>& guesses)
{
	guesses.clear();

	PointCloud empty_cloud;
	boost::dynamic_bitset<> is_interesting(matches.size());

	foreach(const vector<int>& cluster, clusterIndices)
	{
		if ((int)cluster.size() < params.minClusterSize)
			continue;

		// Only focus on the points from the cluster
		is_interesting.reset();
		for (unsigned int i = 0; i < cluster.size(); ++i)
			is_interesting.set(cluster[i]);

		calculateGuesses(test, empty_cloud, object, matches, guesses, is_interesting);
	}
}
