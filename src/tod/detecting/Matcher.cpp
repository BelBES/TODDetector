/*
* Matcher.cpp
*
*  Created on: Dec 15, 2010
*      Author: alex
*/

//TODO fixed BeS 04.11.2012 removed rbrief, lsh, flann

#include <tod/detecting/Matcher.h>
#include <boost/foreach.hpp>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/highgui/highgui_c.h>

//#include "rbrief/lsh.hpp"

#define foreach BOOST_FOREACH

using namespace std;
using namespace cv;
using namespace tod;

namespace
{
	typedef std::vector<cv::DMatch> MatchesVector;
	struct QueryIdx
	{
		inline int operator()(const DMatch& m) const
		{
			return m.queryIdx;
		}
	};

	struct TrainIdx
	{
		inline int operator()(const DMatch& m) const
		{
			return m.trainIdx;
		}
	};

	/** Given two matches, will sort by index, and then if indices are equal, will sort by distance.
	*  ascending
	*/
	template<class Idx>
	struct CompareOpIdx
	{
		Idx idx;
		inline bool operator()(const DMatch& lhs, const DMatch& rhs) const
		{
			return idx(lhs) < idx(rhs) || (idx(lhs) == idx(rhs) && lhs.distance < rhs.distance);
		}
	};

	template<class CompareOp>
	inline void uniqueMatches(const MatchesVector& matches, MatchesVector& output, const CompareOp& op)
	{
		MatchesVector t_matches = matches;
		std::sort(t_matches.begin(), t_matches.end(), op);
		int prev_index = -1;
		output.clear();
		output.reserve(t_matches.size());
		for (size_t i = 0; i < t_matches.size(); i++)
		{
			int index = op.idx(t_matches[i]);
			if (prev_index != index)
			{
				output.push_back(t_matches[i]);
				prev_index = index;
			}
		}
	}

	/** this will get rid of matches where there are many to one keypoint correspondences.
	*  The
	*/
	void uniqueMatches(const MatchesVector& matches, MatchesVector& output)
	{
		uniqueMatches(matches, output, CompareOpIdx<QueryIdx> ());
		uniqueMatches(output, output, CompareOpIdx<TrainIdx> ());
	}

	void flattenKNN(const vector<tod::Matches>& matches, tod::Matches& out)
	{
		out.clear();
		size_t count = 0;
		for (size_t i = 0; i < matches.size(); i++)
		{
			count += matches[i].size();
		}
		out.reserve(count);
		for (size_t i = 0; i < matches.size(); i++)
		{
			out.insert(out.end(), matches[i].begin(), matches[i].end());
		}

	}

}

void Matcher::drawMatches(const TrainingBase& base, const cv::Ptr<Matcher> matcher, const cv::Mat& testImage,
	const KeypointVector& testKeypoints, std::string directory)
{
	return;
	namedWindow("matches", CV_WINDOW_KEEPRATIO);
	std::vector<cv::Mat> match_images;
	int scaled_width = 500;
	int scaled_height = 0;

	// Build the individual matches
	for (size_t objectInd = 0; objectInd < base.size(); objectInd++)
	{
		for (size_t imageInd = 0; imageInd < base.getObject(objectInd)->observations.size(); imageInd++)
		{
			tod::Matches imageMatches;
			matcher->getImageMatches(objectInd, imageInd, imageMatches);
			if (imageMatches.size() > 7)
			{
				Features3d f3d = base.getObject(objectInd)->observations[imageInd];
				Features2d f2d = f3d.features();
				if (f2d.image.empty())
				{
					string filename = directory + "/" + base.getObject(objectInd)->name + "/" + f2d.image_name;
					f2d.image = imread(filename, 0);
				}
				Mat matchesView;
				cv::drawMatches(testImage, testKeypoints, f2d.image,
					base.getObject(objectInd)->observations[imageInd].features().keypoints, imageMatches,
					matchesView, Scalar(255, 0, 0), Scalar(0, 0, 255));
				// Resize the individual image
				cv::Size smaller_size(scaled_width, (matchesView.rows * scaled_width) / matchesView.cols);
				cv::Mat resize_match;
				cv::resize(matchesView, resize_match, smaller_size);
				// Keep track of the max height of each image
				scaled_height = std::max(scaled_height, smaller_size.height);

				// Add the image to the big_image
				match_images.push_back(resize_match);
			}
		}
	}
	// Display a big image with all the correspondences
	unsigned int big_x = 0, big_y = 0;
	unsigned int n_match_image = 0;
	while (n_match_image < match_images.size())
	{
		if (1.5 * big_x < big_y)
		{
			big_x += scaled_width;
			n_match_image += big_y / scaled_height;
		}
		else
		{
			big_y += scaled_height;
			n_match_image += big_x / scaled_width;
		}
	}
	cv::Mat_<cv::Vec3b> big_image = cv::Mat_<cv::Vec3b>::zeros(big_y, big_x);
	int y = 0, x = 0;
	BOOST_FOREACH(const cv::Mat & match_image, match_images)
	{
		cv::Mat_<cv::Vec3b> sub_image = big_image(cv::Range(y, y + scaled_height), cv::Range(x, x + scaled_width));
		match_image.copyTo(sub_image);
		x += scaled_width;
		if (x >= big_image.cols)
		{
			x = 0;
			y += scaled_height;
		}
	}
	if (!big_image.empty())
		imshow("matches", big_image);
}

Matcher* Matcher::create(MatcherParameters& params)
{
	Matcher* resultMatcher;

	cv::Ptr<DescriptorMatcher> matcher;
	//if (params.type == "FLANN")
	//{
	//	// 2 should be cvflann::FLANN_DIST_MANHATTAN or cvflann::MANHATTAN but we can't check
	//	//cv::flann::set_distance_type(static_cast<cvflann::flann_distance_t>(2), 0);
	//	matcher = new FlannBasedMatcher(new cv::flann::KDTreeIndexParams(4), new cv::flann::SearchParams(64));
	//}
	//else if (params.type == "BF")
	if (params.type == "BF")
	{
		//TODO fixed BeS 29.10.2012
		matcher = cv::DescriptorMatcher::create("BruteForce-L1");
		//matcher = new cv::BruteForceMatcher<L1<float> > ();
	}
	else if (params.type == "BF-BINARY")
	{
		//TODO fixed BeS 29.10.2012
		matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		//matcher = new BruteForceMatcher<Hamming> ();
	}
	else if (params.type == "FLANN-BINARY")
	{
		throw std::logic_error("FLANN-BINARY not implemented :(");
	}
	// else if (params.type == "LSH-BINARY")
	// {
	//   lsh::LshMatcher* lsh_matcher = new lsh::LshMatcher();
	//   //TODO fixed BeS 29.10.2012
	//lsh_matcher->setDimensions(5,24);
	////lsh_matcher->setDimensions(5, 24, 2);
	//   matcher = lsh_matcher;
	// }
	else
	{
		throw std::logic_error(params.type + " not implemented :(");
		return NULL;
	}

	if (params.doRatioTest)
	{
#ifndef TOD_MATCHER
		resultMatcher = new RatioTestMatcher(matcher, params.knn, params.ratioThreshold);
#else
		resultMatcher = new FlannRatioTestMatcher(params.knn, params.ratioThreshold);
#endif
	}
	else
		resultMatcher = new Matcher(matcher, params.knn);

	return resultMatcher;
}

Matcher::Matcher(const Ptr<DescriptorMatcher> matcher, int knn) :
matcher(matcher), knn_(knn)
{
}

Matcher::~Matcher()
{
}

void Matcher::add(const TrainingBase& base)
{
	objectSizes.clear();
	objectSizes.resize(base.size());

	vector<int> objectIndices;
	base.getObjectIds(objectIndices);
	for (size_t objInd = 0; objInd < base.size(); objInd++)
	{
		matcher->add(base.getObject(objectIndices[objInd])->getDescriptors());

		int prevSize = 0;
		if (objInd)
			prevSize = objectSizes[objInd - 1];
		objectSizes[objInd] = base.getObject(objectIndices[objInd])->observations.size() + prevSize;
	}

	base.getObjectIds(objectIds);
}

void Matcher::match(const Mat& queryDescriptors)
{
	tod::Matches matches;
	privateMatch(queryDescriptors, matches);
	convertMatches(matches);
}

void Matcher::getObjectMatches(int objectId, tod::Matches& matches) const
{
	matches.clear();

	for (size_t objInd = 0; objInd < objectIds.size(); objInd++)
	{
		if (objectId == objectIds[objInd])
		{
			matches = objectMatches[objInd];
			break;
		}
	}
}

void Matcher::getLabelSizes(std::vector<std::pair<int, int> >& labels_sizes) const
{

	labels_sizes.clear();
	labels_sizes.reserve(objectIds.size());
	for (size_t objInd = 0; objInd < objectIds.size(); objInd++)
	{
		labels_sizes.push_back(std::make_pair(objectIds[objInd], objectMatches[objInd].size()));
	}
	std::sort(labels_sizes.begin(), labels_sizes.end(), pair_second_greater<std::pair<int, int> > );
}

void Matcher::getImageMatches(int objectId, int imageId, std::vector<cv::DMatch>& imageMatches) const
{
	imageMatches.clear();

	tod::Matches matches;
	getObjectMatches(objectId, matches);

	foreach(DMatch& match, matches)
	{
		if (match.imgIdx == imageId)
			imageMatches.push_back(match);
	}
}

int Matcher::getObjectIndex(const DMatch& match)
{
	for (size_t objInd = 0; objInd < objectSizes.size(); objInd++)
	{
		if (match.imgIdx < objectSizes[objInd])
		{
			return (int)objInd;
		}
	}
	return -1;
}

void Matcher::convertMatches(const tod::Matches& matches)
{
	objectMatches.clear();
	objectMatches.resize(objectIds.size());

	for (size_t matchInd = 0; matchInd < matches.size(); matchInd++)
	{
		DMatch match = matches[matchInd];
		int objectIndex = getObjectIndex(match);

		if (objectIndex)
			match.imgIdx -= objectSizes[objectIndex - 1];

		objectMatches[objectIndex].push_back(match);
	}
	for (size_t i = 0; i < objectMatches.size(); i++)
	{
		//keep, the matches unique on the training index side
		uniqueMatches(objectMatches[i], objectMatches[i], CompareOpIdx<TrainIdx> ());
		std::sort(objectMatches[i].begin(), objectMatches[i].end()); //keep the matches sorted by distance
	}
}

void Matcher::privateMatch(const Mat& queryDescriptors, tod::Matches& matches)
{
	vector<tod::Matches> knnMatches;
	matcher->knnMatch(queryDescriptors, knnMatches, knn_);
	flattenKNN(knnMatches, matches);
}

RatioTestMatcher::RatioTestMatcher(const Ptr<DescriptorMatcher> matcher_, int knnNum_, double ratioThreshold_) :
Matcher(matcher_)
{
	assert(knnNum_ > 1);
	assert(ratioThreshold_ < 1.f && ratioThreshold_ > 0.f);

	knnNum = knnNum_;
	ratioThreshold = ratioThreshold_;
}

RatioTestMatcher::~RatioTestMatcher()
{
}

void RatioTestMatcher::match(const Mat& queryDescriptors)
{
	vector<tod::Matches> knnMatches;
	matcher->knnMatch(queryDescriptors, knnMatches, knnNum);

	objectMatches.clear();
	objectMatches.resize(objectIds.size());
	for (int descInd = 0; descInd < queryDescriptors.rows; descInd++)
	{
		const std::vector<cv::DMatch> & matches = knnMatches[descInd];

		// In the ratio test, we will compare the quality of a match with the next match that is not from the same object:
		// we can accept several matches with similar scores as long as they are for the same object. Those should not be
		// part of the model anyway as they are not discriminative enough
		for (unsigned int first_index = 0; first_index < matches.size(); ++first_index)
		{
			int object_index = getObjectIndex(matches[first_index]);
			unsigned int second_index = first_index + 1;
			while ((second_index < matches.size()) && (object_index == getObjectIndex(matches[second_index])))
				++second_index;

			// Perform the ratio test
			if ((second_index == matches.size()) || (matches[first_index].distance < ratioThreshold
				* matches[second_index].distance))
			{
				DMatch match = matches[first_index];
				/*fixed BeS 6.08.2013
				******************************
				******old version***********
				******************************
				if (object_index)
					match.imgIdx -= objectSizes[object_index - 1];
				objectMatches[object_index].push_back(match);
				******************************
				*/
				if (object_index)
					match.imgIdx -= objectSizes[object_index - 1];
				objectMatches[object_index].push_back(match);
			}
		}
	}

	matchPruner();
}

/** Go over the different matches and remove the ones involving features matching to many.
* E.g., we have a test image with a checkerboard where all corners match one unique feature in the training image.
* Those a re obviously bad matches for geometrical constraints and need to be removed
* @param maximum_match_number maximum allowed number of matches from a feature to the original image
*/
void RatioTestMatcher::matchPruner(unsigned int maximum_match_number)
{
	BOOST_FOREACH(tod::Matches &matches, objectMatches)
	{
		typedef std::map<unsigned int, unsigned int> FeatureCount;
		FeatureCount feature_count;
		// Count how many times each feature is used
		BOOST_FOREACH(const cv::DMatch& match, matches)
		{
			if (feature_count.find(match.trainIdx) == feature_count.end())
				feature_count[match.trainIdx] = 1;
			else
				++feature_count[match.trainIdx];
		}

		// Remove the bad features
		tod::Matches clean_matches;
		clean_matches.reserve(matches.size());
		BOOST_FOREACH(const cv::DMatch& match, matches)
		{
			if (feature_count[match.trainIdx] <= maximum_match_number)
				clean_matches.push_back(match);
		}
		// Replace the original matches with the clean ones
		matches = clean_matches;
	}
}

void FlannRatioTestMatcher::add(const TrainingBase& base)
{
	base.getObjectIds(objectIds);

	objectIndices.clear();
	imageIndices.clear();
	keypointIndices.clear();

	int rowsNumber = 0;
	for (size_t objInd = 0; objInd < objectIds.size(); objInd++)
	{
		int objId = objectIds[objInd];
		for (size_t imgInd = 0; imgInd < base.getObject(objId)->observations.size(); imgInd++)
		{
			rowsNumber += base.getObject(objId)->observations[imgInd].features().descriptors.rows;
		}
	}
	descriptors.create(rowsNumber, base.getObject(0)->observations[0].features().descriptors.cols, CV_32FC1);
	objectIndices.resize(rowsNumber);
	imageIndices.resize(rowsNumber);
	keypointIndices.resize(rowsNumber);

	int rowIndex = 0;
	for (size_t objInd = 0; objInd < base.size(); objInd++)
	{
		int objId = objectIds[objInd];
		for (size_t imgInd = 0; imgInd < base.getObject(objId)->observations.size(); imgInd++)
		{
			for (int k = 0; k < base.getObject(objId)->observations[imgInd].features().descriptors.rows; k++)
			{
				objectIndices[rowIndex] = objInd;
				imageIndices[rowIndex] = imgInd;
				keypointIndices[rowIndex] = k;

				Mat row(1, descriptors.cols, descriptors.type(), descriptors.ptr(rowIndex++));
				base.getObject(objId)->observations[imgInd].features().descriptors.row(k).copyTo(row);
			}
		}
	}
	flann_index = new cv::flann::Index(descriptors, cv::flann::KDTreeIndexParams(4));
}

FlannRatioTestMatcher::FlannRatioTestMatcher(int knnNum_, double ratioThreshold_) :
knnNum(knnNum_), ratioThreshold(ratioThreshold_)
{
	flann_index = NULL;
}

FlannRatioTestMatcher::~FlannRatioTestMatcher()
{
	if (flann_index)
		delete flann_index;
}

void FlannRatioTestMatcher::match(const Mat& queryDescriptors)
{
	//objectMatches.clear();
	//objectMatches.resize(objectIds.size());

	//// 2 should be cvflann::FLANN_DIST_MANHATTAN or cvflann::MANHATTAN but we can't check
	////cvflann::set_distance_type(static_cast<cvflann::flann_distance_t>(2), 0);
	//Mat m_indices(queryDescriptors.rows, knnNum, CV_32S);
	//Mat m_dists(queryDescriptors.rows, knnNum, CV_32F);
	//flann_index->knnSearch(queryDescriptors, m_indices, m_dists, knnNum, cv::flann::SearchParams(64));

	//int* indices_ptr = m_indices.ptr<int> (0);
	//float* dists_ptr = m_dists.ptr<float> (0);
	//int first_index, second_index, j;

	//for (int i = 0; i < m_indices.rows; ++i)
	//{
	//	for (size_t objId = 0; objId < objectIds.size(); objId++)
	//	{
	//		first_index = second_index = -1;
	//		for (j = 0; j < knnNum; j++)
	//		{
	//			if (objectIndices[indices_ptr[i * knnNum + j]] == (int)objId)
	//			{
	//				first_index = j++;
	//				break;
	//			}
	//		}

	//		if (first_index < 0)
	//			continue;

	//		for (; j < knnNum; j++)
	//		{
	//			if (objectIndices[indices_ptr[i * knnNum + j]] == (int)objId)
	//			{
	//				second_index = j;
	//				break;
	//			}
	//		}

	//		if (second_index == -1)
	//		{
	//			if (first_index < knnNum - 1)
	//				second_index = first_index + 1;
	//			else
	//				second_index = first_index;
	//		}

	//		if (dists_ptr[knnNum * i + first_index] < ratioThreshold * dists_ptr[knnNum * i + second_index])
	//		{
	//			DMatch match;
	//			match.queryIdx = i;
	//			//fixed error with index in DMatch
	//			match.trainIdx = keypointIndices[indices_ptr[i * knnNum + first_index]];
	//			match.imgIdx = imageIndices[indices_ptr[i * knnNum + first_index]];
	//			objectMatches[objectIndices[indices_ptr[i * knnNum + first_index]]].push_back(match);
	//		}
	//	}
	//}
}

