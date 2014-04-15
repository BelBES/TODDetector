/*
* Recognizer.cpp
*
*  Created on: Feb 4, 2011
*      Author: Alexander Shishkov
*/

//TODO fixed BeS 04.11.2012 removed the reference to the ROS

#include <boost/unordered_set.hpp>
#include <boost/foreach.hpp>

#include <tod/detecting/Recognizer.h>
#include <tod/detecting/Cluster.h>
#include <tod/detecting/Filter.h>
#include <tod/detecting/GuessQualifier.h>

#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/types_c.h>

#define foreach BOOST_FOREACH
typedef std::pair<int, int> idx_pair_t;

using namespace tod;
using namespace cv;
using namespace std;

Recognizer::Recognizer()
{
	base = NULL;
	params = NULL;
	matcher = NULL;
	verbose = 0;
}

Recognizer::Recognizer(TrainingBase* base_, cv::Ptr<Matcher> matcher_, GuessGeneratorParameters* params_, int verbose_)
{
	base = base_;
	matcher = matcher_;
	params = params_;
	verbose = verbose_;
}

Recognizer::~Recognizer()
{
	base = NULL;
	params = NULL;
	matcher = NULL;
}

KinectRecognizer::KinectRecognizer(TrainingBase* base, cv::Ptr<Matcher> matcher, GuessGeneratorParameters* params,
	int verbose, string baseDirectory_) :
Recognizer(base, matcher, params, verbose), baseDirectory(baseDirectory_)
{
	//  ros::Time::init();
}

KinectRecognizer::~KinectRecognizer()
{
}

bool compareGreaterDMatch(const DMatch & lhs, const DMatch & rhs)
{
	return lhs.imgIdx > rhs.imgIdx; //  && lhs.imgIdx == rhs.imgIdx ? lhs.distance < rhs.distance : true;
}

void KinectRecognizer::match(const Features2d& features_2d, std::vector<Guess>& in_guesses)
{
	//  ros::Time n = ros::Time::now();

//	std::cout<<"Matcher::match\n";
	matcher->match(features_2d.descriptors);
//	std::cout<<"Matcher::match finish\n";
	//  if (verbose >= 1)
	//  {
	//    ROS_INFO_STREAM ("LSH matching :" << (ros::Time::now () - n).toSec ());
	//  }
	if (verbose >= 2)
		Matcher::drawMatches(*base, matcher, features_2d.image, features_2d.keypoints, baseDirectory);

	const PointCloud cloud;
	match_common(features_2d, cloud, /*n,*/ in_guesses);
}

void KinectRecognizer::match(const Features2d& features_2d, const PointCloud &query_cloud,
	std::vector<Guess>& in_guesses)
{
	//ros::Time n = ros::Time::now();

	matcher->match(features_2d.descriptors);

	//if (verbose >= 1)
	//  ROS_INFO_STREAM ("LSH matching :" << (ros::Time::now () - n).toSec ());

	if (verbose >= 2)
		Matcher::drawMatches(*base, matcher, features_2d.image, features_2d.keypoints, baseDirectory);

	match_common(features_2d, query_cloud, /*n,*/ in_guesses);
}

//void KinectRecognizer::match_common(const Features2d& features_2d, const PointCloud &query_cloud, ros::Time n,
//                                    std::vector<Guess>& in_guesses)
void KinectRecognizer::match_common(const Features2d& features_2d, const PointCloud &query_cloud,
	std::vector<Guess>& in_guesses)
{
	in_guesses.clear();

	// Order the objects by their number of guesses
	//n = ros::Time::now();
	std::vector<std::pair<int, int> > labels_sizes;
	matcher->getLabelSizes(labels_sizes);

	std::vector<DMatch> object_matches;
	boost::unordered_set<unsigned int> used_indices;
	foreach (const idx_pair_t & x, labels_sizes)
	{
		if (x.second < params->minInliersCount)
			break;
		matcher->getObjectMatches(x.first, object_matches);

		// Remove the features we've already used
		{
			std::vector<DMatch>::iterator good_match = object_matches.begin(), match = object_matches.begin(),
				end_match = object_matches.end();
			for (; match != end_match;)
			{
				// Keep the index if we never used before
				if (used_indices.find(match->queryIdx) == used_indices.end())
					*(good_match++) = *(match++);
				else
					++match;
			}
			object_matches.erase(good_match, end_match);
		}

		// Figure out guesses for that object
		std::vector<Guess> guesses;
		GuessGenerator generator(*params);
		generator.calculateGuesses(features_2d, query_cloud, base->getObject(x.first), object_matches, guesses);

		in_guesses.insert(in_guesses.end(), guesses.begin(), guesses.end());

		// Make sure we remember the features that have been used
		BOOST_FOREACH (const Guess & guess, guesses)
			BOOST_FOREACH(unsigned int index, guess.image_indices_)
			used_indices.insert(index);
	}

	// Do different display stuff from now on
	//if (verbose >= 1)
	//ROS_INFO_STREAM("pose fitting :" << (ros::Time::now() - n).toSec());

	if ((!in_guesses.empty()) && (verbose >= 1))
	{
		cv::Mat correspondence;
		cv::Mat projection;
		if (verbose >= 3)
			namedWindow("correspondence", CV_WINDOW_KEEPRATIO);
		foreach (const Guess & guess, in_guesses)
		{
			if (verbose >= 1)
				guess.draw(projection, 0, ".");
			if (verbose >= 3)
			{
				guess.draw(correspondence, 1, baseDirectory);
				if (!correspondence.empty())
				{
					imshow("correspondence", correspondence);
					waitKey(0);
				}
			}
		}

		if (!projection.empty() && (verbose >= 1))
		{
			namedWindow("projection", CV_WINDOW_KEEPRATIO);
			imshow("projection", projection);
			if (verbose >= 3)
				waitKey(0);
			else
				waitKey(30);
		}
	}
}

TODRecognizer::TODRecognizer(TrainingBase* base, cv::Ptr<Matcher> matcher, GuessGeneratorParameters* params,
	int verbose, string baseDirectory_, float maxDistance_) :
Recognizer(base, matcher, params, verbose), baseDirectory(baseDirectory_), maxDistance(maxDistance_)
{
}

TODRecognizer::~TODRecognizer()
{
}

void TODRecognizer::printMatches(vector<int>& objectIds)
{
	int matchesSize = 0;
	foreach(int id, objectIds)
	{
		vector<DMatch> objectMatches;
		matcher->getObjectMatches(id, objectMatches);
		matchesSize += objectMatches.size();
	}

	cout << "Total matches: " << matchesSize << endl;

	int maxMatch = 0, maxMatchObj = -1;
	cout << "Matches for each object (row) per view (column)" << endl;
	foreach(int id, objectIds)
	{
		cout << "Object " << id << ":" << "\t";
		const Ptr<TexturedObject>& object = base->getObject(id);
		for (size_t obsInd = 0; obsInd < object->observations.size(); obsInd++)
		{
			vector<DMatch> obsMatches;
			matcher->getImageMatches(id, obsInd, obsMatches);
			cout << obsMatches.size() << "\t";
			if (maxMatch < (int)obsMatches.size())
			{
				maxMatch = obsMatches.size();
				maxMatchObj = id;
			}
		}
		cout << endl;
	}
	cout << "Max match is " << maxMatch << " for object " << maxMatchObj << endl;
}

void TODRecognizer::drawProjections(Mat& image, int id, const vector<Guess>& guesses, const string& baseDirectory)
{
	if (guesses.empty())
		return;

	foreach(const Guess& guess, guesses)
	{
		//guess.draw(image, 0, ".");
		//guess.draw(image, 0, baseDirectory);
		guess.draw(image, 0, baseDirectory);
	}
}

void TODRecognizer::match(const Features2d& test, std::vector<Guess>& objects)
{
	objects.clear();

	matcher->match(test.descriptors);
	if (verbose == 2)
	{
		Matcher::drawMatches(*base, matcher, test.image, test.keypoints, baseDirectory);
	}

	vector<int> objectIds;
	base->getObjectIds(objectIds);

	printMatches(objectIds);

	Mat drawImage;
	if (verbose)
	{
		if (test.image.channels() > 1)
			test.image.copyTo(drawImage);
		else
			cvtColor(test.image, drawImage, CV_GRAY2BGR);
	}

	StdDevFilter stdFilter(*params);
	OverlappingFilter overlappingFilter;
	CloudFilter cloudFilter(test.image.cols, test.image.rows);

	foreach(int id, objectIds)
	{
		vector<DMatch> objectMatches;
		matcher->getObjectMatches(id, objectMatches);

		vector<vector<int> > clusterIndices;
		vector<int> imgIndices;
		vector<Point2f> points;
		for (size_t k = 0; k < objectMatches.size(); k++)
		{
			points.push_back(test.keypoints[objectMatches[k].queryIdx].pt);
			imgIndices.push_back((int)objectMatches[k].imgIdx);
		}
		ClusterBuilder clusterBuilder(maxDistance);
		clusterBuilder.clusterPoints(points, imgIndices, clusterIndices);
		if (verbose == 2)
			drawClusters(test.image.clone(), points, clusterIndices);

		GuessGenerator generator(*params);
		vector<Guess> guesses;
		const Ptr<TexturedObject>& object = base->getObject(id);
		Features2d f2d(test);
		if (f2d.camera.K.empty())
			f2d.camera = object->observations[0].camera();
		generator.calculateGuesses(object, clusterIndices, objectMatches, f2d, guesses);

		GuessQualifier qualifier(object, objectMatches, f2d, *params);
		qualifier.clarify(guesses);
		stdFilter.filterGuesses(guesses);
		overlappingFilter.filterGuesses(guesses);
		cloudFilter.filterGuesses(guesses);

		foreach(const Guess& guess, guesses)
		{
			objects.push_back(guess);
		}
		if (verbose) //returned this equations, because we show this image when verbose > 0 (see condition below)
		{
			if (guesses.size() > 0)
			{
				drawProjections(drawImage, id, guesses, baseDirectory);
			}
		}
	}

	if (verbose)
	{
		namedWindow("projection", CV_WINDOW_KEEPRATIO);
		imshow("projection", drawImage);
#ifdef TOD_MATCHER
		waitKey(0);
#else
		waitKey(30);
#endif
	}
}

void TODRecognizer::drawClusters(const cv::Mat img, const std::vector<cv::Point2f>& imagePoints,
	const vector<vector<int> >& indices)
{
	string clusterWinName = "clusters";
	namedWindow(clusterWinName, CV_WINDOW_AUTOSIZE);
	//cout << "NumClusters = " << indices.size() << endl;
	vector<Scalar> colors(indices.size());
	for (size_t k = 0; k < indices.size(); k++)
		colors[k] = CV_RGB(rand() % 255, rand() % 255, rand() % 255);
	Size size(img.cols, img.rows);
	Mat drawImg(size, CV_MAKETYPE(img.depth(), 3));
	drawImg.setTo(Scalar::all(0));
	Mat drawImg1 = drawImg(Rect(0, 0, img.cols, img.rows));
	cvtColor(img, drawImg1, CV_GRAY2RGB);

	for (size_t j = 0; j < indices.size(); j++)
	{
		for (size_t k = 0; k < indices[j].size(); k++)
			circle(drawImg, Point(imagePoints[indices[j][k]].x, imagePoints[indices[j][k]].y), 7, colors[j], 2);
	}

	Mat smallDrawImg;
	resize(drawImg, smallDrawImg, Size(), 0.3, 0.3);
	imshow(clusterWinName, smallDrawImg);
	waitKey(0);
}
