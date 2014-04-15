/*
* GuessQualifier.cpp
*
*  Created on: Jan 15, 2011
*      Author: Alexander Shishkov
*/
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

#include <tod/detecting/Tools.h>
#include <tod/detecting/GuessQualifier.h>
#include <tod/detecting/project3dPoints.h>

using namespace tod;
using namespace cv;
using namespace std;

GuessQualifier::GuessQualifier(const Ptr<TexturedObject>& texturedObject, const vector<DMatch>& objectMatches,
	const Features2d& test, const GuessGeneratorParameters& generatorParams) :
prevImageIdx(-2), object(texturedObject), matches(objectMatches), test(test), params(generatorParams)
{
}

GuessQualifier::~GuessQualifier()
{
}

void GuessQualifier::initPointsVector(vector<Point2f>& imagePoints,
	vector<Point3f>& objectPoints, vector<unsigned int>& imageIndices)
{
	imagePoints.resize(matches.size());
	objectPoints.resize(matches.size());
	imageIndices.resize(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		imagePoints[i] = test.keypoints[matches[i].queryIdx].pt;
		vector<Point3f> point, convertedPoint;
		int observationIndex = matches[i].imgIdx;
		imageIndices[i] = observationIndex;
		point.push_back(object->observations[observationIndex].cloud()[matches[i].trainIdx]);
		PoseRT pose = Tools::invert(object->observations[observationIndex].camera().pose);
		project3dPoints(point, pose.rvec, pose.tvec, convertedPoint);
		objectPoints[i] = convertedPoint[0];
	}
}

void GuessQualifier::clarifyGuess(Guess& guess, vector<Point2f>& imagePoints,
	vector<Point3f>& objectPoints, vector<unsigned int>& imageIndices)
{
	cout << "Guess: " << endl;
	cout << guess.aligned_pose().r() << endl << guess.aligned_pose().t() << " " << guess.inliers.size() << endl;

	vector<Point2f> projected_points;
	cv::Mat rvec, tvec = guess.aligned_pose().t<cv::Mat> ();
	cv::Rodrigues(guess.aligned_pose().r<cv::Mat> (), rvec);
	cv::projectPoints(cv::Mat(objectPoints), rvec, tvec, guess.getK(), guess.getD(), projected_points);

	vector<Point2f> inlierImagePoints;
	vector<Point3f> inlierObjectPoints;
	for (size_t j = 0; j < imagePoints.size(); j++)
	{
		float dist = norm(imagePoints[j] - projected_points[j]);
		if (dist < params.maxProjectionError * 5)
		{
			inlierImagePoints.push_back(imagePoints[j]);
			inlierObjectPoints.push_back(objectPoints[j]);
		}
	}
	if (inlierObjectPoints.size() >= 5)
	{
		vector<int> inliers;
		opencv_candidate::Pose pose = guess.aligned_pose();

		rvec.convertTo(rvec, CV_64F);
		tvec.convertTo(tvec, CV_64F);
		solvePnPRansac(inlierObjectPoints, inlierImagePoints,
			test.camera.K, test.camera.D,
			rvec, tvec, true, params.ransacIterationsCount * 10, params.maxProjectionError,
			params.minInliersCount, inliers);
		pose.setR(rvec);
		pose.setT(tvec);

		guess.set_aligned_pose(pose);
		guess.inliers = inliers;
		guess.image_points_ = inlierImagePoints;
		guess.aligned_points_ = inlierObjectPoints;
		guess.image_indices_ = imageIndices;
	}
	float stddev = Tools::computeStdDev(inlierObjectPoints);
	guess.stddev = stddev;
	cout << "After clarifying: " << endl;
	cout << guess.aligned_pose().r() << endl << guess.aligned_pose().t() << " " << guess.inliers.size() << endl << endl;
}

void GuessQualifier::clarify(vector<Guess>& guesses)
{
	if (!guesses.size())
		return;

	vector<Point2f> imagePoints;
	vector<Point3f> objectPoints;
	vector<unsigned int> imageIndices;

	initPointsVector(imagePoints, objectPoints, imageIndices);
	foreach(Guess& guess, guesses)
	{
		clarifyGuess(guess, imagePoints, objectPoints, imageIndices);
	}
}
