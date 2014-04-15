#include <tod/test/GuessTest.h>
#include <tod/detecting/Tools.h>
#include <tod/detecting/Loader.h>
#include <tod/detecting/project3dPoints.h>
#include <opencv_candidate/PoseRT.h>
#include <boost/foreach.hpp>
#include "boost/filesystem.hpp"

#define foreach BOOST_FOREACH
namespace bfs = boost::filesystem;

namespace tod
{
	std::string GuessTest::BASE_NAME = "objects.txt";

	GuessTest::GuessTest()
	{
	}

	GuessTest::~GuessTest()
	{
	}

	void GuessTest::generateGuessMask(const tod::Guess& obj, cv::Mat& binMask)
	{
		std::vector<cv::Point2f> projectedPoints;
		std::vector<cv::Point3f> rotatedPoints;
		std::vector<cv::Point3f> cloud;
		int cloudIndex = obj.image_indices_[0];

		PoseRT inverted = Tools::invert(obj.getObject()->observations[cloudIndex].camera().pose);
		project3dPoints(obj.getObject()->observations[cloudIndex].cloud(), inverted.rvec, inverted.tvec, cloud);
		cv::Mat rvec;
		cv::Rodrigues(obj.aligned_pose().r<cv::Mat> (), rvec);
		cv::projectPoints(cv::Mat(cloud), rvec, obj.aligned_pose().t<cv::Mat>(), obj.getK(), obj.getD(), projectedPoints); 

		binMask = cv::Scalar(0);
		for(std::vector<cv::Point2f>::iterator point = projectedPoints.begin(); point != projectedPoints.end();)
		{
			if (point->x >= binMask.cols || point->y >= binMask.rows || point->x < 0 || point->y < 0)
				point = projectedPoints.erase(point);
			else
			{
				point++;
			}
		}

		std::vector<cv::Point2f> hull;
		if(projectedPoints.size() > 0)
		{
			cv::convexHull(cv::Mat(projectedPoints), hull);
			std::vector<cv::Point> thull;
			for(int i = 0; i < hull.size(); i++)
			{
				thull.push_back(cv::Point(hull[i].x, hull[i].y));
			}
			cv::fillConvexPoly(binMask, cv::Mat(thull), cv::Scalar(255));
		}
	}

	double GuessTest::overlapValue(const cv::Mat& mask, const cv::Mat& guessMask)
	{
		cv::Mat intersectMask;
		cv::bitwise_and(mask, guessMask, intersectMask);

		cv::Mat unionMask;
		cv::bitwise_or(mask, guessMask, unionMask);

		int intersectArea = cv::countNonZero(intersectMask);
		int unionArea = cv::countNonZero(unionMask);

		return (double)intersectArea/(double)unionArea;
	}

	void GuessTest::generateBaseMasks(std::string& masksFolder, std::string& testImName, int scale)
	{
		std::ifstream fin(masksFolder + "\\" + BASE_NAME);
		std::list<std::string> masksNames = getStringsList(fin);
		fin.close();

		foreach(const std::string& name, masksNames)
		{
			if(baseMasks.find(name) == baseMasks.end())
			{
				baseMasks[name] = cv::imread( masksFolder + "\\" + name + "\\" + testImName + ".mask.png", 0);
				if(scale > 1)
				{
					cv::resize(baseMasks[name], baseMasks[name], cv::Size(baseMasks[name].size().width/scale, baseMasks[name].size().width/scale));
				}
			}
		}
	}

	void GuessTest::markedGuess(const std::vector<tod::Guess>& foundObjects, std::string& masksFolder, std::string& testImName, int scale, std::vector<tod::GuessTestResult>& testResult)
	{
		generateBaseMasks(masksFolder, testImName, scale);

		foreach(const tod::Guess& obj, foundObjects)
		{
			tod::GuessTestResult tmp;
			if(baseMasks.find(obj.getObject()->name) != baseMasks.end())
			{
				cv::Mat mask = baseMasks[obj.getObject()->name];
				cv::Mat guessMask(mask.size(), CV_8UC1);
				generateGuessMask(obj, guessMask);
				tmp.name = obj.getObject()->name;
				tmp.overlapValue = overlapValue(mask, guessMask);
			}
			else
			{
				tmp.name = obj.getObject()->name;
				tmp.overlapValue = 0;
			}
			testResult.push_back(tmp);
		}
	}

}/*namespace tod*/