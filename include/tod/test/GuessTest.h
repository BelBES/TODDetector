#ifndef __GUESS_TEST_H__
#define __GUESS_TEST_H__

#include <iostream>
#include <map>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <tod/detecting/GuessGenerator.h>

namespace tod
{
	#define OVERLAP_THRESHOLD 0.5

	struct GuessTestResult
	{
		std::string name;
		double overlapValue;
		bool isInlier()
		{
			if(overlapValue > OVERLAP_THRESHOLD)
			{
				return true;
			}
			return false;
		}
	};

	class GuessTest
	{
	public:
		GuessTest();
		virtual ~GuessTest();
		void markedGuess(const std::vector<tod::Guess>& foundObjects, std::string& masksFolder, std::string& testImName, int scale, std::vector<tod::GuessTestResult>& testResult);
	private:
		void generateGuessMask(const tod::Guess& obj, cv::Mat& mask);
		double overlapValue(const cv::Mat& mask, const cv::Mat& guessMask);
		void generateBaseMasks(std::string& masksFolder, std::string& testImName, int scale);

		std::map<std::string, cv::Mat> baseMasks;
		static std::string BASE_NAME;
	};

}/*namespace tod*/

#endif /*GuessTest.h*/