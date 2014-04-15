#define BOOST_ALL_DYN_LINK

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <tod/core/TexturedObject.h>
#include <tod/detecting/Loader.h>
#include <tod/training/feature_extraction.h>
#include <tod/detecting/Matcher.h>
#include <tod/detecting/Recognizer.h>
#include <tod/training/base_creation.h>
#include <tod/test/GuessTest.h>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>

#define foreach BOOST_FOREACH
namespace po = boost::program_options;

struct Options
{
	std::string baseDirectory;
	std::string testDirectory;
	std::string config;
	tod::TODParameters params;
	int verbos;
	bool generate;
	int scale;
};

int options(int argc, char ** argv, Options& opts)
{
	po::options_description desc("General options");
	desc.add_options()("help,H", "Produce help message.");
	desc.add_options()("base,B", po::value<std::string>(&opts.baseDirectory), "The directory that the training base is in.");
	desc.add_options()("config,C", po::value<std::string>(&opts.config), "The name of the configuration file.");
	desc.add_options()("verbos,V", po::value<int>(&opts.verbos)->default_value(3), "Verbosity level.");
	desc.add_options()("generate,G", po::value<bool>(&opts.generate)->default_value(false), "Generate base.");
	desc.add_options()("scale,S", po::value<int>(&opts.scale)->default_value(1), "Generate base.");
	desc.add_options()("tbase,T", po::value<std::string>(&opts.testDirectory), "The directory that the test masks is in.");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if (vm.count("help"))
	{
		std::cout << desc << "\n";
		return 1;
	}

	cv::FileStorage fs;
	if (opts.config.empty() || !(fs = cv::FileStorage(opts.config, cv::FileStorage::READ)).isOpened())
	{
		std::cout << "Must supply configuration. see newly generated sample.config.yaml" << "\n";
		std::cout << desc << std::endl;
		cv::FileStorage fs("./sample.config.yaml", cv::FileStorage::WRITE);
		fs << tod::TODParameters::YAML_NODE_NAME;
		tod::TODParameters::CreateSampleParams().write(fs);
		return 1;
	}
	else
	{
		opts.params.read(fs[tod::TODParameters::YAML_NODE_NAME]);
	}

	if ((!vm.count("base")) || (!vm.count("tbase")))
	{
		std::cout << "Must supply training or test base directory." << "\n";
		std::cout << desc << std::endl;
		return 1;
	}

	return 0;
}

int main(int argc, char** argv)
{
	Options opts;
	if( options(argc, argv, opts))
	{
		return 1;
	}

	cv::Ptr<tod::FeatureExtractor> extractor = tod::FeatureExtractor::create(opts.params.feParams);
	if(extractor.empty())
	{
		std::cout<<"Bad FeatureExtractorParams\n";
		return 1;
	}

	if(opts.generate)
	{
		tod::BaseCreator creator(opts.baseDirectory, extractor);
		creator.createBase(opts.scale);
		return 0;
	}

	tod::Loader loader(opts.baseDirectory);

	std::vector<cv::Ptr<tod::TexturedObject>> objects;
	loader.readTexturedObjects(objects);

	if(!objects.size())
	{
		std::cout<<"Empty base\n";
		return 1;
	}

	tod::TrainingBase base(objects);

	cv::Ptr<tod::Matcher> matcher = tod::Matcher::create(opts.params.matcherParams);
	matcher->add(base);
	if(matcher.empty())
	{
		std::cout<<"Bad MatcherParameters\n";
		return 1;
	}

	cv::Ptr<tod::Recognizer> recognizer;
	recognizer = new tod::TODRecognizer(&base, matcher, &opts.params.guessParams, opts.verbos, opts.baseDirectory, opts.params.clusterParams.maxDistance);

	std::ifstream fin(opts.testDirectory + "\\images\\images.txt");
	std::list<std::string> testNames = tod::getStringsList(fin);
	fin.close();
	std::ofstream fout(opts.baseDirectory + "out.txt");

	foreach(const std::string testName, testNames)
	{
		tod::Features2d test;

		test.image = cv::imread(opts.testDirectory + "\\images\\" + testName, 0);
		if(opts.scale > 1)
		{
			cv::resize(test.image, test.image, cv::Size(test.image.size().width/opts.scale, test.image.size().width/opts.scale));
		}
		extractor->detectAndExtract(test);
		std::cout<<"Extracted "<<test.keypoints.size()<<" points\n";

		std::vector<tod::Guess> foundObjects;
		recognizer->match(test, foundObjects);
		if(foundObjects.size() > 0)
		{
			cv::Mat guessMask(test.image.size(), CV_8UC1);
			tod::GuessTest gtest;
			std::vector<tod::GuessTestResult> testResult;
			gtest.markedGuess(foundObjects, opts.testDirectory, std::string(testName), opts.scale, testResult);
			foreach(tod::GuessTestResult& result, testResult)
			{
				fout<<"{"<<result.name<<":"<<result.overlapValue<<":"<<result.isInlier()<<"}";
				std::cout<<result.name<<": "<<result.overlapValue<<"-"<<result.isInlier()<<std::endl;
			}
			fout<<"\n";
		}
	}
	fout.close();

	cv::waitKey();

	return 0;
}