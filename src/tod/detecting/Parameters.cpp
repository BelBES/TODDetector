/*
* Parameters.cpp
*
*  Created on: Dec 15, 2010
*      Author: alex
*/

#include <tod/detecting/Parameters.h>

using namespace tod;
using namespace cv;

void ClusterParameters::write(FileStorage& fs) const
{
	cvWriteComment(*fs, "ClusterParameters", 0);
	fs << "{";
	fs << "max_distance" << maxDistance;
	fs << "}";
}

void ClusterParameters::read(const FileNode& fn)
{
	maxDistance = (int)fn["max_distance"];
}

const std::string ClusterParameters::YAML_NODE_NAME = "ClusterParameters";

void MatcherParameters::write(FileStorage& fs) const
{
	cvWriteComment(*fs, "MatcherParameters", 0);
	fs << "{";
	fs << "matcher_type" << type << "ratio_threshold" << ratioThreshold << "knn" << knn;
	if (doRatioTest)
		fs << "do_ratio_test" << 1;
	else
		fs << "do_ratio_test" << 0;
	fs << "}";
}

void MatcherParameters::read(const cv::FileNode& fn)
{
	type = (std::string)fn["matcher_type"];
	ratioThreshold = (float)fn["ratio_threshold"];
	knn = (int)fn["knn"];
	doRatioTest = (int)fn["do_ratio_test"];
}

const std::string MatcherParameters::YAML_NODE_NAME = "MatcherParameters";

void TODParameters::write(FileStorage& fs) const
{
	cvWriteComment(*fs, "TODParameters", 0);
	fs << "{";
	fs << GuessGeneratorParameters::YAML_NODE_NAME;
	guessParams.write(fs);
	fs << FeatureExtractionParams::YAML_NODE_NAME;
	feParams.write(fs);
	fs << MatcherParameters::YAML_NODE_NAME;
	matcherParams.write(fs);
	fs << ClusterParameters::YAML_NODE_NAME;
	clusterParams.write(fs);
	fs << "}";
}

void TODParameters::read(const cv::FileNode& fn)
{
	guessParams.read(fn[GuessGeneratorParameters::YAML_NODE_NAME]);
	feParams.read(fn[FeatureExtractionParams::YAML_NODE_NAME]);
	matcherParams.read(fn[MatcherParameters::YAML_NODE_NAME]);
	clusterParams.read(fn[ClusterParameters::YAML_NODE_NAME]);
}

const std::string TODParameters::YAML_NODE_NAME = "TODParameters";

TODParameters TODParameters::CreateSampleParams()
{
	TODParameters params;
	params.clusterParams.maxDistance = 100;
	params.matcherParams.type = "FLANN";
	params.matcherParams.knn = 30;
	params.matcherParams.ratioThreshold = 0.9;
	params.matcherParams.doRatioTest = true;
	params.guessParams.maxProjectionError = 4.0;
	params.guessParams.minClusterSize = 10;
	params.guessParams.minInliersCount = 10;
	params.guessParams.ransacIterationsCount = 100;
	params.feParams = FeatureExtractionParams::CreateSampleParams();

	return params;
}
