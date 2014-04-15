#include <tod/training/base_creation.h>
#include <tod/training/file_io.h>
#include <tod/detecting/Tools.h>
#include <tod/detecting/Loader.h>
#include <tod/detecting/project3dPoints.h>
#include <tod/training/clouds.h>
#include <boost/filesystem.hpp>
#include <opencv_candidate/PoseRT.h>
#include <limits>

#define filename(arg) (directory+"/"+arg)
#define foreach BOOST_FOREACH

namespace tod
{
	std::string BaseCreator::CONFIG_NAME = "config.txt";
	std::string BaseCreator::IMAGES_LIST = "images.txt";
	std::string BaseCreator::CLOUDS_LIST = "pcds.txt";

	std::list<std::pair<std::string, std::string>> getStringsList(std::istream& images, std::ifstream& clouds)
	{
		std::list<std::pair<std::string, std::string>> objList;
		while((!images.eof() && images.good()) && (!clouds.eof() && clouds.good()))
		{
			std::string imname, clname;
			images >> imname;
			clouds >> clname;
			if(!imname.empty() && !clname.empty())
			{
				objList.push_back(std::pair<std::string, std::string>(imname, clname));
			}
		}
		return objList;
	}

	std::vector<cv::Point3f> getCloud(std::ifstream& clFile)
	{
		std::vector<cv::Point3f> clList;
		while(!clFile.eof())
		{
			cv::Point3f p;
			clFile >> p.x >> p.y >> p.z;
			clList.push_back(p);
		}
		return clList;
	}

	void featuresToCloudMatch(tod::Features2d& obj, std::vector<cv::Point3f>& objCloud)
	{
		std::vector<cv::Point3f> cloud;

		cv::Mat K, D;
		K = obj.camera.K;
		D = obj.camera.D;

		PoseRT pose = Tools::invert(obj.camera.pose);
		std::vector<cv::Point2f> projectedPoints;
		projectPoints(cv::Mat(objCloud), cv::Mat(3, 1, CV_64F, cv::Scalar(0.0)), cv::Mat(3, 1, CV_64F, cv::Scalar(0.0)), 
			K, D, projectedPoints);

		cv::Mat map(obj.image.rows, obj.image.cols, CV_32S, cv::Scalar(-1));
		int stride = obj.image.rows;
		int *mapPt = (int*)map.data;
		int size = projectedPoints.size();
		for(int i = 0; i < size; i++)
		{
			cv::Point p(projectedPoints[i]);
			if(p.x >= 0 && p.y >=0)
			{
				mapPt[p.x*stride + p.y] = i;
			}
		}
		size = obj.keypoints.size();
		for(int i = 0; i < size; i++)
		{
			cv::Point p(obj.keypoints[i].pt);
			if(mapPt[p.x*stride + p.y] >= 0)
			{
				cloud.push_back(objCloud[mapPt[p.x*stride + p.y]]);
			}
			else
			{
				bool flag = false;
				for(int i = -1; i <= 1; i++)
				{
					for(int j = -1; j <= 1; j++)
					{
						if(mapPt[(p.x+i)*stride + (p.y+j)] >= 0)
						{
							cloud.push_back(objCloud[mapPt[(p.x+i)*stride + (p.y+j)]]);
							flag = true;
							break;
						}
					}
					if(flag) break;
				}
				if(!flag)
				{
					cloud.push_back(cv::Point3f(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()));
				}
			}
		}

		objCloud.clear();
		objCloud = cloud;
		tod::filterCloudNan(objCloud, obj);
	}

	BaseCreator::BaseCreator(const std::string& directory_, cv::Ptr<tod::FeatureExtractor> extractor_)
	{
		directory = directory_;
		extractor = extractor_;
	}

	int BaseCreator::createBase(int scale)
	{
		std::ifstream fin(filename(CONFIG_NAME).c_str());
		std::list<std::string> objectNames = getStringsList(fin);
		fin.close();

		foreach(const std::string& name, objectNames)
		{
			std::ifstream imageFiles(filename(name + "/" + IMAGES_LIST).c_str());
			std::ifstream cloudFiles(filename(name + "/" + CLOUDS_LIST).c_str());
			std::list<std::pair<std::string, std::string>> objNames = getStringsList(imageFiles, cloudFiles);

			for(std::list<std::pair<std::string, std::string>>::iterator it = objNames.begin(); it != objNames.end(); it++)
			{
				tod::Features2d f2d;
				cv::FileStorage fs;

				f2d.image_name = name;
				f2d.mask_name = name + ".mask.png";
				f2d.image = cv::imread(filename(name + "/" + it->first).c_str());
				f2d.mask = cv::imread(filename(name + "/" + it->first + ".mask.png").c_str(), 0);
				if(scale > 1)
				{
					cv::resize(f2d.image, f2d.image, cv::Size(f2d.image.size().width/scale, f2d.image.size().height/scale));
					cv::resize(f2d.mask, f2d.mask, cv::Size(f2d.mask.size().width/scale, f2d.mask.size().height/scale));
				}
				extractor->detectAndExtract(f2d);

				fs = cv::FileStorage(filename("camera.yaml").c_str(), cv::FileStorage::READ);
				f2d.camera.read(fs["camera"]);
				if(scale > 1)
				{
					double *matrix = (double*)f2d.camera.K.data;
					matrix[0] /= scale;
					matrix[2] /= scale;
					matrix[3] /= scale;
					matrix[5] /= scale;
				}

				fs = cv::FileStorage(filename(name + "/" + it->first + ".pose.yaml").c_str(), cv::FileStorage::READ);
				f2d.camera.pose.read(fs["pose"]);

				std::ifstream clFile(filename(name + "/" + it->second).c_str());
				std::vector<cv::Point3f> cloud = getCloud(clFile);
				featuresToCloudMatch(f2d, cloud);

				tod::Features3d f3d(f2d, cloud);

				fs = cv::FileStorage(filename(name + "/" + it->first + ".f3d.yaml.gz").c_str(), cv::FileStorage::WRITE);
				fs << tod::Features3d::YAML_NODE_NAME;
				f3d.write(fs);
				std::cout<<name<<"/"<<it->first<<std::endl;
			}
		}

		return 0;
	}
}