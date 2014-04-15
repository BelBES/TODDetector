/*
* Loader.cpp
*
*  Created on: Dec 15, 2010
*      Author: alex
*/
#include <boost/filesystem.hpp>
#include <Eigen/Core>
#include <fstream>
#include <string>
#include <iostream>
#include <list>
#include <vector>
#include <tod/detecting/Loader.h>
#include <tod/training/file_io.h>
#include <tod/training/clouds.h>
#include <tod/detecting/Tools.h>
#include <boost/foreach.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "fiducial/fiducial.h"
#include "opencv_candidate/PoseRT.h"

#include <opencv2/core/eigen.hpp>

#include <string>

#define filename(arg) (directory+"/"+arg)
#define foreach BOOST_FOREACH

using std::list;
using std::string;
using std::istream;
//using namespace cv;

namespace tod
{
	std::string Loader::CONFIG_NAME = "config.txt";
	std::string Loader::IMAGES_LIST = "images.txt";
	std::string Loader::CLOUDS_LIST = "pcds.txt";
	std::string Loader::POSTFIX = ".f3d.yaml.gz";
	static std::string IMAGES_LIST;
	Loader::Loader(const std::string& folderName)
	{
		directory = folderName;
	}

	list<string> getStringsList(istream& input)
	{
		list<string> imlist;
		while (!input.eof() && input.good())
		{
			string imname;
			input >> imname;
			if (!imname.empty())
				imlist.push_back(imname);
		}
		return imlist;
	}

	void Loader::findCloudFile(string& cloudFile, const string& objectPath)
	{
		std::ifstream pcdsFile;
		pcdsFile.open((objectPath + "/" + CLOUDS_LIST).c_str());
		//pcdsFile.open((objectPath + "\\" + CLOUDS_LIST).c_str());
		if (pcdsFile.is_open())
		{
			pcdsFile >> cloudFile;
			pcdsFile.close();
		}
		else
		{
			list<string> cloudNames = getFileList(objectPath, ".pcd");
			if (cloudNames.size())
				cloudFile = *cloudNames.begin();
		}
		cloudFile = objectPath + "/" + cloudFile;
		//cloudFile = objectPath + "\\" + cloudFile;
	}

	void Loader::calculateStddev(cv::Ptr<TexturedObject>& object, const string& objectPath)
	{
		string cloudFile;
		findCloudFile(cloudFile, objectPath);
		if (!cloudFile.empty())
		{
			vector<cv::Point3f> cloud;
			if (cloudFile.find(".txt") != std::string::npos)
			{
				std::ifstream txtfile;
				txtfile.open(cloudFile.c_str());
				while (!txtfile.eof() && txtfile.good())
				{
					cv::Point3f p;
					txtfile >> p.x >> p.y >> p.z;
					cloud.push_back(p);
				}
			}
			else
			{
				pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
				pcl::io::loadPCDFile(cloudFile, pcl_cloud);

				for (size_t ind = 0; ind < pcl_cloud.points.size(); ind++)
				{
					pcl::PointXYZRGB& p = pcl_cloud.points[ind];
					cloud.push_back(cv::Point3f(p.data[0], p.data[1], p.data[2]));
				}
				filterCloudNan(cloud);
			}
			if (cloud.size())
			{
				object->stddev = Tools::computeStdDev(cloud);
			}
		}
	}

	void Loader::calculateStddev(cv::Ptr<TexturedObject>& object)
	{
		float stddev = -1.0;
		for (size_t obsIndex = 0; obsIndex < 1; obsIndex++)
		{
			//fixed BeS 2013
			//const vector<cv::Point3f>& cloud = object->observations[obsIndex].cloud();
			//if (cloud.size())
			//{
			//	float curStdDev = Tools::computeStdDev(cloud);
			//	if (curStdDev > stddev)
			//		stddev = curStdDev;
			//}
			if (object->observations.size())
			{
				const vector<cv::Point3f>& cloud = object->observations[obsIndex].cloud();
				float curStdDev = Tools::computeStdDev(cloud);
				if (curStdDev > stddev)
					stddev = curStdDev;
			}
		}
		object->stddev = stddev;
	}

	void Loader::readTexturedObjects(vector<cv::Ptr<TexturedObject> >& objects)
	{
		objects.clear();

		std::ifstream fin(filename(CONFIG_NAME).c_str());
		list<string> objectNames = getStringsList(fin);
		fin.close();

		//get a one to one list of database keys.
		vector<string> database_ids;
		{
			std::ifstream din(filename("db.txt").c_str());
			list<string> dil = getStringsList(din);
			database_ids.assign(dil.begin(),dil.end());
			din.close();
		}

		size_t idx = 0;
		foreach(const string& name, objectNames)
		{
			cv::Ptr<TexturedObject> object = new TexturedObject();
			object->name = name;
			object->directory_ = directory;
			if(database_ids.size() == objectNames.size())
			{
				object->id = atoi(database_ids[idx].c_str());
				++idx;
			}
			// Get the list of files to read
			std::ifstream imagesFile(filename(name + "/" + IMAGES_LIST).c_str());
			list<string> imageNames = getStringsList(imagesFile);
			imagesFile.close();

			// If that list is empty, figure it out automatically by looking at the camera files and sampling them smartly
			if (imageNames.empty())
			{
				// Go over each camera file in the folder
				boost::filesystem::directory_iterator end_itr; // default construction yields past-the-end
				for (boost::filesystem::directory_iterator itr(directory + "/" + name); itr != end_itr; ++itr)
				{
					// Check out the camera file
					//TODO fixed BeS 04.11.2012
					//size_t position = itr->leaf().rfind(POSTFIX);
					size_t position = itr->path().string().rfind(POSTFIX);
					//size_t position = itr->path().filename(POSTFIX);

					if (position != string::npos)
					{

						//string f3dName = itr->string();
						//TODO fixed BeS 04.11.2012
						string f3dName = itr->path().string();

						cv::FileStorage fs = cv::FileStorage(f3dName, cv::FileStorage::READ);
						if (fs.isOpened())
						{
							Features3d f3d;
							f3d.read(fs["features3d"]);
							if(f3d.cloud().size())
							{
								object->observations.push_back(f3d);
							}
						}else
						{
							std::cout << "could not open: " << f3dName << std::endl;
						}
					}
				}
			}
			else
			{

				std::cout << "Loading " << imageNames.size() << " images" << std::endl;

				foreach(const std::string& imageName, imageNames)
				{
					std::string f3dName = filename(name + "/" + imageName + POSTFIX);

					cv::FileStorage fs = cv::FileStorage(f3dName, cv::FileStorage::READ);
					if (fs.isOpened())
					{
						Features3d f3d;
						f3d.read(fs["features3d"]);
						//TODO remove this for memory reasons, or come up with an intelligent way to
						// load up the image.
						//f3d.features().image = imread(filename(name + "/" + imageName ));
						if(f3d.cloud().size())
						{
							object->observations.push_back(f3d);
						}
					}
					fs.release();
				}
			}

			calculateStddev(object);//, filename(name));
			objects.push_back(object);
		}
	}
}

