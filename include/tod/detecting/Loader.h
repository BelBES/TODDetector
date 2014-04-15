/*
 * Loader.h
 *
 *  Created on: Dec 15, 2010
 *      Author: alex
 */

#ifndef LOADER_H_
#define LOADER_H_

#include <tod/core/TrainingBase.h>
#include <list>
#include <iostream>

namespace tod
{

std::list<std::string> getStringsList(std::istream& input);

class Loader
{
public:
  Loader(const std::string& folderName_);
  void readTexturedObjects(std::vector<cv::Ptr<TexturedObject> >& objects);

private:
  static std::string CONFIG_NAME;
  static std::string IMAGES_LIST;
  static std::string CLOUDS_LIST;
  static std::string POSTFIX;
  std::string directory;

  void calculateStddev(cv::Ptr<TexturedObject>& object, const std::string& objectPath);
  void calculateStddev(cv::Ptr<TexturedObject>& object);
  void findCloudFile(std::string& cloudFile, const std::string& objectPath);
};
}

#endif /* LOADER_H_ */
