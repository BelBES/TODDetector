/*
* file_io.cpp
*
*  Created on: Dec 9, 2010
*      Author: erublee
*/

#include <tod/training/file_io.h>
#include <boost/filesystem.hpp>
#include <vector>
#include <algorithm>
using std::vector;
using std::list;
using std::string;
using std::istream;
namespace fs = boost::filesystem;
namespace tod
{
	list<string> getImageList(istream& input)
	{
		list<string> imlist;
		while (input.good())
		{
			string imname;
			input >> imname;
			if (!input.eof() && !imname.empty())
				imlist.push_back(imname);
		}
		return imlist;
	}

	std::list<std::string> getFileList(const std::string& path, const std::string& extension)
	{
		vector<string> filelist;
		fs::path file_dir(path);
		int file_count = 0;
		int err_count = 0;
		if (!fs::is_directory(file_dir))
			throw std::runtime_error("path is not a directory");

		fs::directory_iterator end_iter;
		for (fs::directory_iterator dir_itr(file_dir); dir_itr != end_iter; ++dir_itr)
		{
			try
			{
				if (fs::is_regular_file(dir_itr->status()))
				{
					++file_count;
					fs::path f = dir_itr->path();
					if(f.extension()  == extension){
						if(f.string().rfind('.',f.string().size() - extension.size() - 1) == std::string::npos) // check for simple exstension (throw out .mask.png ...) TODO do this better
						{
							//TODO fixed 28.10.2012 BeS
							filelist.push_back(f.leaf().string());
							//filelist.push_back(f.leaf());
						}

					}
				}
			}
			catch (const std::exception & ex)
			{
				++err_count;
				std::cerr << dir_itr->path().filename() << " " << ex.what() << std::endl;
			}
		}
		std::sort(filelist.begin(),filelist.end());

		return std::list<std::string>(filelist.begin(),filelist.end());
	}
}
