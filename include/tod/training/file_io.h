/*
 * file_io.h
 *
 *  Created on: Dec 9, 2010
 *      Author: erublee
 */

#ifndef FILE_IO_H_
#define FILE_IO_H_

#include <string>
#include <iostream>
#include <list>
#include <vector>

#include <boost/foreach.hpp>

namespace tod{
/** process input stream to get list of image names
 */
std::list<std::string> getImageList(std::istream& input);

std::list<std::string> getFileList(const std::string& path, const std::string& extension);

/** \brief split the data into equal sublists
 */
template<class T>
  std::vector<T> splitList(const T& list, int n)
  {
    typedef typename T::value_type VT;
    std::vector<T> vlist(n);
    int i = 0;
    BOOST_FOREACH(const VT& x, list)
          {
            vlist[i % n].push_back(x);
            i++;
          }
    return vlist;
  }

}
#endif /* FILE_IO_H_ */
