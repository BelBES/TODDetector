/*
 * TrainingBase.cpp
 *
 *  Created on: Dec 7, 2010
 *      Author: erublee
 */

#include "tod/core/TrainingBase.h"
#include <list>
namespace tod
{

TrainingBase::TrainingBase()
{

}

TrainingBase::~TrainingBase()
{

}

TrainingBase::TrainingBaseInserter::TrainingBaseInserter(TrainingBase* tb) :
  tb(tb)
{

}

void TrainingBase::TrainingBaseInserter::operator()(const cv::Ptr<TexturedObject>& obj)
{
  tb->insertObject(obj);
}

size_t TrainingBase::size() const
{
  return objects_.size();
}

int TrainingBase::id_gen() const
{
  //todo make better ;)
  int id = size();
  while (ids_.count(id))
  {
    id++;
  }
  return id;
}

const cv::Ptr<TexturedObject> TrainingBase::getObject(int objectID) const
{
  ObjectCollection::const_iterator it = objects_.find(objectID);
  if (it == objects_.end())
  {
    return cv::Ptr<TexturedObject>();
  }
  return it->second;
}
void TrainingBase::getObjectIds(std::set<int>& ids) const
{
  ids = ids_;
}
void TrainingBase::getObjectIds(std::vector<int>& ids) const
{
  ids.clear();
  ids.reserve(ids_.size());
  ids.insert(ids.end(), ids_.begin(), ids_.end());
}

void TrainingBase::insertObject(cv::Ptr<TexturedObject> object)
{
  if (object.empty())
    return;
  if (object->id < 0)
  {
    object->id = id_gen();
  }
  ids_.insert(object->id);
  objects_[object->id] = object;
}
void TrainingBase::clear()
{
  ids_.clear();
  objects_.clear();
}
namespace
{
struct SerializeTO
{
  SerializeTO(cv::FileStorage& fs) :
    fs(fs)
  {
  }
  cv::FileStorage& fs;
  void operator()(const cv::Ptr<TexturedObject>& obj)
  {
    obj->write(fs);
  }
};
}
//serialization
void TrainingBase::write(cv::FileStorage& fs) const
{
  fs << "{" << "objects" << "[";
  forEachObject(SerializeTO(fs));
  fs << "]" << "}";
}
void TrainingBase::read(const cv::FileNode& fn)
{
  clear();
  cv::FileNode os = fn["objects"];
  CV_Assert(os.type() == cv::FileNode::SEQ)
    ;

  for (size_t i = 0; i < os.size(); i++)
  {
    cv::Ptr<TexturedObject> obj(new TexturedObject());
    obj->read(os[i]);
    insertObject(obj);
  }

}

}
