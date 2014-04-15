/*
 * TrainingBase.h
 *
 *  Created on: Dec 7, 2010
 *      Author: erublee
 */

#ifndef TOD_TRAININGBASE_H_
#define TOD_TRAININGBASE_H_

#include <tod/core/TexturedObject.h>
#include <opencv2/core/core.hpp>
#include <map>
#include <set>
#include <algorithm>
namespace tod
{
/** \brief A database like class that holds textured objects and can be serialized and deserialized.
 */
class TrainingBase : public Serializable
{
public:

  /**\brief Default constructor, doesn't do much...
   */
  TrainingBase();
  /**\brief Initialize the training base with any type of collection of cv::Ptr<TexturedObject>
   \code
   vector<Ptr<TexturedObject> > textured_objects(13);
   for (size_t i = 0; i < textured_objects.size(); i++)
   {
   textured_objects[i] = new TexturedObject(...); //populate the TexturedObject
   }
   TrainingBase base(textured_objects);
   \endcode
   */
  template<typename ObjectCollectionT>
  explicit  TrainingBase(const ObjectCollectionT& objects)
    {
      std::for_each(objects.begin(), objects.end(), TrainingBaseInserter(this));
    }
  virtual ~TrainingBase();

  /**\brief Get the number of objects, do not use this for iterating over the object ids
   */
  size_t size() const;

  /** \brief Retrieve and object by id, ids are not necessarily in the set [0,size()]
   */
  const cv::Ptr<TexturedObject> getObject(int objectID) const;

  /** \brief Get a vector of the object ids, useful for custom iteration over all objects.
   */
  void getObjectIds(std::vector<int>& ids) const;

  /** \brief Get a set of the object ids, useful for custom iteration over all objects.
   */
  void getObjectIds(std::set<int>& ids) const;

  /** \brief Insert an object into the TrainingBase.
   */
  void insertObject(cv::Ptr<TexturedObject> object);

  /** \brief Run a callable on every object in the TrainingBase.
   *Order of iteration is not specified.
   \code
   //this is the form of the expected callable
   struct TCallable{
   void operator()(const cv::Ptr<TexturedObject>&);
   };
   \endcode
   */
  template<typename Callable>
    void forEachObject(Callable c) const
    {
      std::for_each(objects_.begin(), objects_.end(), CallableCaller<Callable> (c));
    }

  /**\brief Clear all objects.
   */
  void clear();

  //serialization
  virtual void write(cv::FileStorage& fs) const;
  virtual void read(const cv::FileNode& fn);

private:

  //! The collection type for the textured objects, this is currently a mapping between TexturedObject::id's and
  //! pointers to TexturedObject
  typedef std::map<int, cv::Ptr<TexturedObject> > ObjectCollection;
  /** \brief Generate an id that is not in the set of already used ids.
   */
  int id_gen() const;

  /** \brief Private convenience class for inserting textured objects into the TrainingBase.
   */
  struct TrainingBaseInserter
  {
    TrainingBaseInserter(TrainingBase* tb);
    void operator()(const cv::Ptr<TexturedObject>& obj);
    TrainingBase* tb;
  };

  /** \brief Private convenience class for iterating over the TexturedObjects in the collection.
   */
  template<typename Callable>
    struct CallableCaller
    {
      Callable c;
      CallableCaller(Callable c);
      void operator()(TrainingBase::ObjectCollection::const_reference t);
    };

  ObjectCollection objects_; //!< The textured objects, mapped by id.
  std::set<int> ids_; //!< The ids, should be kept in sync with the objects_ keys.
};

template<typename Callable>
  TrainingBase::CallableCaller<Callable>::CallableCaller(Callable c) :
    c(c)
  {
  }
template<typename Callable>
  void TrainingBase::CallableCaller<Callable>::operator()(TrainingBase::ObjectCollection::const_reference t)
  {
    c(t.second);
  }

}

#endif /* TRAININGBASE_H_ */
