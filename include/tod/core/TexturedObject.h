/*
* TexturedObject.h
*
*  Created on: Dec 7, 2010
*      Author: erublee
*/

#ifndef TEXTUREDOBJECT_H_
#define TEXTUREDOBJECT_H_

#include <tod/core/common.h>
#include <tod/core/Features3d.h>
#include <opencv2/core/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

namespace tod
{

	/** \brief The canonical collection of view based data that represents an object and its observed texture data.
	*
	* The TexturedObject is defined by a set of Features3d which can be though of as different views of the same object.
	* It also is referable to by name and id.
	*/
	class TexturedObject : public Serializable
	{
	public:
		TexturedObject();
		virtual ~TexturedObject();

		/** \brief Get a stacked vector of all the descriptors from each observation.
		* This is useful for one to many matching in opencv
		*/
		std::vector<cv::Mat> getDescriptors() const;
		//serialization
		virtual void write(cv::FileStorage& fs) const;
		virtual void read(const cv::FileNode& fn);

		std::vector<Features3d> observations; //!< View based representation of the object.
		int id; //!< An id for the object, useful for database lookups - default value is -1, i.e not set.
		std::string name; //!< Human readable name for the TexturedObject detector - default value is ""

		double stddev;

		// TODO remove the following, that should be transparent to the user in the new API
		std::string directory_;

		/** Give the 3d model of the object, and load it if necessary
		*/
		const pcl::PointCloud<pcl::PointXYZRGB> & cloud() const;

	private:
		/** The point cloud of the object
		*/
		pcl::PointCloud<pcl::PointXYZRGB> cloud_;
	public:
		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	};

}

#endif /* TEXTUREDOBJECT_H_ */
