/*
* fitting.cpp
*
*  Created on: Nov 4, 2010
*      Author: ethan
*/

#include "fiducial/fiducial.h"
#include <boost/foreach.hpp>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d/calib3d_c.h>

#include <iostream>
#include <string>
#include "opencv_candidate/circles_grid.hpp"
#define foreach BOOST_FOREACH

using namespace cv;
using std::vector;

namespace fiducial
{
	CheckerboardPoseEstimator::CheckerboardPoseEstimator(const cv::Size& chess_size, double square_size,
		const Camera& camera) :
	chess_size_(chess_size), square_size_(square_size), camera_(camera), K_(camera_.K),
		boardPoints_(CalcChessboardCorners(chess_size_, square_size_))
	{

	}
	CheckerboardPoseEstimator::Pose CheckerboardPoseEstimator::estimatePose(const Observation& observation) const
	{
		Pose pose;
		std::vector<Point2f> corners;
		//get some lower res images to work with
		cv::Mat low_res = observation;
		float scale_factor = ScaleImage(low_res, 640);
		if (!AccurateChesscorners(observation, chess_size_, corners, scale_factor, low_res))
			return Pose();
		solvePnP(Mat(boardPoints_), Mat(corners), K_, Mat(), pose.rvec, pose.tvec, false);
		pose.estimated = true;
		return pose;
	}

	/////////////////////////////////////////////////////////////////////////
	//static CheckerboardPoseEstimator functions
	/////////////////////////////////////////////////////////////////////////
	vector<Point3f> CheckerboardPoseEstimator::CalcChessboardCorners(cv::Size chess_size, float square_size)
	{
		vector<Point3f> corners;
		corners.reserve(chess_size.area());

		for (int i = 0; i < chess_size.height; i++)
			for (int j = 0; j < chess_size.width; j++)
				corners.push_back(Point3f(float(i * square_size), float(j * square_size), 0));
		return corners;
	}

	float CheckerboardPoseEstimator::ScaleImage(cv::Mat & image, float desired_width)
	{
		Size nsize = image.size();
		float factor = nsize.width / desired_width;
		Mat timage;
		if (factor > 1)
		{
			resize(image, timage, Size(), 1 / factor, 1 / factor, CV_INTER_LANCZOS4);
			image = timage;
			return float(nsize.width) / image.size().width;
		}
		return 1;
	}
	bool CheckerboardPoseEstimator::AccurateChesscorners(const cv::Mat& image_, cv::Size chess_size,
		std::vector<Point2f>& corners, float scale_factor,
		const cv::Mat low_res)
	{
		Mat image = image_;
		if (low_res.empty())
			scale_factor = ScaleImage(image, 800);
		else
			image = low_res;
		if (findChessboardCorners(
			image,
			chess_size,
			corners,
			CV_CALIB_CB_FAST_CHECK + CV_CALIB_CB_NORMALIZE_IMAGE + CV_CALIB_CB_ADAPTIVE_THRESH
			+ CV_CALIB_CB_FILTER_QUADS))
		{
			Mat tc(corners);
			tc *= scale_factor;
			cornerSubPix(image_, corners, Size(9, 9), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			return true;
		}
		else
			return false;
	}
	KnownPoseEstimator::KnownPoseEstimator(const std::string& pose_file)
	{
		FileStorage fs(pose_file, FileStorage::READ);
		if (!fs.isOpened())
			throw std::runtime_error("bad pose file");
		pose_.read(fs[PoseRT::YAML_NODE_NAME]);
	}

	void ComputeLinePoints(const Vec3f& l, Point2f & p1, Point2f& p2, Size sz)
	{
		float a = l[0], b = l[1], c = l[2];
		float h = sz.height;//, w=sz.width;

		p1.x = c / -a;
		p1.y = 0;

		p2.x = -(b * h + c) / a;
		p2.y = h;
	}

	void DrawEpilines(const std::vector<cv::Vec3f>& lines, cv::Mat& outputimage, cv::Scalar color)
	{
		foreach(cv::Vec3f l,lines)
		{

			Point2f p1, p2;

			ComputeLinePoints(l, p1, p2, outputimage.size());

			line(outputimage, p1, p2, color);

		}
	}
	namespace
	{
		struct EpiData
		{
			EpiData() :
		active(false)
		{
		}

		Point2f p;
		bool active;
		};
		void onEpiMouse(int event, int x, int y, int flags, void* param)
		{
			EpiData * cdata = reinterpret_cast<EpiData*> (param);
			if (CV_EVENT_LBUTTONDOWN == event)
			{
				cdata->p.x = x;
				cdata->p.y = y;
				cdata->active = true;
			}
			else if (CV_EVENT_LBUTTONUP)
			{
				cdata->active = false;
			}

		}
	}
	void EpiliPolarDrawer(cv::Mat image1, cv::Mat image2, cv::Mat F)
	{
		std::string win1 = "epipolar 1", win2 = "epipolar 2";
		namedWindow(win1, CV_WINDOW_KEEPRATIO);
		namedWindow(win2, CV_WINDOW_KEEPRATIO);

		imshow(win1, image1);
		imshow(win2, image2);

		EpiData e1, e2;

		setMouseCallback(win1, onEpiMouse, &e1);
		setMouseCallback(win2, onEpiMouse, &e2);

		while (char(waitKey(15)) != ' ')
		{
			std::vector<Vec3f> lines(1);
			Point2f p;
			if (e1.active)
			{
				p = e1.p;

			}
			else if (e2.active)
			{
				p = e2.p;
			}
			else
				continue;
			vector<Point2f> ps(1, p);
			computeCorrespondEpilines(Mat(ps), e1.active ? 1 : 2, F, lines);
			DrawEpilines(lines, e1.active ? image2 : image1, Scalar(255, 0, 255));
			imshow(win1, image1);
			imshow(win2, image2);
		}

	}

	Fiducial::Fiducial(Type ftype, const vector<Fiducial::Points>& patterns) :
	type_(ftype), templates_(patterns)
	{

	}
	Fiducial::Fiducial(const std::vector<Size>& corner_counts, const std::vector<float>& spacings,
		const std::vector<Point3f>& offsets, Fiducial::Type ftype) :
	type_(ftype), templates_(corner_counts.size()), corner_counts_(corner_counts)
	{

		int i = 0;
		foreach( Points& x, templates_ )
		{
			x = CalcObjectPoints(corner_counts[i], spacings[i], type_);
			foreach( Point3f& xpoint, x )
				xpoint += offsets[i];
			i++;
		}
	}
	void Fiducial::detect(const cv::Mat& test_img, std::vector<vector<cv::Point2f> >& observations, vector<bool>& found) const
	{

		cv::Mat gray;
		if (test_img.channels() == 3)
		{
			cvtColor(test_img, gray, CV_RGB2GRAY);
		}
		else
			gray = test_img;

		observations.resize(templates_.size());
		found.resize(templates_.size());

		Mat scaled_down = gray;
		float scale_factor = 1;//CheckerboardPoseEstimator::ScaleImage(scaled_down, 640);

		for (size_t i = 0; i < templates_.size(); i++)
		{
			switch (type_)
			{
			case Fiducial::CHECKER_BOARD:
				{
					found[i] = CheckerboardPoseEstimator::AccurateChesscorners(gray, corner_counts_[i], observations[i],
						scale_factor, scaled_down);
					break;
				}
			case Fiducial::DOTS:
				found[i] = opencv_candidate::findCirclesGrid(gray, corner_counts_[i], observations[i],
					opencv_candidate::CALIB_CB_SYMMETRIC_GRID);
				break;
			case Fiducial::ASYMMETRIC_DOTS:
				found[i] = opencv_candidate::findCirclesGrid(gray, corner_counts_[i], observations[i],
					opencv_candidate::CALIB_CB_ASYMMETRIC_GRID);
			}
		}
	}

	std::vector<cv::Point3f> Fiducial::CalcObjectPoints(cv::Size pattern_size, float square_size, Type pattern_type)
	{
		vector<Point3f> corners;
		corners.reserve(pattern_size.area());

		switch (pattern_type)
		{
		case CHECKER_BOARD:
		case DOTS:
			for (int i = 0; i < pattern_size.height; i++)
				for (int j = 0; j < pattern_size.width; j++)
					corners.push_back(Point3f(float(i * square_size), float(j * square_size), 0));
			break;
		case ASYMMETRIC_DOTS:
			for (int i = 0; i < pattern_size.height; i++)
				for (int j = 0; j < pattern_size.width; j++)
					corners.push_back(
					Point3f(float((2 * j + i % 2) * square_size),
					float(square_size * (pattern_size.height - 1 - i)), 0));
			break;

		}
		return corners;
	}

	void Fiducial::draw(cv::Mat& drawimage, const std::vector<vector<cv::Point2f> >& observations,
		const vector<bool>& found) const
	{
		for (size_t i = 0; i < observations.size(); i++)
		{
			cv::drawChessboardCorners(drawimage, corner_counts_[i], observations[i], found[i]);
		}
	}
	const std::vector<Fiducial::Points>&
		Fiducial::getTemplates() const
	{
		return templates_;
	}

	void Fiducial::write(cv::FileStorage& fs) const
	{
		cvWriteComment(*fs, "Fiducial", 0);
		fs << "{";
		cvWriteComment(*fs, "type, 0 == CHECKER_BOARD, 1 == DOTS, 2 == ASYMMETRIC_DOTS", 0);
		fs << "type" << (int)type_;
		fs << "templates" << "[";
		for (size_t i = 0; i < templates_.size(); ++i)
		{
			fs << Mat(templates_[i]);
		}
		fs << "]";
		fs << "corner_counts" << "[";
		for (size_t i = 0; i < corner_counts_.size(); ++i)
		{
			fs << "{" << "width" << corner_counts_[i].width << "height" << corner_counts_[i].height << "}";
		}
		fs << "]";
		fs << "}";
	}
	void Fiducial::read(const cv::FileNode& fn)
	{
		type_ = (Fiducial::Type)(int)fn["type"];
		FileNode mats = fn["templates"];
		CV_Assert(mats.type() == FileNode::SEQ)
			;
		templates_.resize(mats.size());
		for (size_t i = 0; i < mats.size(); i++)
		{
			Mat tm;
			try
			{
				cv::read(mats[i], tm, Mat());
			}
			catch (Exception exc)
			{
				if (exc.code == -201)
				{
					tm = Mat();
				}
				else
					throw exc;
			}
			if (!tm.empty())
			{

				templates_[i].resize(tm.rows);
				templates_[i].assign(tm.begin<Point3f> (), tm.end<Point3f> ());
			}
		}
		FileNode cc = fn["corner_counts"];
		CV_Assert(cc.type() == FileNode::SEQ)
			;
		corner_counts_.resize(cc.size());
		for (size_t i = 0; i < cc.size(); i++)
		{
			corner_counts_[i].width = (int)cc[i]["width"];
			corner_counts_[i].height = (int)cc[i]["height"];
		}
	}

	FiducialPoseEstimator::FiducialPoseEstimator(const Fiducial& fudicial, const Camera& camera, bool verbose) :
	fiducial_(fudicial), camera_(camera), verbose_(verbose)
	{

	}

	PoseRT FiducialPoseEstimator::estimatePose(const Observation& observation) const
	{
		vector<vector<Point2f> > corners;
		vector<bool> found;
		fiducial_.detect(observation, corners, found);

		Mat K = camera_.K.clone();
		if (observation.size().area() != camera_.image_size.area())
		{
			std::cout << "warning different calibrated size than observation size\n";
			float scale = observation.size().width / float(camera_.image_size.width);

			K *= scale;
			K.at<float> (3, 3) = 1;
			std::cout << "K = " << K << "\n";
		}

		vector<Point3f> all_object_pts;
		vector<Point2f> all_observation_pts;
		vector<PoseRT> poses(corners.size());
		for (size_t i = 0; i < corners.size(); i++)
		{
			const vector<Point3f>& template_tmp = fiducial_.getTemplates()[i];
			if (found[i])
			{
				all_observation_pts.insert(all_observation_pts.end(), corners[i].begin(), corners[i].end());
				all_object_pts.insert(all_object_pts.end(), template_tmp.begin(), template_tmp.end());
			}
		}

		if (all_object_pts.empty())
			return PoseRT();
		else
		{
			PoseRT pose;
			solvePnP(Mat(all_object_pts), Mat(all_observation_pts), K, Mat(), pose.rvec, pose.tvec, false);
			pose.estimated = true;

			if (verbose_)
			{
				cv::Mat drawImage;
				if (observation.channels() == 1)
					cvtColor(observation, drawImage, CV_GRAY2BGR);
				else
					observation.copyTo(drawImage);
				fiducial_.draw(drawImage, corners, found);
				PoseDrawer(drawImage, K, pose);
				namedWindow("Pose", CV_WINDOW_KEEPRATIO);
				imshow("Pose", drawImage);
				std::cout << "pose --- rvec = " << pose.rvec << " tvec = " << pose.tvec << std::endl;
				waitKey(20);
			}
			return pose;
		}
	}

	void PoseDrawer(cv::Mat& drawImage, const cv::Mat& K, const PoseRT& pose)
	{
		Point3f z(0, 0, 0.25);
		Point3f x(0.25, 0, 0);
		Point3f y(0, 0.25, 0);
		Point3f o(0, 0, 0);
		vector<Point3f> op(4);
		op[1] = x, op[2] = y, op[3] = z, op[0] = o;
		vector<Point2f> ip;
		projectPoints(Mat(op), pose.rvec, pose.tvec, K, Mat(4, 1, CV_64FC1, Scalar(0)), ip);

		vector<Scalar> c(4); //colors
		c[0] = Scalar(255, 255, 255);
		c[1] = Scalar(255, 0, 0);//x
		c[2] = Scalar(0, 255, 0);//y
		c[3] = Scalar(0, 0, 255);//z
		line(drawImage, ip[0], ip[1], c[1]);
		line(drawImage, ip[0], ip[2], c[2]);
		line(drawImage, ip[0], ip[3], c[3]);
		std::string scaleText = "scale 0.25 meters";
		int baseline = 0;
		Size sz = getTextSize(scaleText, CV_FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
		rectangle(drawImage, Point(10, 30 + 5), Point(10, 30) + Point(sz.width, -sz.height - 5), Scalar::all(0), -1);
		putText(drawImage, scaleText, Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 1.0, c[0], 1, CV_AA, false);
		putText(drawImage, "Z", ip[3], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[3], 1, CV_AA, false);
		putText(drawImage, "Y", ip[2], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[2], 1, CV_AA, false);
		putText(drawImage, "X", ip[1], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[1], 1, CV_AA, false);

	}
}
