/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
*                                                                             *
* This library is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU Lesser General Public License as published by    *
* the Free Software Foundation; either version 2.1 of the License, or (at     *
* your option) any later version.                                             *
*                                                                             *
* This library is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License *
* for more details.                                                           *
*                                                                             *
* You should have received a copy of the GNU Lesser General Public License    *
* along with this library; if not, write to the Free Software Foundation,     *
* Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA.          *
*******************************************************************************
*                               SOFA :: Modules                               *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#ifndef IMAGE_OPENNICAM_H
#define IMAGE_OPENNICAM_H


#include "initImage.h"
#include "ImageTypes.h"
#include <sofa/defaulttype/Vec.h>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/objectmodel/DataFileName.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/objectmodel/Event.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

#ifdef Success
  #undef Success
#endif

#include <pcl-1.7/pcl/common/common_headers.h>
#include <pcl-1.7/pcl/point_cloud.h>
#include <pcl-1.7/pcl/filters/extract_indices.h>
#include <pcl-1.7/pcl/point_types.h>
#include <pcl-1.7/pcl/impl/point_types.hpp>
#include <pcl-1.7/pcl/features/normal_3d.h>
#include <pcl-1.7/pcl/PointIndices.h>
#include <pcl-1.7/pcl/PolygonMesh.h>
#include <pcl-1.7/pcl/visualization/common/actor_map.h>
#include <pcl-1.7/pcl/visualization/common/common.h>
#include <pcl-1.7/pcl/visualization/point_cloud_geometry_handlers.h>
#include <pcl-1.7/pcl/visualization/point_cloud_color_handlers.h>
#include <pcl-1.7/pcl/visualization/point_picking_event.h>
#include <pcl-1.7/pcl/visualization/area_picking_event.h>
#include <pcl-1.7/pcl/visualization/interactor_style.h>
#include <pcl-1.7/pcl/visualization/pcl_visualizer.h>
#include <pcl-1.7/pcl/visualization/keyboard_event.h>
#include <pcl-1.7/pcl/correspondence.h>
#include <pcl-1.7/pcl/features/normal_3d_omp.h>
#include <pcl-1.7/pcl/features/shot_omp.h>
#include <pcl-1.7/pcl/features/board.h>
#include <pcl-1.7/pcl/console/parse.h>
#include <pcl-1.7/pcl/common/projection_matrix.h>
#include <pcl-1.7/pcl/common/pca.h>
#include <pcl-1.7/pcl/surface/reconstruction.h>
#include <pcl-1.7/pcl/io/pcd_io.h>
#include <pcl-1.7/pcl/common/io.h>
#include <pcl-1.7/pcl/registration/transforms.h>
#include <pcl-1.7/pcl/keypoints/sift_keypoint.h>
#include <pcl-1.7/pcl/keypoints/harris_3d.h>
#include <pcl-1.7/pcl/ModelCoefficients.h>
#include <pcl-1.7/pcl/sample_consensus/method_types.h>
#include <pcl-1.7/pcl/sample_consensus/model_types.h>
#include <pcl-1.7/pcl/segmentation/sac_segmentation.h>
#include <pcl-1.7/pcl/search/kdtree.h>
#include <pcl-1.7/pcl/segmentation/extract_clusters.h>
#include <pcl-1.7/pcl/features/fpfh_omp.h>
#include <pcl-1.7/pcl/features/feature.h>
#include <pcl-1.7/pcl/features/pfh.h>
#include <pcl-1.7/pcl/features/pfhrgb.h>
#include <pcl-1.7/pcl/features/3dsc.h>
#include <pcl-1.7/pcl/features/shot_omp.h>
#include <pcl-1.7/pcl/filters/filter.h>
#include <pcl-1.7/pcl/registration/transformation_estimation_svd.h>
#include <pcl-1.7/pcl/common/transforms.h>


#include <pthread.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "opencv2/videoio/videoio.hpp"

/*#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>*/

#include <visp/vpIoTools.h>
#include <visp/vpImageIo.h>
#include <visp/vpParseArgv.h>


#include <libfreenect.h>
//#include <libfreenect_sync.h>
//#include <libfreenect-registration.h>


namespace sofa
{

namespace component
{

namespace container
{

using namespace cimg_library;
using defaulttype::Vec;
using defaulttype::Vector3;


using namespace std;
using namespace openni_wrapper;
using namespace cv;
using namespace boost;

void* globalOpenniCamClassPointer;


pthread_mutex_t backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
//pthread_cond_t frame_cond = PTHREAD_COND_INITIALIZER;


class OpenniCam : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS( OpenniCam , Inherited);

    // image data
    typedef defaulttype::ImageUC ImageTypes;
    typedef ImageTypes::T T;
    typedef ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > image;

    // transform data
    typedef SReal Real;
    typedef defaulttype::ImageLPTransform<Real> TransformType;
    typedef helper::WriteAccessor<Data< TransformType > > waTransform;
    typedef helper::ReadAccessor<Data< TransformType > > raTransform;
    Data< TransformType > transform;

    // depth data
    typedef defaulttype::ImageF DepthTypes;
    typedef DepthTypes::T dT;
    typedef DepthTypes::imCoord dCoord;
    typedef helper::WriteAccessor<Data< DepthTypes > > waDepth;
    typedef helper::ReadAccessor<Data< DepthTypes > > raDepth;
    Data< DepthTypes > depthImage;
    Data< TransformType > depthTransform;

    Data<unsigned int> deviceID;
    Data<helper::OptionsGroup> resolution;
    Data<helper::OptionsGroup> videoMode;
    Data<helper::OptionsGroup> depthMode;
    Data<helper::OptionsGroup> ledMode;
    Data<int> tiltAngle;
    Data<defaulttype::Vector3> accelerometer;
    Data<bool> drawBB;
    Data<bool> drawGravity;
    Data<float> showArrowSize;
	
			public:

/** The thread in which the data will be captured */
//boost::thread thread;

/** The images in which to store the different data types */
cv::Mat_<cv::Vec3b> color;
cv::Mat_<uchar> depth;
cv::Mat_<cv::Vec3f> points3d;
//typename pcl::PointCloud<pcl::PointXYZRGB> cloud;

bool saveImageFlag;
bool saveDepthFlag;

int niterations;


    virtual std::string getTemplateName() const	{ return templateName(this); }
    static std::string templateName(const OpenniCam* = NULL) {	return std::string(); }

    OpenniCam() : Inherited()
	        , depthImage(initData(&depthImage,DepthTypes(),"depthImage","depth map"))
			, depthTransform(initData(&depthTransform, TransformType(), "depthTransform" , ""))
        , image(initData(&image,ImageTypes(),"image","image"))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , deviceID ( initData ( &deviceID,(unsigned int)0,"deviceID","device ID" ) )
        , resolution ( initData ( &resolution,"resolution","resolution" ) )
        , videoMode ( initData ( &videoMode,"videoMode","video mode" ) )
        , depthMode ( initData ( &depthMode,"depthMode","depth mode" ) )
        , ledMode ( initData ( &ledMode,"ledMode","led mode" ) )
        , tiltAngle(initData(&tiltAngle,0,"tiltAngle","tilt angle in [-30,30]"))
        , accelerometer(initData(&accelerometer,Vector3(0,0,0),"accelerometer","Accelerometer data"))
        , drawBB(initData(&drawBB,false,"drawBB","draw bounding box"))
        , drawGravity(initData(&drawGravity,true,"drawGravity","draw acceleration"))
        , showArrowSize(initData(&showArrowSize,0.1f,"showArrowSize","size of the axis"))
        , die(0)
        , got_rgb(0)
        , got_depth(0)
    {

    globalOpenniCamClassPointer = (void*) this; // used for SoftKinetic callbacks
    depth = cv::Mat::zeros(120, 160, CV_8U);
	color = cv::Mat::zeros(480, 640, CV_8UC3);
	drawBB = false;

    }


    virtual void clear()
    {
        /*waImage wimage(this->image);
        wimage->clear();
        waDepth wdepth(this->depthImage);
        wdepth->clear();*/
    }

    virtual ~OpenniCam()
    {
        die = 1;

        pthread_join(opennicam_thread, NULL);

    }

    // mutex functions
    void mutex_lock()
    {
        //                    pthread_cond_wait(&frame_cond, &backbuf_mutex);
        pthread_mutex_lock(&backbuf_mutex);
    }

    void mutex_unlock()
    {
        //        pthread_cond_signal(&frame_cond);
        pthread_mutex_unlock(&backbuf_mutex);
    }

    // callbacks with static wrappers
    static void _depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)    { reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->depth_cb(dev, v_depth, timestamp);  }
    void depth_cb(freenect_device *dev, void *v_depth, uint32_t /*timestamp*/)
    {
        /*mutex_lock();
        // swap buffers
        depth_back = depth_mid;
        freenect_set_depth_buffer(dev, depth_back);
        depth_mid = (unsigned char*)v_depth;
        got_depth++;
        mutex_unlock();*/
    }

    static void _rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp)    { reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->rgb_cb(dev, rgb, timestamp);  }
    void rgb_cb(freenect_device *dev, void *rgb, uint32_t /*timestamp*/)
    {
        /*mutex_lock();
        // swap buffers
        rgb_back = rgb_mid;
        freenect_set_video_buffer(dev, rgb_back);
        rgb_mid = (unsigned char*)rgb;
        got_rgb++;
        mutex_unlock();*/
    }
	
	/*----------------------------------------------------------------------------*/
// New audio sample event handler
void onNewAudioSample(AudioNode node, AudioNode::NewSampleReceivedData data)
{
    //printf("A#%u: %d\n",g_aFrames,data.audioData.size());
    audio_frames++;
}

static void onNewAudioSample_t(AudioNode node, AudioNode::NewSampleReceivedData data)
{
     reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->onNewAudioSample(node,data);
}

/*----------------------------------------------------------------------------*/
// New color sample event handler
void onNewColorSample(ColorNode node, ColorNode::NewSampleReceivedData data)
{
		int t = (int)this->getContext()->getTime();
		
		//if (t%niterations == 0)
		{
    // Read the color buffer and display
    int32_t w, h;
	
	waImage wimage(this->image);
	CImg<T>& img =wimage->getCImg(0);
	
    DepthSense::FrameFormat_toResolution(data.captureConfiguration.frameFormat, &w, &h);
		
	mutex_lock();
	

    cv::Mat color_yuy2(h, w, CV_8UC2, const_cast<void*>((const void*) (data.colorMap)));
	
	mutex_unlock();

    {
      //boost::unique_lock<boost::mutex> lock(color_mutex);
      cv::cvtColor(color_yuy2, color, CV_YUV2BGR_YUY2);
	  
	  //depth.convertTo(depth_single,CV_16UC1);
	  
		if(img.spectrum()==3)  // deinterlace
                    {
                        unsigned char* rgb = (unsigned char*)color.data;
                        unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
                        for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++); }
                    }
                    else memcpy(img.data(),  color.data, img.width()*img.height()*sizeof(T));
	  
	  	//memcpy(img.data(), (unsigned char*)color.data ,3*w*h*sizeof(unsigned char));
    }
    //color_cond.notify_all();

    color_frames++;

    /*cv::namedWindow("image_softkinetic");
    cv::imshow("image_softkinetic",color);
    if (cvWaitKey(20) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
   {
        std::cout << "esc key is pressed by user" << std::endl;
   }*/
		}

}

static void onNewColorSample_t(ColorNode node, ColorNode::NewSampleReceivedData data)
{
     reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->onNewColorSample(node,data);
}

/*----------------------------------------------------------------------------*/

/*void downsampleCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_to_filter)
{
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud (cloud_to_filter);
    sor.setLeafSize (voxel_grid_side, voxel_grid_side, voxel_grid_side);
    sor.filter (*cloud_to_filter);
}*/

/*void softKinetic::filterCloudRadiusBased(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_to_filter)
{
    // radius based filter:
    pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> ror;
    ror.setInputCloud(cloud_to_filter);
    ror.setRadiusSearch(search_radius);
    ror.setMinNeighborsInRadius(minNeighboursInRadius);
    // apply filter
    ROS_DEBUG_STREAM("Starting filtering");
    int before = cloud_to_filter->size();
    double old_ = ros::Time::now().toSec();
    ror.filter(*cloud_to_filter);
    double new_= ros::Time::now().toSec() - old_;
    int after = cloud_to_filter->size();
    ROS_DEBUG_STREAM("filtered in " << new_ << " seconds;"
                  << "points reduced from " << before << " to " << after);
    cloud.header.stamp = ros::Time::now();
}*/

// New depth sample event varsace tieshandler

/*void softKinetic::onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    int count = -1;

    // Project some 3D points in the Color Frame
    if (!g_pProjHelper)
    {
        g_pProjHelper = new ProjectionHelper(data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }
    else if (g_scp != data.stereoCameraParameters)
    {
        g_pProjHelper->setStereoCameraParameters(data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }

    int32_t w, h;
    FrameFormat_toResolution(data.captureConfiguration.frameFormat,&w,&h);
    int cx = w/2;
    int cy = h/2;

    Vertex p3DPoints[1];
    Point2D p2DPoints[1];

    g_dFrames++;

    current_cloud->height = h;
    current_cloud->width = w;
    current_cloud->is_dense = false;
    current_cloud->points.resize(w*h);

    uchar b,g,r;
    uint32_t rgb;
    cv::Vec3b bgr;

    for(int i = 1;i < h ;i++){
        for(int j = 1;j < w ; j++){
            count++;
            current_cloud->points[count].x = -data.verticesFloatingPoint[count].x;
            current_cloud->points[count].y = data.verticesFloatingPoint[count].y;
            if(data.verticesFloatingPoint[count].z == 32001){
                current_cloud->points[count].z = 0;
            }else{
                current_cloud->points[count].z = data.verticesFloatingPoint[count].z;
            }
            p3DPoints[0] = data.vertices[count];
            g_pProjHelper->get2DCoordinates ( p3DPoints, p2DPoints, 1, CAMERA_PLANE_COLOR);
            int x_pos = (int)p2DPoints[0].x;
            int y_pos = (int)p2DPoints[0].y;
        }
    }

    cloud = *current_cloud;

    //check for usage of radius filtering
//    if(use_radius_filter)
//    {
        //we must downsample the cloud so the filter don't take too long
//        downsampleCloud(current_cloud);
//        filterCloudRadiusBased(current_cloud);
//    }

    g_context.quit();
}*/

/*void softKinetic::onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

    int count = -1;

    // Project some 3D points in the Color Frame
    if (!g_pProjHelper)
    {
        g_pProjHelper = new ProjectionHelper(data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }
    else if (g_scp != data.stereoCameraParameters)
    {
        g_pProjHelper->setStereoCameraParameters(data.stereoCameraParameters);
        g_scp = data.stereoCameraParameters;
    }
	
	waDepth wdepth(this->depthImage);
	waTransform wdt(this->depthTransform);
	CImg<dT>& depth)                // if depthValue is not NaN
			{
				// Find 3D position respect to rgb frame:
				newPoint.z = depthValue;
				newPoint.x = (sample*j - rgbIntrinsicMatrix(0,2)) * newPoint.z * rgbFocalInvertedX;
				newPoint.y = (sample*i - rgbIntrinsicMatrix(1,2)) * newPoint.z * rgbFocalInvertedY;
				std::cout << " x " << newPoint.x << " y " << newPoint.y << " z " << newPoint.z << std::endl;
				std::cout << " xf " << data.verticesFloatingPoint[count].x << " yf " << data.verticesFloatingPoint[count].y << " zf " << data.verticesFloatingPoint[count].z << std::endl;
			}
			/*else
			{
				newPoint.z = std::numeric_limits<float>::quiet_NaN();
				newPoint.x = std::numeric_limits<float>::quiet_NaN();
				newPoint.y = std::numeric_limits<float>::quiet_NaN();
				newPoint.r = std::numeric_limits<unsigned char>::quiet_NaN();
				newPoint.g = std::numeric_limits<unsigned char>::quiet_NaN();
				newPoint.b = std::numeric_limits<unsigned char>::quiet_NaN();
				//outputPointcloud.push_back(newPoint);
			}*/
//		}
//	}


	  //std::cout << depthimg.width() << std::endl;
//memcpy(depthimg.data(),  (unsigned short*)const_cast<void*>((const void*) (data.depthMap)) , w*h*sizeof(unsigned short));

	  /*for (int i = 0; i<depth.rows; i++)
		  for (int j = 0; j<depth.cols; j++)
	  {
		  std::cout << " cimg " << (double)*depthimg.data(j,i) << std::endl;
		  std::cout << " cv " << (double)depth_singlef.at<float>(i,j) << std::endl;

	  
	  }*/
	  
	  
      /*uchar* depth_buffer = depth.ptr<uchar>(0);
      if (data.depthMapFloatingPoint!=0)
  for (int i = 0; i < data.depthMapFloatingPoint.size(); i++)
  {
      float val = data.depthMapFloatingPoint[i];
    if (!saveImageFlag && !saveDepthFlag) val*=150;
    if (val<0) val=255; // catch the saturated points
      *depth_buffer++ = (unsigned char)val;
  }*/
    
	//}

    //depth_cond.notify_all();
    //depth_frames++;

    /*cv::namedWindow("depth_softkinetic");
    cv::imshow("depth_softkinetic",depth);


    if (saveImageFlag || saveDepthFlag) { // save a timestamped image pair; synched by depth image time
      char filename[100];
      //g_fTime = clock();
      //sprintf(filename,"df%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
      //cvSaveImage(filename,g_depthImage);
      //sprintf(filename,"vf%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
      //if (g_saveImageFlag)
    //cvSaveImage(filename,g_videoImage);
    }

    // Allow OpenCV to shut down the program
    char key = cvWaitKey(10);

    if (key==27) {
      printf("Quitting main loop from OpenCV\n");
        context.quit();
    } else
      if (key=='W') saveImageFlag = !saveImageFlag;
      else if (key=='w') saveDepthFlag = !saveDepthFlag;*/
		//}
	  	  	  
//}

void onNewDepthSample(DepthNode node, DepthNode::NewSampleReceivedData data)
{
	
			int t = (int)this->getContext()->getTime();
		
		//if (t%niterations == 0)
		{
    // Read the color buffer and display
    int32_t w, h;
	
	waDepth wdepth(this->depthImage);
	waTransform wdt(this->depthTransform);
	CImg<dT>& depthimg =wdepth->getCImg(0);
	//CImg<dT>& depthimg0;
	//depthimg.resize(160,120,1,1);
	
    /*double fx = data.stereoCameraParameters.depthIntrinsics.fx;
    double fy = data.stereoCameraParameters.depthIntrinsics.fy;

    double cx = data.stereoCameraParameters.depthIntrinsics.cx;
    double cy = data.stereoCameraParameters.depthIntrinsics.cy;
	
	std::cout << " fx " << fx << " fy " << fy << " cx " << cx << " cy " << cy << std::endl; */

	
    DepthSense::FrameFormat_toResolution(data.captureConfiguration.frameFormat, &w, &h);
	
				//std::cout << " ok depth " << std::endl;

	mutex_lock();
					//std::cout << " ok depth !" << std::endl;
    //cv::Mat depth_single(h, w, CV_16U, const_cast<void*>((const void*) (data.depthMap)));
	cv::Mat depth_singlef(h, w, CV_32F, const_cast<void*>((const void*) (data.depthMapFloatingPoint)));

		  		mutex_unlock();


    {
      //boost::unique_lock<boost::mutex> lock(color_mutex);
      //depth_single.copyTo(depth);
      
	  //depth_single.convertTo(depth, CV_8U, 255./400., -100);
	  
	  //depth.convertTo(depth_single,CV_16UC1);
	  	
		//memcpy(depthimg.data(), (unsigned char*)depth.data , w*h*sizeof(unsigned char));
		memcpy(depthimg.data(), (float*)depth_singlef.data , w*h*sizeof(float));
		//memcpy(depthimg.data(), (unsigned short*)depth_single.data , w*h*sizeof(unsigned short));

	Eigen::Matrix3f rgbIntrinsicMatrix;
	rgbIntrinsicMatrix(0,0) = 166.53;
	rgbIntrinsicMatrix(1,1) = 166.146;
	rgbIntrinsicMatrix(0,2) = 80;
	rgbIntrinsicMatrix(1,2) = 60;
    //cv::imwrite("depth.png",depthImage);
    //cv::imwrite("RGB.png",frgd);

int count = 0;
        float rgbFocalInvertedX = 1/rgbIntrinsicMatrix(0,0);	// 1/fx
	float rgbFocalInvertedY = 1/rgbIntrinsicMatrix(1,1);	// 1/fy
	pcl::PointXYZRGB newPoint;
	int sample = 1;
	for (int i=0;i<(int)depth_singlef.rows/sample;i++)
	{
		for (int j=0;j<(int)depth_singlef.cols/sample;j++)
		{
			float depthValue = depth_singlef.at<float>(sample*i,sample*j);
	count++;			
			//std::cout << " x " << avalue << std::endl;  
			if (depthValue>0)                // if depthValue is not NaN
			{
				// Find 3D position respect to rgb frame:
				newPoint.z = depthValue;
				newPoint.x = (sample*j - rgbIntrinsicMatrix(0,2)) * newPoint.z * rgbFocalInvertedX;
				newPoint.y = (sample*i - rgbIntrinsicMatrix(1,2)) * newPoint.z * rgbFocalInvertedY;
				//std::cout << " x " << newPoint.x << " y " << newPoint.y << " z " << newPoint.z << std::endl;
				//std::cout << " xf " << data.verticesFloatingPoint[count].x << " yf " << data.verticesFloatingPoint[count].y << " zf " << data.verticesFloatingPoint[count].z << std::endl;
			}
			/*else
			{
				newPoint.z = std::numeric_limits<float>::quiet_NaN();
				newPoint.x = std::numeric_limits<float>::quiet_NaN();
				newPoint.y = std::numeric_limits<float>::quiet_NaN();
				newPoint.r = std::numeric_limits<unsigned char>::quiet_NaN();
				newPoint.g = std::numeric_limits<unsigned char>::quiet_NaN();
				newPoint.b = std::numeric_limits<unsigned char>::quiet_NaN();
				//outputPointcloud.push_back(newPoint);
			}*/
		}
	}


	  //std::cout << depthimg.width() << std::endl;
//memcpy(depthimg.data(),  (unsigned short*)const_cast<void*>((const void*) (data.depthMap)) , w*h*sizeof(unsigned short));

	  /*for (int i = 0; i<depth.rows; i++)
		  for (int j = 0; j<depth.cols; j++)
	  {
		  std::cout << " cimg " << (double)*depthimg.data(j,i) << std::endl;
		  std::cout << " cv " << (double)depth_singlef.at<float>(i,j) << std::endl;

	  
	  }*/
	  
	  
      /*uchar* depth_buffer = depth.ptr<uchar>(0);
      if (data.depthMapFloatingPoint!=0)
  for (int i = 0; i < data.depthMapFloatingPoint.size(); i++)
  {
      float val = data.depthMapFloatingPoint[i];
    if (!saveImageFlag && !saveDepthFlag) val*=150;
    if (val<0) val=255; // catch the saturated points
      *depth_buffer++ = (unsigned char)val;
  }*/
    
	}

    //depth_cond.notify_all();
    depth_frames++;

    /*cv::namedWindow("depth_softkinetic");
    cv::imshow("depth_softkinetic",depth);


    if (saveImageFlag || saveDepthFlag) { // save a timestamped image pair; synched by depth image time
      char filename[100];
      //g_fTime = clock();
      //sprintf(filename,"df%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
      //cvSaveImage(filename,g_depthImage);
      //sprintf(filename,"vf%d.%d.jpg",(int)(g_fTime/CLOCKS_PER_SEC), (int)(g_fTime%CLOCKS_PER_SEC));
      //if (g_saveImageFlag)
    //cvSaveImage(filename,g_videoImage);
    }

    // Allow OpenCV to shut down the program
    char key = cvWaitKey(10);

    if (key==27) {
      printf("Quitting main loop from OpenCV\n");
        context.quit();
    } else
      if (key=='W') saveImageFlag = !saveImageFlag;
      else if (key=='w') saveDepthFlag = !saveDepthFlag;*/
		}
	  	  	  
}

static void onNewDepthSample_t(DepthNode node, DepthNode::NewSampleReceivedData data)
{
     reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->onNewDepthSample(node,data);
}


/*----------------------------------------------------------------------------*/
void configureAudioNode()
{
    audio_node.newSampleReceivedEvent().connect(&onNewAudioSample_t);

    AudioNode::Configuration audio_configuration = audio_node.getConfiguration();
    audio_configuration.sampleRate = 44100;

    try
    {
        context.requestControl(audio_node,0);
        audio_node.setConfiguration(audio_configuration);
        audio_node.setInputMixerLevel(0.5f);
        context.releaseControl(audio_node);

    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }
}

/*----------------------------------------------------------------------------*/
void configureDepthNode()
{
    depth_node.newSampleReceivedEvent().connect(&onNewDepthSample_t);

    DepthSense::DepthNode::DepthNode::Configuration depth_configuration(DepthSense::FRAME_FORMAT_QQVGA, 30,
                                                                        DepthSense::DepthNode::CAMERA_MODE_CLOSE_MODE,
                                                                        true);

    context.requestControl(depth_node,0);

    depth_node.setEnableVertices(true);
    depth_node.setEnableConfidenceMap(true);
    depth_node.setConfidenceThreshold(confidence_threshold);
    depth_node.setEnableVerticesFloatingPoint(true);
    depth_node.setEnableDepthMap(true);
    depth_node.setEnableDepthMapFloatingPoint(true);

    try
    {
        context.requestControl(depth_node,0);
        depth_node.setConfiguration(depth_configuration);
        context.releaseControl(depth_node);

    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }

}

/*----------------------------------------------------------------------------*/
void configureColorNode()
{
    // connect new color sample handler
    color_node.newSampleReceivedEvent().connect(&onNewColorSample_t);

    DepthSense::ColorNode::Configuration color_configuration(DepthSense::FRAME_FORMAT_VGA, 30,
                                                             DepthSense::POWER_LINE_FREQUENCY_60HZ,
                                                             DepthSense::COMPRESSION_TYPE_YUY2);
    context.requestControl(color_node,0);
    color_node.setEnableColorMap(true);

    try
    {
        context.requestControl(color_node,0);
        color_node.setConfiguration(color_configuration);
        context.releaseControl(color_node);
    }
    catch (ArgumentException& e)
    {
        printf("Argument Exception: %s\n",e.what());
    }
    catch (UnauthorizedAccessException& e)
    {
        printf("Unauthorized Access Exception: %s\n",e.what());
    }
    catch (IOException& e)
    {
        printf("IO Exception: %s\n",e.what());
    }
    catch (InvalidOperationException& e)
    {
        printf("Invalid Operation Exception: %s\n",e.what());
    }
    catch (ConfigurationException& e)
    {
        printf("Configuration Exception: %s\n",e.what());
    }
    catch (StreamingException& e)
    {
        printf("Streaming Exception: %s\n",e.what());
    }
    catch (TimeoutException&)
    {
        printf("TimeoutException\n");
    }

}

/*----------------------------------------------------------------------------*/
void configureNode(DepthSense::Node node)
{
    if ((node.is<DepthSense::DepthNode>())&&(!depth_node.isSet()))
    {
        depth_node = node.as<DepthSense::DepthNode>();
        configureDepthNode();
        context.registerNode(node);
    }

    if ((node.is<DepthSense::ColorNode>())&&(!color_node.isSet()))
    {
        color_node = node.as<DepthSense::ColorNode>();
        configureColorNode();
        context.registerNode(node);
    }

    /*if ((node.is<DepthSense::AudioNode>())&&(!audio_node.isSet()))
    {
        audio_node = node.as<DepthSense::AudioNode>();
        configureAudioNode();
        context.registerNode(node);
    }*/
}

/*----------------------------------------------------------------------------*/
void onNodeConnected(Device device, Device::NodeAddedData data)
{
    configureNode(data.node);
}

static void onNodeConnected_t(Device device, Device::NodeAddedData data)
{
     reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->onNodeConnected(device,data);
}
/*----------------------------------------------------------------------------*/
void onNodeDisconnected(Device device, Device::NodeRemovedData data)
{
    /*if (data.node.is<AudioNode>() && (data.node.as<AudioNode>() == audio_node))
        audio_node.unset();*/
    if (data.node.is<ColorNode>() && (data.node.as<ColorNode>() == color_node))
        color_node.unset();
    if (data.node.is<DepthNode>() && (data.node.as<DepthNode>() == depth_node))
        depth_node.unset();
    printf("Node disconnected\n");
}

static void onNodeDisconnected_t(Device device, Device::NodeRemovedData data)
{
     reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->onNodeDisconnected(device,data);
}

/*----------------------------------------------------------------------------*/
void onDeviceConnected(Context context, Context::DeviceAddedData data)
{
    if (!devicesFound)
    {
        data.device.nodeAddedEvent().connect(&onNodeConnected_t);
        data.device.nodeRemovedEvent().connect(&onNodeDisconnected_t);
        devicesFound = true;
    }
}

static void onDeviceConnected_t(Context context, Context::DeviceAddedData data)
{
     reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->onDeviceConnected(context,data);
}


/*----------------------------------------------------------------------------*/
void onDeviceDisconnected(Context context, Context::DeviceRemovedData data)
{
    devicesFound = false;
    printf("Device disconnected\n");
}

static void onDeviceDisconnected_t(Context context, Context::DeviceRemovedData data)
{
     reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->onDeviceDisconnected(context,data);
}


void getAvailableNodes()
{

}

/*----------------------------------------------------------------------------*/

virtual void init()
{

    saveImageFlag = false;
    saveDepthFlag = false;
	
    bool isColorizeDisp, isFixedMaxDisp;
    int imageMode;
    bool retrievedImageFlags[5];
    string filename;
    bool isVideoReading = true;
    //parseCommandLine( argc, argv, isColorizeDisp, isFixedMaxDisp, imageMode, retrievedImageFlags, filename, isVideoReading );

    cout << "Device opening ..." << endl;
    VideoCapture capture;
    if( isVideoReading )
        capture.open( filename );
    else
        capture.open( CAP_OPENNI );

    cout << "done." << endl;

    if( !capture.isOpened() )
    {
        cout << "Can not open a capture object." << endl;
        return -1;
    }

    if( !isVideoReading )
    {
        bool modeRes=false;
        switch ( imageMode )
        {
            case 0:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_VGA_30HZ );
                break;
            case 1:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_15HZ );
                break;
            case 2:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_SXGA_30HZ );
                break;
                //The following modes are only supported by the Xtion Pro Live
            case 3:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_30HZ );
                break;
            case 4:
                modeRes = capture.set( CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE, CAP_OPENNI_QVGA_60HZ );
                break;
            default:
                CV_Error( Error::StsBadArg, "Unsupported image mode property.\n");
        }
        if (!modeRes)
            cout << "\nThis image mode is not supported by the device, the default value (CV_CAP_OPENNI_SXGA_15HZ) will be used.\n" << endl;
    }

    // Print some avalible device settings.
    cout << "\nDepth generator output mode:" << endl <<
            "FRAME_WIDTH      " << capture.get( CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT     " << capture.get( CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FRAME_MAX_DEPTH  " << capture.get( CAP_PROP_OPENNI_FRAME_MAX_DEPTH ) << " mm" << endl <<
            "FPS              " << capture.get( CAP_PROP_FPS ) << endl <<
            "REGISTRATION     " << capture.get( CAP_PROP_OPENNI_REGISTRATION ) << endl;
    if( capture.get( CAP_OPENNI_IMAGE_GENERATOR_PRESENT ) )
    {
        cout <<
            "\nImage generator output mode:" << endl <<
            "FRAME_WIDTH   " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_WIDTH ) << endl <<
            "FRAME_HEIGHT  " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FRAME_HEIGHT ) << endl <<
            "FPS           " << capture.get( CAP_OPENNI_IMAGE_GENERATOR+CAP_PROP_FPS ) << endl;
    }
    else
    {
        cout << "\nDevice doesn't contain image generator." << endl;
        if (!retrievedImageFlags[0] && !retrievedImageFlags[1] && !retrievedImageFlags[2])
            return 0;
    }

    is_initialized = true;
	
	waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
	if(wdepth->isEmpty()) wdepth->getCImgList().push_back(CImg<dT>());
	CImg<dT>& depthimg=wdepth->getCImg(0);
	depthimg.resize(320,240,1,1);
	
	wdt->setCamPos((Real)(wdepth->getDimensions()[0]-1)/2.0,(Real)(wdepth->getDimensions()[1]-1)/2.0); // for perspective transforms
	wdt->update(); // update of internal data
	
	waImage wimage(this->image);
    waTransform wit(this->transform);
	if(wimage->isEmpty()) wimage->getCImgList().push_back(CImg<T>());
	CImg<T>& img=wimage->getCImg(0);
    img.resize(640, 480,1,3);
	
	wit->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
	wit->update(); // update of internal data

        // run kinect thread
		
        pthread_create( &opennicam_thread, NULL, sofa::component::container::OpenniCam::_opennicam_threadfunc, this);

        //loadCamera();
    }
	
	void getImages() const
	{
	//images.clear();
	//if (hasImageType(Reader::COLOR))
 /* {
    {
      boost::unique_lock<boost::mutex> lock(color_mutex);
      color_cond.wait(lock);
    }
    images.resize(images.size() + 1);
    color.copyTo(images.back());
  }

  //if ((hasImageType(Reader::DEPTH)) || (hasImageType(Reader::POINTS3D)))
   {
    boost::unique_lock<boost::mutex> lock(depth_mutex);
    depth_cond.wait(lock);
    //if (hasImageType(Reader::DEPTH))
    {
      images.resize(images.size() + 1);
      depth.copyTo(images.back());
    }
    //if (hasImageType(Reader::POINTS3D))
    {
      images.resize(images.size() + 1);
      points3d.copyTo(images.back());
    }
  }*/
	}


    pthread_t opennicam_thread;
    static void* _opennicam_threadfunc (void *arg) { reinterpret_cast<sofa::component::container::OpenniCam*>(arg)->softkinetic_threadfunc(); return NULL; }
//#endif


    void opennicam_threadfunc()
    {

		Mat depthMap;
        Mat validDepthMap;
        Mat disparityMap;
        Mat bgrImage;
        Mat grayImage;

        if( !capture.grab() )
        {
            cout << "Can not grab images." << endl;
            return -1;
        }
        else
        {
            if( retrievedImageFlags[0] && capture.retrieve( depthMap, CAP_OPENNI_DEPTH_MAP ) )
            {
                const float scaleFactor = 0.05f;
                Mat show; depthMap.convertTo( show, CV_8UC1, scaleFactor );
                imshow( "depth map", show );
            }

            if( retrievedImageFlags[1] && capture.retrieve( disparityMap, CAP_OPENNI_DISPARITY_MAP ) )
            {
                if( isColorizeDisp )
                {
                    Mat colorDisparityMap;
                    colorizeDisparity( disparityMap, colorDisparityMap, isFixedMaxDisp ? getMaxDisparity(capture) : -1 );
                    Mat validColorDisparityMap;
                    colorDisparityMap.copyTo( validColorDisparityMap, disparityMap != 0 );
                    imshow( "colorized disparity map", validColorDisparityMap );
                }
                else
                {
                    imshow( "original disparity map", disparityMap );
                }
            }

        }

        if( waitKey( 30 ) >= 0 )
            break;

    }


protected:

    // to kill kinect thread
    int die;

    // buffers for kinect thread
    //unsigned char *rgb_mid, *rgb_back,*depth_mid, *depth_back;

    // kinect params
/*    freenect_context *f_ctx;
    freenect_device *f_dev;
    freenect_resolution res,backup_res;
    freenect_depth_format df,backup_df;
    freenect_video_format vf,backup_vf;
    freenect_led_options led,backup_led;
    int backup_tiltangle;*/

    // flags to know if images are ready
    int got_rgb ;
    int got_depth ;

    // copy video buffers to image Data (done at init and at each simulation step)
    /*void loadCamera()
    {
        if (vf==backup_vf && df==backup_df && res==backup_res) // wait for resolution update in kinect thread
        {
            if (got_depth)
            {
                got_depth = 0;

                waDepth wdepth(this->depthImage);
                waTransform wdt(this->depthTransform);

                if(!wdepth->isEmpty())
                {
                    CImg<dT>& depth=wdepth->getCImg(0);
                    mutex_lock();
                    memcpy(depth.data(),  (unsigned short*)depth_mid , depth.width()*depth.height()*sizeof(unsigned short));
                    mutex_unlock();
                }
            }
            if (got_rgb)
            {
                got_rgb = 0;
                waImage wimage(this->image);
                waTransform wt(this->transform);

                if(!wimage->isEmpty())
                {
                    CImg<T>& rgbimg=wimage->getCImg(0);
                    mutex_lock();
                    if(rgbimg.spectrum()==3)  // deinterlace
                    {
                        unsigned char* rgb = (unsigned char*)rgb_mid;
                        unsigned char *ptr_r = rgbimg.data(0,0,0,0), *ptr_g = rgbimg.data(0,0,0,1), *ptr_b = rgbimg.data(0,0,0,2);
                        for ( int siz = 0 ; siz<rgbimg.width()*rgbimg.height(); siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++); }
                    }
                    else memcpy(rgbimg.data(),  rgb_mid, rgbimg.width()*rgbimg.height()*sizeof(T));
                    mutex_unlock();
                }
            }

            // update accelerometer data
            freenect_update_tilt_state(f_dev);
            freenect_raw_tilt_state* state = freenect_get_tilt_state(f_dev);
            double dx,dy,dz;
            freenect_get_mks_accel(state, &dx, &dy, &dz);
            this->accelerometer.setValue(Vector3(dx,dy,dz));
        }
    }*/



    void handleEvent(sofa::core::objectmodel::Event *event)
    {

        context.run();
		
    }


    void getCorners(Vec<8,Vector3> &c) // get image corners
    {
        raDepth rimage(this->depthImage);
        const imCoord dim= rimage->getDimensions();

        Vec<8,Vector3> p;
        p[0]=Vector3(-0.5,-0.5,-0.5);
        p[1]=Vector3(dim[0]-0.5,-0.5,-0.5);
        p[2]=Vector3(-0.5,dim[1]-0.5,-0.5);
        p[3]=Vector3(dim[0]-0.5,dim[1]-0.5,-0.5);
        p[4]=Vector3(-0.5,-0.5,dim[2]-0.5);
        p[5]=Vector3(dim[0]-0.5,-0.5,dim[2]-0.5);
        p[6]=Vector3(-0.5,dim[1]-0.5,dim[2]-0.5);
        p[7]=Vector3(dim[0]-0.5,dim[1]-0.5,dim[2]-0.5);

        raTransform rtransform(this->depthTransform);
        for(unsigned int i=0; i<p.size(); i++) c[i]=rtransform->fromImage(p[i]);
		
		//std::cout << " c0 " << c[0] << std::endl;
    }

    virtual void computeBBox(const core::ExecParams*  params )
    {
        if (!drawBB.getValue()) return;
        Vec<8,Vector3> c;
        getCorners(c);

        Real bbmin[3]  = {c[0][0],c[0][1],c[0][2]} , bbmax[3]  = {c[0][0],c[0][1],c[0][2]};
        for(unsigned int i=1; i<c.size(); i++)
            for(unsigned int j=0; j<3; j++)
            {
                if(bbmin[j]>c[i][j]) bbmin[j]=c[i][j];
                if(bbmax[j]<c[i][j]) bbmax[j]=c[i][j];
            }
        this->f_bbox.setValue(params,sofa::defaulttype::TBoundingBox<Real>(bbmin,bbmax));
    }
	

    void draw(const core::visual::VisualParams* vparams)
    {
        // draw bounding box

//std::cout << " depth cols " << depth.cols << std::endl;
	
	/*for (int i = 0; i<depth.rows; i++)
		  for (int j = 0; j<depth.rows; j++)
	  std::cout << (int)depth.at<uchar>(i,j) << std::endl;*/
	  

/*cv::namedWindow("depth_softkinetic");
    cv::imshow("depth_softkinetic",	depth);



    // Allow OpenCV to shut down the program
    char key = cvWaitKey(10);

    if (key==27) {
      printf("Quitting main loop from OpenCV\n");
        context.quit();
    }*/
	
        //if (!vparams->displayFlags().getShowVisualModels()) return;
        //if (!drawBB.getValue() && !drawGravity.getValue()) return;

        glPushAttrib( GL_LIGHTING_BIT | GL_ENABLE_BIT | GL_LINE_BIT );
        glPushMatrix();
		
        if (drawBB.getValue())
        {
            const float color[]= {1.,0.5,0.5,0.}, specular[]= {0.,0.,0.,0.};
            glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT_AND_DIFFUSE,color);
            glMaterialfv(GL_FRONT_AND_BACK,GL_SPECULAR,specular);
            glMaterialf(GL_FRONT_AND_BACK,GL_SHININESS,0.0);
            glColor4fv(color);
            glLineWidth(2.0);

            Vec<8,Vector3> c;
            getCorners(c);
            glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);  glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]); glVertex3d(c[2][0],c[2][1],c[2][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[0][0],c[0][1],c[0][2]); glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[1][0],c[1][1],c[1][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[5][0],c[5][1],c[5][2]); glVertex3d(c[4][0],c[4][1],c[4][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();
            glBegin(GL_LINE_LOOP);	glVertex3d(c[2][0],c[2][1],c[2][2]); glVertex3d(c[3][0],c[3][1],c[3][2]); glVertex3d(c[7][0],c[7][1],c[7][2]); glVertex3d(c[6][0],c[6][1],c[6][2]);	glEnd ();
        }

        /*if(drawGravity.getValue())
        {
            const Vec<4,float> col(0,1,0,1);
            raTransform rtransform(this->depthTransform);
            Vector3 camCenter = rtransform->fromImage(Vector3(-0.5,-0.5,-0.5));
            Vector3 acc = rtransform->qrotation.rotate(this->accelerometer.getValue());
            vparams->drawTool()->drawArrow(camCenter, camCenter+acc*showArrowSize.getValue(), showArrowSize.getValue()*0.1, col);
        }*/

        glPopMatrix ();
        glPopAttrib();
    }






};






}

}

}


#endif ///*IMAGE_SOFTKINETIC_H*/