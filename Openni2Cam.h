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
#ifndef IMAGE_OPENNI2CAM_H
#define IMAGE_OPENNI2CAM_H

#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include <iostream>
#include <map>
#include <XnCppWrapper.h>
#include <boost/thread.hpp>
#include <sys/times.h>


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

#include <boost/chrono.hpp>


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
using namespace cv;
using namespace boost;

void* globalOpenni2CamClassPointer;

class Openni2Cam : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS( Openni2Cam , Inherited);

    // image data
    typedef defaulttype::ImageUC ImageTypes;
    typedef ImageTypes::T T;
    typedef ImageTypes::imCoord imCoord;
    typedef helper::WriteAccessor<Data< ImageTypes > > waImage;
    typedef helper::ReadAccessor<Data< ImageTypes > > raImage;
    Data< ImageTypes > imageO;

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
/*cv::Mat_<cv::Vec3b> color;
cv::Mat_<uchar> depth;*/
cv::Mat_<cv::Vec3f> points3d;
//typename pcl::PointCloud<pcl::PointXYZRGB> cloud;

bool saveImageFlag;
bool saveDepthFlag;

int niterations;


cv::VideoCapture cam;

cv::Mat color, depth, depC;

public:

  typedef struct ImgContext : public boost::noncopyable
  {

    ImgContext () : is_new (false)
    {
    }

    ImgContext (const Mat & img) : image (img), is_new (false)
    {
    }
    Mat image;
    mutable boost::mutex lock;
    bool is_new;
  } ImageContext;
  
Openni2Cam();// : Inherited();
virtual void clear();
virtual ~Openni2Cam();
public:

virtual std::string getTemplateName() const	{ return templateName(this); }
static std::string templateName(const Openni2Cam* = NULL) {	return std::string(); }


virtual void init();


protected:

    // to kill kinect thread
    int die;

    // flags to know if images are ready
    int got_rgb ;
    int got_depth ;


void handleEvent(sofa::core::objectmodel::Event *event)
{

    {
		
    int32_t w, h, w_depth, h_depth;
	
	w = 640;
	h = 480;
	w_depth = 640;
	h_depth = 480;
		
	waImage wimage(this->imageO);
	CImg<T>& img =wimage->getCImg(0);
	
	waDepth wdepth(this->depthImage);
	waTransform wdt(this->depthTransform);
	CImg<dT>& depthimg =wdepth->getCImg(0);	
	
        //Grab frames
        cout << cam.grab();
        cam.retrieve( depth, CV_CAP_OPENNI_DEPTH_MAP );
        cam.retrieve( color, CV_CAP_OPENNI_BGR_IMAGE );
		
			if(img.spectrum()==3)  // deinterlace
			{
			unsigned char* rgb = (unsigned char*)color.data;
			unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
			for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++); 
					  	//cout << (int)*(rgb) << endl; 
						}
			}
			else memcpy(img.data(),  color.data, img.width()*img.height()*sizeof(T));
			
			memcpy(depthimg.data(), (float*)depth.data , w_depth*h_depth*sizeof(float));



        //Process depth -> colormap
        double min, max;
        minMaxIdx(depth, &min, &max);
        depth.convertTo(depth, CV_8UC1, 255./(max-min), -min);
        applyColorMap(depth, depC, COLORMAP_HOT);

        //Show
        imshow("Depth", depC);
        imshow("Color", color);

        char c = waitKey(1);

    }
		
}
	
    void draw(const core::visual::VisualParams* vparams)
    {
	}






};


Openni2Cam::Openni2Cam() : Inherited()
	        , depthImage(initData(&depthImage,DepthTypes(),"depthImage","depth map"))
			, depthTransform(initData(&depthTransform, TransformType(), "depthTransform" , ""))
        , imageO(initData(&imageO,ImageTypes(),"image","image"))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , deviceID ( initData ( &deviceID,(unsigned int)1,"deviceID","device ID" ) )
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

    globalOpenni2CamClassPointer = (void*) this; // used for SoftKinetic callbacks
    //depth = cv::Mat::zeros(240, 320, CV_8U);
	//color = cv::Mat::zeros(480, 640, CV_8UC3);
	drawBB = false;
    }


void Openni2Cam::clear()
    {
        /*waImage wimage(this->image);
        wimage->clear();
        waDepth wdepth(this->depthImage);
        wdepth->clear();*/
    }

Openni2Cam::~Openni2Cam()
    {
        die = 1;



    }
	
/*----------------------------------------------------------------------------*/

void Openni2Cam::init()
{

    //is_initialized = true;
	
	cam.open(CAP_OPENNI2);
	
	waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
	if(wdepth->isEmpty()) wdepth->getCImgList().push_back(CImg<dT>());
	CImg<dT>& depthimg=wdepth->getCImg(0);
	depthimg.resize(640,480,1,1);
	
	wdt->setCamPos((Real)(wdepth->getDimensions()[0]-1)/2.0,(Real)(wdepth->getDimensions()[1]-1)/2.0); // for perspective transforms
	wdt->update(); // update of internal data
	
	waImage wimage(this->imageO);
    waTransform wit(this->transform);
	if(wimage->isEmpty()) wimage->getCImgList().push_back(CImg<T>());
	CImg<T>& img = wimage->getCImg(0);
    img.resize(640, 480,1,3);
	
	wit->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
	wit->update(); // update of internal data

    }


}

}

}


#endif ///*IMAGE_SOFTKINETIC_H*/
