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
#ifndef IMAGE_REALSENSECAM_H
#define IMAGE_REALSENSECAM_H

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
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
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/defaulttype/Mat.h>
#include <sofa/defaulttype/Quat.h>
#include <sofa/helper/rmath.h>
#include <sofa/helper/OptionsGroup.h>

#ifdef Success
  #undef Success
#endif

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
using namespace rs;

void* globalRealSenseCamClassPointer;


//pthread_mutex_t backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
//pthread_cond_t frame_cond = PTHREAD_COND_INITIALIZER;


class RealSenseCam : public virtual core::objectmodel::BaseObject
{
public:
    typedef core::objectmodel::BaseObject Inherited;
    SOFA_CLASS( RealSenseCam , Inherited);

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
	
cv::Mat_<cv::Vec3f> points3d;

int niterations;

// Window size and frame rate
int INPUT_WIDTH;
int INPUT_HEIGHT;
int FRAMERATE;

// Named windows
char* WINDOW_DEPTH;
char* WINDOW_RGB;


context 	_rs_ctx;
device* 	_rs_camera;
intrinsics 	_depth_intrin;
intrinsics  _color_intrin;
bool 		_loop = true;


public:
  
RealSenseCam();// : Inherited();
virtual void clear();
virtual ~RealSenseCam();

virtual std::string getTemplateName() const	{ return templateName(this); }
static std::string templateName(const RealSenseCam* = NULL) {	return std::string(); }


virtual void init();


protected:
	

void handleEvent(sofa::core::objectmodel::Event *event)
    {

	// Get current frames intrinsic data.
	_depth_intrin 	= _rs_camera->get_stream_intrinsics( rs::stream::depth );
	_color_intrin 	= _rs_camera->get_stream_intrinsics( rs::stream::color );

	// Create depth image
	cv::Mat depth16( _depth_intrin.height,
					 _depth_intrin.width,
					 CV_16U,
					 (uchar *)_rs_camera->get_frame_data( rs::stream::depth ) );

	// Create color image
	cv::Mat rgb( _color_intrin.height,
				 _color_intrin.width,
				 CV_8UC3,
				 (uchar *)_rs_camera->get_frame_data( rs::stream::color ) );

	// < 800
	cv::Mat depth8u = depth16;
	depth8u.convertTo( depth8u, CV_8UC1, 255.0/1000 );

	    // Read the color buffer and display
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


        cv::Mat bgr_image;
        cvtColor (rgb, bgr_image, CV_RGB2BGR);
		  
		  if(img.spectrum()==3)  // deinterlace
			{
			unsigned char* rgb = (unsigned char*)bgr_image.data;
			unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
			for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++); 
						}
			}
			else memcpy(img.data(),  bgr_image.data, img.width()*img.height()*sizeof(T));

          memcpy(depthimg.data(), (float*)depth16.data , w_depth*h_depth*sizeof(float));
	/*imshow( WINDOW_DEPTH, depth8u );
	cvWaitKey( 1 );

	cv::cvtColor( rgb, rgb, cv::COLOR_BGR2RGB );
	imshow( WINDOW_RGB, rgb );
	cvWaitKey( 1 );

	return true;*/
		
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


RealSenseCam::RealSenseCam() : Inherited()
	        , depthImage(initData(&depthImage,DepthTypes(),"depthImage","depth map"))
			, depthTransform(initData(&depthTransform, TransformType(), "depthTransform" , ""))
        , imageO(initData(&imageO,ImageTypes(),"image","image"))
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
    {

    globalRealSenseCamClassPointer = (void*) this; 
	drawBB = false;

// Window size and frame rate
INPUT_WIDTH 	= 320;
INPUT_HEIGHT 	= 240;
FRAMERATE 	= 60;

// Named windows
WINDOW_DEPTH = "Depth Image";
WINDOW_RGB	 = "RGB Image";
_rs_camera = NULL;
_loop = true;

    }


void RealSenseCam::clear()
    {
    }

RealSenseCam::~RealSenseCam()
    {

    }
	
/*----------------------------------------------------------------------------*/

void RealSenseCam::init()
{


	bool success = false;
	if( _rs_ctx.get_device_count( ) > 0 )
	{
		_rs_camera = _rs_ctx.get_device( 0 );

		_rs_camera->enable_stream( rs::stream::color, INPUT_WIDTH, INPUT_HEIGHT, rs::format::rgb8, FRAMERATE );
		_rs_camera->enable_stream( rs::stream::depth, INPUT_WIDTH, INPUT_HEIGHT, rs::format::z16, FRAMERATE );

		_rs_camera->start( );

		success = true;
	}
	
	/*if( !success )
	{
		std::cout << "Unable to locate a camera" << std::endl;
		rs::log_to_console( rs::log_severity::fatal );
		return EXIT_FAILURE;
	}*/
	
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
