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
#ifndef SOFA_IMAGE_REALSENSECAM_H
#define SOFA_IMAGE_REALSENSECAM_H


#include <image/config.h>
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
#include <sofa/helper/gl/Color.h>

#define GL_GLEXT_PROTOTYPES 1
#define GL4_PROTOTYPES 1
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glext.h>
#include <GL/glu.h>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>
#include <map>
#include <boost/thread.hpp>
#include <sys/times.h>

#include <fstream>
#include <algorithm>
#include <cstring>

#include <chrono>
#include <thread>

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
using namespace rs2;


void* globalRealSenseCamClassPointer;

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

    Data<helper::OptionsGroup> resolution;
    Data<int> depthMode;
    Data<bool> drawBB;
    Data<float> showArrowSize;
    Data<int> depthScale;

    int widthc, heightc;
    int widthd, heightd;

// Declare depth colorizer for pretty visualization of depth data
rs2::colorizer color_map;
CImg<T> img,depthimg;

// Declare RealSense pipeline, encapsulating the actual device and sensors
rs2::pipeline pipe;
// Start streaming with default recommended configuration

public:

RealSenseCam();// : Inherited();
virtual void clear();
virtual ~RealSenseCam();

virtual std::string getTemplateName() const	{ return templateName(this); }
static std::string templateName(const RealSenseCam* = NULL) {	return std::string(); }

virtual void init();
void initRaw();
void initAlign();

protected:

void acquireRaw()
{

    rs2::frameset data;
    data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

    // Trying to get both color and depth frames
     rs2::video_frame color = data.get_color_frame();
     rs2::depth_frame depth = data.get_depth_frame();

    // Create depth image
    if (depth && color){

     widthc = color.get_width();
     heightc = color.get_height();

     cv::Mat rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data());

     widthd = depth.get_width();
     heightd = depth.get_height();

     cv::Mat depth16,depth32;

     cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );
     depth16=depth160.clone();
     depth16.convertTo(depth32, CV_32F,(float)1/8190);

     widthc = color.get_width();
     heightc = color.get_height();

    waImage wimage(this->imageO);
    img =wimage->getCImg(0);

    waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
    depthimg =wdepth->getCImg(0);

    cv::Mat bgr_image;
    cvtColor (rgb0, bgr_image, CV_RGB2BGR);

            if(img.spectrum()==3)
            {
            unsigned char* rgb = (unsigned char*)color.get_data();
            unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
                for ( int siz = 0 ; siz<widthc*heightc; siz++)
                {
                    *(ptr_r++) = *(rgb++);
                    *(ptr_g++) = *(rgb++);
                    *(ptr_b++) = *(rgb++);
                }
            }

    memcpy(depthimg.data(), (float*)depth32.data , widthd*heightd*sizeof(float));

}

}

void acquireAligned()
{
    rs2::align align(RS2_STREAM_COLOR);
    rs2::frameset frameset;

    while (!frameset.first_or_default(RS2_STREAM_DEPTH) || !frameset.first_or_default(RS2_STREAM_COLOR))
    {
        frameset = pipe.wait_for_frames();
    }

    auto processed = align.process(frameset);

   // Trying to get both color and aligned depth frames
    rs2::video_frame color = processed.get_color_frame();
    rs2::depth_frame depth = processed.get_depth_frame();

    if (depth && color)
    {

    widthc = color.get_width();
    heightc = color.get_height();

    cv::Mat depth16, depth32,depth8u;

    // Create depth image
    widthd = depth.get_width();
    heightd = depth.get_height();

    cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );

    depth160.convertTo(depth32, CV_32F,(float)1/8190*depthScale.getValue());
    // Read the color buffer and display

    waImage wimage(this->imageO);
    img = wimage->getCImg(0);

    waDepth wdepth(this->depthImage);
    //depthimg =wdepth->getCImg(0);

        if(img.spectrum()==3)
        {
        unsigned char* rgb = (unsigned char*)color.get_data();
        unsigned char *ptr_b = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_r = img.data(0,0,0,0);
            for ( int siz = 0 ; siz<widthc*heightc; siz++)
            {
                *(ptr_r++) = *(rgb++);
                *(ptr_g++) = *(rgb++);
                *(ptr_b++) = *(rgb++);
            }
        }
    //memcpy(depthimg.data(), (float*)depth32.data , widthd*heightd*sizeof(float));
    }

}

void handleEvent(sofa::core::objectmodel::Event *event)
{
if (dynamic_cast<simulation::AnimateEndEvent*>(event))
{
       if(this->depthMode.getValue()==0) acquireRaw();
        else acquireAligned();
}

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
    GLfloat projectionMatrixData[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrixData);
    GLfloat modelviewMatrixData[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelviewMatrixData);

    std::cout << " width " << widthc << " " << heightc << std::endl;

    std::stringstream imageString;
    unsigned char *ptr_b = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_r = img.data(0,0,0,0);
    for ( int siz = 0 ; siz<img.width()*img.height(); siz++)
    {
        imageString.write((const char*)ptr_r++, sizeof(T));
        imageString.write((const char*)ptr_g++, sizeof(T));
        imageString.write((const char*)ptr_b++, sizeof(T));

    }
    // PERSPECTIVE
    glMatrixMode(GL_PROJECTION);	//init the projection matrix
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);  // orthogonal view
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    // BACKGROUND TEXTURING
    //glDepthMask (GL_FALSE);		// disable the writing of zBuffer
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_TEXTURE_2D);	// enable the texture
    glDisable(GL_LIGHTING);		// disable the light
    glBindTexture(GL_TEXTURE_2D, 0);  // texture bind
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, widthc, heightc, 0, GL_RGB, GL_UNSIGNED_BYTE, imageString.str().c_str());

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);	// Linear Filtering

                                                                        // BACKGROUND DRAWING
                                                                        //glEnable(GL_DEPTH_TEST);

    glBegin(GL_QUADS); //we draw a quad on the entire screen (0,0 1,0 1,1 0,1)
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glTexCoord2f(0, 1);		glVertex2f(0, 0);
    glTexCoord2f(1, 1);		glVertex2f(1, 0);
    glTexCoord2f(1, 0);		glVertex2f(1, 1);
    glTexCoord2f(0, 0);		glVertex2f(0, 1);
    glEnd();

    //glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);		// enable light
    glDisable(GL_TEXTURE_2D);	// disable texture 2D
    glEnable(GL_DEPTH_TEST);
    //glDepthMask (GL_TRUE);		// enable zBuffer

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);


    vparams->drawTool()->restoreLastState();

    }
};


RealSenseCam::RealSenseCam() : Inherited()
        , depthImage(initData(&depthImage,DepthTypes(),"depthImage","depth map"))
        , depthTransform(initData(&depthTransform, TransformType(), "depthTransform" , ""))
        , imageO(initData(&imageO,ImageTypes(),"image","image"))
        , transform(initData(&transform, TransformType(), "transform" , ""))
        , resolution ( initData ( &resolution,"resolution","resolution" ))
        , depthMode ( initData ( &depthMode,1,"depthMode","depth mode" ))
        , drawBB(initData(&drawBB,false,"drawBB","draw bounding box"))
        , depthScale(initData(&depthScale,1,"depthScale","scale for the depth values, 1 for SR300, 10 for 435"))
    {
    globalRealSenseCamClassPointer = (void*) this;
    this->addAlias(&imageO, "inputImage");
    this->addAlias(&transform, "inputTransform");
    transform.setGroup("Transform");
    depthTransform.setGroup("Transform");
    f_listening.setValue(true);  // to update camera during animate
    drawBB = false;
    }


void RealSenseCam::clear()
    {
        waImage wimage(this->imageO);
        wimage->clear();
        waDepth wdepth(this->depthImage);
        wdepth->clear();
    }

RealSenseCam::~RealSenseCam()
    {
	clear();
    }
	
/*----------------------------------------------------------------------------*/

void RealSenseCam::initRaw()
{
    pipe.start();
    rs2::frameset data;
    for (int i = 0; i < 100 ; i++)
    data = pipe.wait_for_frames(); // Wait for next set of frames from the camera
    rs2::video_frame depth = data.get_depth_frame();
    rs2::video_frame color = data.get_color_frame();
    while(!depth || !color)
    {
    data = pipe.wait_for_frames(); // Wait for next set of frames from the camera

    depth = data.get_depth_frame(); // Find and colorize the depth data
    color = data.get_color_frame(); // Find the color data
    }

    widthd = depth.get_width();
    heightd = depth.get_height();

    widthc = color.get_width();
    heightc = color.get_height();

    waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
    if(wdepth->isEmpty()) wdepth->getCImgList().push_back(CImg<dT>());
    depthimg=wdepth->getCImg(0);
    depthimg.resize(widthd,heightd,1,1);

    wdt->setCamPos((Real)(wdepth->getDimensions()[0]-1)/2.0,(Real)(wdepth->getDimensions()[1]-1)/2.0); // for perspective transforms
    wdt->update(); // update of internal data

     waImage wimage(this->imageO);
     waTransform wit(this->transform);
    if(wimage->isEmpty()) wimage->getCImgList().push_back(CImg<T>());
    CImg<T>& img = wimage->getCImg(0);
    img.resize(widthc,heightc,1,3);

    wit->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
    wit->update(); // update of internal data

    cv::Mat rgb,depth8u,depth16, depth32;

    // Create depth image

    cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );
    depth16=depth160.clone();
    depth16.convertTo(depth32, CV_32F, (float)1/8190);
    cv::Mat rgb0(heightc,widthc, CV_8UC3, (void*) color.get_data());

    //depth16.convertTo( depth8u, CV_8UC1, 255.0/1000 );
    //cv::imwrite("bgr.png", depth8u);

            if(img.spectrum()==3)
            {
            unsigned char* rgb = (unsigned char*)color.get_data();
            unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
            for ( int siz = 0 ; siz<widthc*heightc; siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++);
                        }
            }

    memcpy(depthimg.data(), (float*)depth32.data , widthd*heightd*sizeof(float));

}

void RealSenseCam::initAlign()
{
    pipe.start();
    rs2::align align(RS2_STREAM_COLOR);
    rs2::frameset frameset;

        for (int it= 0; it < 100 ; it++)
        frameset = pipe.wait_for_frames();

        while ((!frameset.first_or_default(RS2_STREAM_DEPTH) || !frameset.first_or_default(RS2_STREAM_COLOR)))
        {
            frameset = pipe.wait_for_frames();
        }

    auto processed = align.process(frameset);

    // Trying to get both color and aligned depth frames
    rs2::video_frame color = processed.get_color_frame();
    rs2::depth_frame depth = processed.get_depth_frame();

    widthd = depth.get_width();
    heightd = depth.get_height();

    widthc = color.get_width();
    heightc = color.get_height();

    waDepth wdepth(this->depthImage);
    waTransform wdt(this->depthTransform);
    if(wdepth->isEmpty()) wdepth->getCImgList().push_back(CImg<dT>());
    depthimg=wdepth->getCImg(0);
    depthimg.resize(widthd,heightd,1,1);

    wdt->setCamPos((Real)(wdepth->getDimensions()[0]-1)/2.0,(Real)(wdepth->getDimensions()[1]-1)/2.0); // for perspective transforms
    wdt->update(); // update of internal data

     waImage wimage(this->imageO);
     waTransform wit(this->transform);
    if(wimage->isEmpty()) wimage->getCImgList().push_back(CImg<T>());
    img = wimage->getCImg(0);
    img.resize(widthc,heightc,1,3);

    wit->setCamPos((Real)(wimage->getDimensions()[0]-1)/2.0,(Real)(wimage->getDimensions()[1]-1)/2.0); // for perspective transforms
    wit->update(); // update of internal data
    cv::Mat rgb,depth8u,depth16, depth32;

    // Create depth image

    cv::Mat depth160( heightd, widthd, CV_16U, (void*)depth.get_data() );
    depth16=depth160.clone();
    depth16.convertTo(depth32, CV_32F, (float)1/8190);

            if(img.spectrum()==3)
            {
            unsigned char* rgb = (unsigned char*)color.get_data();
            unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
                for ( int siz = 0 ; siz<widthc*heightc; siz++)
                {
                    *(ptr_r++) = *(rgb++);
                    *(ptr_g++) = *(rgb++);
                    *(ptr_b++) = *(rgb++);
                }
            }

    //memcpy(depthimg.data(), (float*)depth32.data , widthd*heightd*sizeof(float));

}

void RealSenseCam::init()
{

   if(this->depthMode.getValue()==0)
   {
       initRaw();
       //acquireRaw();
   }
   else
   {
       initAlign();
       //acquireAligned();
   }


}


}

}

}


#endif ///*IMAGE_SOFTKINETIC_H*/
