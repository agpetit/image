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

#include "openni_driver.h"
#include "openni_device.h"
#include "openni_image.h"
#include "openni_depth_image.h"
#include <iostream>
#include <string>
#include <map>
#include <XnCppWrapper.h>
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

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <boost/thread/mutex.hpp>
#include <boost/make_shared.hpp>

/*#include <pcl/io/io.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>*/

#include <visp/vpIoTools.h>
#include <visp/vpImageIo.h>
#include <visp/vpParseArgv.h>



//#include <libfreenect.h>
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


//pthread_mutex_t backbuf_mutex = PTHREAD_MUTEX_INITIALIZER;
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
  
OpenniCam();// : Inherited();
virtual void clear();
virtual ~OpenniCam();
void writeImages ();

void imageCallback (boost::shared_ptr<Image> image, void* cookie);
void depthCallback (boost::shared_ptr<DepthImage> depth, void* cookie);

private:

  map<string, ImageContext*> rgb_images_;
  map<string, ImageContext*> gray_images_;
  map<string, ImageContext*> depth_images_;
  vector< boost::shared_ptr<OpenNIDevice> > devices_;
  bool running_;
  unsigned selected_device_;

  double image_timestamp;
  double depth_timestamp;

			public:

virtual std::string getTemplateName() const	{ return templateName(this); }
static std::string templateName(const OpenniCam* = NULL) {	return std::string(); }


virtual void init();


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
	
void run1 ()
{
		  /*if (devices_[selected_device_]->isDepthStreamRunning ())
            devices_[selected_device_]->stopDepthStream ();
          else*/
		  
			/*depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.lock ();
            //depth_images_[devices_[selected_device_]->getConnectionString ()]->image.create (300, 400, CV_32FC1);
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.rows = 240;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.cols = 320;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.unlock ();
            devices_[selected_device_]->setDepthCropping (100, 100, 320, 240);*/
            devices_[selected_device_]->startDepthStream ();
            devices_[selected_device_]->startImageStream ();
			
		
}
	
void run ()
{
  running_ = true;
  
  int t = (int)this->getContext()->getTime();
		
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
	
	cout << "ok run 1" << endl;

	
  //try
  {
    while (running_)
    {
		
	cout << "ok run 1" << endl;
		
      for (map<string, ImageContext*>::iterator imageIt = rgb_images_.begin (); imageIt != rgb_images_.end (); ++imageIt)
      {
        //if (imageIt->second->is_new && imageIt->second->lock.try_lock ())
        {
          cv::Mat bgr_image;
          cvtColor (imageIt->second->image, bgr_image, CV_RGB2BGR);
          //imshow (imageIt->first + "RGB", bgr_image);
          imageIt->second->is_new = false;
          imageIt->second->lock.unlock ();
		  
		  if(img.spectrum()==3)  // deinterlace
			{
			unsigned char* rgb = (unsigned char*)bgr_image.data;
			unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
			for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++); 
					  	//cout << (int)*(rgb) << endl; 
						}
			}
			else memcpy(img.data(),  bgr_image.data, img.width()*img.height()*sizeof(T));
        }
      }
	  
	  cout << "ok run 2" << endl;

      for (map<string, ImageContext*>::iterator imageIt = gray_images_.begin (); imageIt != gray_images_.end (); ++imageIt)
      {
        //if (imageIt->second->is_new && imageIt->second->lock.try_lock ())
        {
          //imshow (imageIt->first + "Gray", imageIt->second->image);
          imageIt->second->is_new = false;
          imageIt->second->lock.unlock ();
        }
      }

      for (map<string, ImageContext*>::iterator imageIt = depth_images_.begin (); imageIt != depth_images_.end (); ++imageIt)
      {
       //if (imageIt->second->is_new && imageIt->second->lock.try_lock ())
        {// depth image is in range 0-10 meter -> convert to 0-255 values
          Mat gray_image;
          imageIt->second->image.convertTo (gray_image, CV_8UC1, 25.5);
          //imshow (imageIt->first + "Depth", gray_image);
          imageIt->second->is_new = false;
          imageIt->second->lock.unlock ();
		  
		  	//cout << "ok copy depth " << endl;
		  memcpy(depthimg.data(), (float*)imageIt->second->image.data , w_depth*h_depth*sizeof(float));

        }
      }

      unsigned char key = waitKey (30) & 0xFF;
	  
            devices_[selected_device_]->startDepthStream ();
            devices_[selected_device_]->startImageStream ();
			
			/*depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.lock ();
            //depth_images_[devices_[selected_device_]->getConnectionString ()]->image.create (300, 400, CV_32FC1);
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.rows = 240;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.cols = 320;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.unlock ();*/

      /*switch (key)
      {
        case 27:
        case 'q':
        case 'Q': running_ = false;
          break;

        case '1':
          selected_device_ = 0;
          break;
        case '2':
          selected_device_ = 1;
          break;
        case '3':
          selected_device_ = 2;
          break;

        case 'r':
        case 'R':
          devices_[selected_device_]->setDepthRegistration (!devices_[selected_device_]->isDepthRegistered ());
          break;
        case 's':
        case 'S':
          if (devices_[selected_device_]->isSynchronizationSupported ())
            devices_[selected_device_]->setSynchronization (!devices_[selected_device_]->isSynchronized ());
          break;
        case 'c':
        case 'C':
          if (devices_[selected_device_]->isDepthCropped ())
          {
            depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.lock ();
            //depth_images_[devices_[selected_device_]->getConnectionString ()]->image.create (480, 640, CV_32FC1);
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.rows = 480;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.cols = 640;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.unlock ();
            devices_[selected_device_]->setDepthCropping (0, 0, 0, 0);
          }
          else if (devices_[selected_device_]->isDepthCroppingSupported ())
          {
            depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.lock ();
            //depth_images_[devices_[selected_device_]->getConnectionString ()]->image.create (300, 400, CV_32FC1);
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.rows = 240;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->image.cols = 320;
            depth_images_[devices_[selected_device_]->getConnectionString ()]->lock.unlock ();
            devices_[selected_device_]->setDepthCropping (100, 100, 320, 240);
			//devices_[selected_device_]->setDepthCropping (0, 0, 0, 0);
          }
          break;

        case 'd':
        case 'D':
          if (devices_[selected_device_]->isDepthStreamRunning ())
            devices_[selected_device_]->stopDepthStream ();
          else
            devices_[selected_device_]->startDepthStream ();
          break;
        case 'i':
        case 'I':
          if (devices_[selected_device_]->isImageStreamRunning ())
            devices_[selected_device_]->stopImageStream ();
          else
            devices_[selected_device_]->startImageStream ();
          break;

        case 'w':
        case 'W':
          writeImages ();
          break;
      }*/
    }
  }
  /*catch (const OpenNIException& exception)
  {
    cout << "exception caught: " << exception.what () << endl;
  }
  catch (...)
  {

    cout << "unknown exception caught" << endl;
  }*/
}	

void handleEvent(sofa::core::objectmodel::Event *event)
    {

        run1();
		
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


OpenniCam::OpenniCam() : Inherited() : grabber_ (device_id)
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
        , die(0)
        , got_rgb(0)
        , got_depth(0)
		, most_recent_frame_ ()
        , frame_counter_ (0)
        , use_trigger_ (false)
        , trigger_ (false)
    {

    globalOpenniCamClassPointer = (void*) this; // used for SoftKinetic callbacks
    //depth = cv::Mat::zeros(240, 320, CV_8U);
	//color = cv::Mat::zeros(480, 640, CV_8UC3);
	drawBB = false;

    }


void OpenniCam::clear()
    {
        /*waImage wimage(this->image);
        wimage->clear();
        waDepth wdepth(this->depthImage);
        wdepth->clear();*/
    }

OpenniCam::~OpenniCam()
    {
        die = 1;

  // Stop the grabber when shutting down
  grabber_.stop ();

    }

void OpenniCam::setTriggerMode (bool use_trigger)
{
  use_trigger_ = use_trigger;
}

const PointCloudPtr OpenniCam::snap ()
{
  if (use_trigger_)
  {
    if (!preview_)
    {
      // Initialize the visualizer ONLY if use_trigger is set to true
      preview_ = boost::shared_ptr<pcl::visualization::PCLVisualizer> (new pcl::visualization::PCLVisualizer ());

      boost::function<void (const pcl::visualization::KeyboardEvent&)> keyboard_cb =
	  boost::bind (&OpenNICapture::onKeyboardEvent, this, _1);

      preview_->registerKeyboardCallback (keyboard_cb);
    }
    waitForTrigger ();
  }
  // Wait for a fresh frame
  int old_frame = frame_counter_;
  while (frame_counter_ == old_frame) continue;
  return (most_recent_frame_);
}

    // callbacks with static wrappers
    //static void _depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)    { reinterpret_cast<sofa::component::container::SoftKinetic*>(globalSoftKineticClassPointer)->depth_cb(dev, v_depth, timestamp);  }

void OpenniCam::imageCallback (boost::shared_ptr<Image> image, void* cookie)
{
  timeval timestamp;
  gettimeofday (&timestamp, NULL);

  double now = timestamp.tv_sec + timestamp.tv_usec * 0.000001;
//  double diff1 = min (fabs (now - depth_timestamp), fabs (depth_timestamp - image_timestamp));
//  double diff2 = max (fabs (now - depth_timestamp), fabs (depth_timestamp - image_timestamp));
  //cout << diff1 * 1000.0 << "\tms vs. " << diff2 * 1000.0 << endl;

  image_timestamp = now;
  OpenNIDevice* device = reinterpret_cast<OpenNIDevice*>(cookie);
  ImageContext* rgb_image_context = rgb_images_[device->getConnectionString ()];
  ImageContext* gray_image_context = gray_images_[device->getConnectionString ()];
  
   int32_t w, h;
  
  	w = 640;
	h = 480;
		
	waImage wimage(this->imageO);
	CImg<T>& img =wimage->getCImg(0);

  // lock image so it does not get drawn
  unique_lock<mutex> rgb_lock (rgb_image_context->lock);
  unsigned char* rgb_buffer = (unsigned char*)(rgb_image_context->image.data + (rgb_image_context->image.cols >> 2) * rgb_image_context->image.elemSize () +
                                              (rgb_image_context->image.rows >> 2) * rgb_image_context->image.step);
  image->fillRGB (rgb_image_context->image.cols >> 1, rgb_image_context->image.rows >> 1, rgb_buffer, rgb_image_context->image.step);
  
/*
  unsigned char* rgb_buffer = (unsigned char*)(rgb_image_context->image.data + (rgb_image_context->image.cols >> 3 ) * 3 * rgb_image_context->image.elemSize () +
                                              (rgb_image_context->image.rows >> 3 ) * 3 * rgb_image_context->image.step);
  image->fillRGB (rgb_image_context->image.cols >> 2, rgb_image_context->image.rows >> 2, rgb_buffer, rgb_image_context->image.step);
*/
  //image->fillRGB (rgb_image_context->image.cols, rgb_image_context->image.rows, rgb_image_context->image.data, rgb_image_context->image.step);
  /*
  cv::Mat raw (image->getHeight(), image->getWidth(), CV_8UC1);
  image->fillRaw (raw.data);

  static int calls = 0;
  if (++calls % 30)
  {
    int index = calls / 30;
    char filename [1024];
    sprintf (filename, "image_%03d.png", index);
    imwrite (filename, raw);
  }
  imshow ("raw", raw);
  */
  
        for (map<string, ImageContext*>::iterator imageIt = rgb_images_.begin (); imageIt != rgb_images_.end (); ++imageIt)
      {
        //if (imageIt->second->is_new && imageIt->second->lock.try_lock ())
        {
          cv::Mat bgr_image;
          cvtColor (imageIt->second->image, bgr_image, CV_RGB2BGR);
          //imshow (imageIt->first + "RGB", bgr_image);
          imageIt->second->is_new = false;
          imageIt->second->lock.unlock ();
		  

		  if(img.spectrum()==3)  // deinterlace
			{
			unsigned char* rgb = (unsigned char*)bgr_image.data;
			unsigned char *ptr_r = img.data(0,0,0,2), *ptr_g = img.data(0,0,0,1), *ptr_b = img.data(0,0,0,0);
			for ( int siz = 0 ; siz<img.width()*img.height(); siz++)    { *(ptr_r++) = *(rgb++); *(ptr_g++) = *(rgb++); *(ptr_b++) = *(rgb++); 
					  	//cout << (int)*(rgb) << endl; 
						}
			}
			else memcpy(img.data(),  bgr_image.data, img.width()*img.height()*sizeof(T));
        }
      }
	  
	  
  rgb_image_context->is_new = true;
  rgb_lock.unlock ();

  unique_lock<mutex> gray_lock (gray_image_context->lock);

  unsigned char* gray_buffer = (unsigned char*)(gray_image_context->image.data + (gray_image_context->image.cols >> 2) +
                                               (gray_image_context->image.rows >> 2) * gray_image_context->image.step);
  image->fillGrayscale (gray_image_context->image.cols >> 1, gray_image_context->image.rows >> 1, gray_buffer, gray_image_context->image.step);
  //image->fillGrayscale (gray_image_context->image.cols, gray_image_context->image.rows, gray_image_context->image.data, gray_image_context->image.step);
  gray_image_context->is_new = true;
}

/*static void imageCallback_t(boost::shared_ptr<Image> image, void* cookie)
{
     this->imageCallback(image,cookie);
}*/

void OpenniCam::depthCallback (boost::shared_ptr<DepthImage> depth, void* cookie)
{
	
  timeval timestamp;
  gettimeofday (&timestamp, NULL);
  depth_timestamp = timestamp.tv_sec + timestamp.tv_usec * 0.000001;

  OpenNIDevice* device = reinterpret_cast<OpenNIDevice*>(cookie);
  ImageContext* depth_image_context = depth_images_[device->getConnectionString ()];
  
  int32_t w_depth, h_depth;
	
	w_depth = 640;
	h_depth = 480;
	
	waDepth wdepth(this->depthImage);
	waTransform wdt(this->depthTransform);
	CImg<dT>& depthimg =wdepth->getCImg(0);	

  // lock depth image so it does not get drawn
  unique_lock<mutex> depth_lock (depth_image_context->lock);
  float* buffer = (float*)(depth_image_context->image.data + (depth_image_context->image.cols >> 2) * sizeof(float) +
                          (depth_image_context->image.rows >> 2) * depth_image_context->image.step );
  depth->fillDepthImage (depth_image_context->image.cols >> 1, depth_image_context->image.rows >> 1, buffer, depth_image_context->image.step);
  //depth.fillDepthImage (depth_image_context->image.cols, depth_image_context->image.rows, (float*)depth_image_context->image.data, depth_image_context->image.step);
  depth_image_context->is_new = true;
  
	for (map<string, ImageContext*>::iterator imageIt = depth_images_.begin (); imageIt != depth_images_.end (); ++imageIt)
      {
       //if (imageIt->second->is_new && imageIt->second->lock.try_lock ())
        {// depth image is in range 0-10 meter -> convert to 0-255 values
          Mat gray_image;
          imageIt->second->image.convertTo (gray_image, CV_8UC1, 100);
          //imshow (imageIt->first + "Depth", gray_image);
          imageIt->second->is_new = false;
          imageIt->second->lock.unlock ();
		  
		  	cout << "ok copy depth " << depth_image_context->image.cols << endl;
		  
		 memcpy(depthimg.data(), (float*)imageIt->second->image.data , w_depth*h_depth*sizeof(float));
		  
		//memcpy(depthimg.data(), (float*)gray_image.data , w_depth*h_depth*sizeof(unsigned char));

		  //memcpy(depthimg.data(), imageIt->second->image.data , w_depth*h_depth*sizeof(float));
		  //imwrite ("depth.png", imageIt->second->image);
		  
		/*for (int i = 0; i< 240; i++)
		  for (int j = 0; j< 320; j++)
	      std::cout << (int)gray_image.at<uchar>(i,j) << std::endl;
			//std::cout << (double)*depthimg.data(i,j,0,0)<< std::endl;*/

        }
      }
	    depth_lock.unlock ();
  
}



/*static void depthCallback_t(boost::shared_ptr<DepthImage> depth, void* cookie)
{
     this->depthCallback(depth,cookie);
}*/
	
/*----------------------------------------------------------------------------*/

void OpenniCam::init()
{

    saveImageFlag = false;
    saveDepthFlag = false;
	
  OpenNIDriver& driver = OpenNIDriver::getInstance ();
  //if (argc == 1)
  {
    if (driver.getNumberDevices () > 0)
    {
      for (unsigned deviceIdx = 0; deviceIdx < driver.getNumberDevices (); ++deviceIdx)
      {
        cout << "Device: " << deviceIdx << ", vendor: " << driver.getVendorName (deviceIdx) << ", product: " << driver.getProductName (deviceIdx)
                << ", connected: " << (int)driver.getBus (deviceIdx) << " @ " << (int)driver.getAddress (deviceIdx) << ", serial number: \'" << driver.getSerialNumber (deviceIdx) << "\'" << endl;
      }
    }
    else
      cout << "No devices connected." << endl;
    //exit (1);
  }

  vector <unsigned> device_indices;
  //for (int argIdx = 1; argIdx < argc; ++argIdx)
  {
	  
    unsigned deviceIdx = 0;//(unsigned)atoi (argv[argIdx]);
    if (deviceIdx >= driver.getNumberDevices ())
    {
      if (driver.getNumberDevices () > 0)
      {
        cout << "Device index out of range. " << driver.getNumberDevices () << " devices found." << endl;
        for (unsigned deviceIdx = 0; deviceIdx < driver.getNumberDevices (); ++deviceIdx)
        {
          cout << "Device: " << deviceIdx << ", vendor: " << driver.getVendorName (deviceIdx) << ", product: "
                  << driver.getProductName (deviceIdx) << ", connected: " << (int)driver.getBus (deviceIdx) << " @ "
                  << (int)driver.getAddress (deviceIdx) << ", serial number: \'" << driver.getSerialNumber (deviceIdx) << "\'" << endl;
        }
      }
      else
        cout << "No devices connected." << endl;
      exit (-1);
    }
    device_indices.push_back ((unsigned)deviceIdx);
  }
  

  cout << "<1,2,3...> to select device" << endl;
  cout << "<I> to start or stop image stream of selected device" << endl;
  cout << "<D> to start or stop depth stream of selected device" << endl;
  cout << "<R> to turn on or off registration for selected device" << endl;
  cout << "<S> to turn on or off synchronization for selected device" << endl;
  cout << "<C> to turn on or off image cropping for selected device" << endl;
  cout << "<W> write current images" << endl;
  cout << "<Q> to quit application" << endl;	
	
  //OpenNIDriver& driver = OpenNIDriver::getInstance ();

  for (vector<unsigned>::const_iterator indexIt = device_indices.begin (); indexIt != device_indices.end (); ++indexIt)
  {
    if (*indexIt >= driver.getNumberDevices ())
    {
      cout << "Index out of range." << driver.getNumberDevices () << " devices found." << endl;
      exit (1);
    }

    boost::shared_ptr<OpenNIDevice> device = driver.getDeviceByIndex (*indexIt);
    cout << devices_.size () + 1 << ". device on bus: " << (int)device->getBus () << " @ " << (int)device->getAddress ()
            << " with serial number: " << device->getSerialNumber () << "  "
            << device->getVendorName () << " : " << device->getProductName () << endl;
    devices_.push_back (device);

    const int width = 640;
    const int height = 480;
    XnMapOutputMode mode;
    mode.nXRes = width;
    mode.nYRes = height;
    mode.nFPS = 30;
	cout << "ok init 1" << endl;
    
    if (device->hasImageStream())
    {
      if (!device->isImageModeSupported (mode))
      {
        cout << "image stream mode " << mode.nXRes << " x " << mode.nYRes << " @ " << mode.nFPS << " not supported" << endl;
        exit (-1);
      }
      //namedWindow (string (device->getConnectionString ()) + "RGB", WINDOW_AUTOSIZE);
      //namedWindow (string (device->getConnectionString ()) + "Gray", WINDOW_AUTOSIZE);
      rgb_images_[device->getConnectionString ()] = new ImageContext (Mat::zeros (height, width, CV_8UC3));
      gray_images_[device->getConnectionString ()] = new ImageContext (Mat::zeros (height, width, CV_8UC1));
      device->registerImageCallback (&OpenniCam::imageCallback, *this, &(*device));
	cout << "ok init 2" << endl;

    }
    if (device->hasDepthStream())
    {
      if (!device->isDepthModeSupported (mode))
      {
        cout << "depth stream mode " << mode.nXRes << " x " << mode.nYRes << " @ " << mode.nFPS << " not supported" << endl;
        exit (-1);
      }
      //namedWindow (string (device->getConnectionString ()) + "Depth", WINDOW_AUTOSIZE);
      depth_images_[device->getConnectionString ()] = new ImageContext (Mat::zeros (480, 640, CV_32FC1));
      device->registerDepthCallback (&OpenniCam::depthCallback, *this, &(*device));
	cout << "ok init 3" << endl;
    }
  }

  timeval timestamp;
  gettimeofday (&timestamp, NULL);
  image_timestamp = depth_timestamp = timestamp.tv_sec + timestamp.tv_usec * 0.000001;

    //is_initialized = true;
	
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

        // run kinect thread
		
		run1();
		
        //pthread_create( &opennicam_thread, NULL, sofa::component::container::OpenniCam::_opennicam_threadfunc, this);

        //loadCamera();
    }


}

}

}


#endif ///*IMAGE_SOFTKINETIC_H*/
