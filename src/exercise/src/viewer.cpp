#include "ros/ros.h"
#include <iostream>
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <math.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> TimeSyncPolicy;

using namespace std;
using namespace cv;
using namespace Eigen;
class occacc{
public:
    occacc();
    void getResidual(cv::Mat res, cv::Mat IR, cv::Mat IRef, cv::Mat DRef, cv::Mat I, VectorXf xi);
    void DVO();
    void rgbdCallbackOcc(const sensor_msgs::ImageConstPtr& rgbMsg, const sensor_msgs::ImageConstPtr& depthMsg);
    void showimages();
    void showtheimage(cv::Mat img);
    void init();
private:
    cv::Mat rgb_cur;
    cv::Mat depth_cur;
    cv::Mat rgb_prev;
    cv::Mat depth_prev;

    cv::Mat img;

    message_filters::Synchronizer<TimeSyncPolicy> *message_sync;
    message_filters::Subscriber<sensor_msgs::Image> *rgb_sub;
    message_filters::Subscriber<sensor_msgs::Image> *depth_sub;
    Matrix3f K;
    VectorXf xi;

    void xi2T(Matrix4f& T, VectorXf& xi);
    void T2xi(VectorXf& xi, Matrix4f& T);
    void getWarp(cv::Mat xImg, cv::Mat yImg, cv::Mat xyz_point, cv::Mat DRef, VectorXf xi);
    void downScaleImage(cv::Mat &src_img, cv::Mat &dst_img, Matrix3f &dst_K, int num);
    void deriveErrAnalytic(MatrixXf &JacI, MatrixXf &JacD, cv::Mat &resI, cv::Mat &resD,
                           cv::Mat IRef, cv::Mat DRef, cv::Mat I, cv::Mat D, Matrix3f K);
};

occacc::occacc(){
    rgb_cur = cv::Mat::zeros(cv::Size(640,480), CV_8UC3);
    depth_cur = cv::Mat::zeros(cv::Size(640,480), CV_32FC1);
    rgb_prev = cv::Mat::zeros(cv::Size(640,480), CV_8UC3);
    depth_prev= cv::Mat::zeros(cv::Size(640,480), CV_32FC1);
    xi.resize(6);
    K << 544.3, 0, 326.9,
         0, 546.1, 236.1,
         0, 0, 1;
    ROS_INFO_STREAM("\n" << K);
}
void occacc::DVO(){
    cv::Mat IRef, I, DRef, D, IRefGray, IGray, resI, resD;
    cv::cvtColor(rgb_cur, IGray, CV_BGR2GRAY);
    cv::cvtColor(rgb_prev, IRefGray, CV_BGR2GRAY);
    MatrixXf JacI, JacD;
    Matrix3f Klvl;
    int lvlmax = 5;
    for(int lvl = lvlmax;lvl>0;lvl--){
        double errLast = 1e10;
        downScaleImage(IGray,I,Klvl,lvl);
        downScaleImage(IRefGray,IRef,Klvl,lvl);
        downScaleImage(depth_cur,D,Klvl,lvl);
        downScaleImage(depth_prev,DRef,Klvl,lvl);

        //TODO: Jacobian, residual, IR


    }
}
void occacc::deriveErrAnalytic(MatrixXf &JacI, MatrixXf &JacD, cv::Mat &resI, cv::Mat &resD,
                               cv::Mat IRef, cv::Mat DRef, cv::Mat I, cv::Mat D, Matrix3f K){
    int JacSize = I.rows*I.cols;
    cv::Mat xImg, yImg, xp, yp, zp, dxI, dyI, dxD, dyD, dxIK, dyIK, dxDK, dyDK, kernel_x, kernel_y;
    Vector3f t, p, pTrans, pTransProj;
    Matrix3f R;
    Matrix4f T;

    JacI.resize(JacSize, 6);
    JacD.resize(JacSize, 6);

    xi2T(T,xi);
    R = T.block<3,3>(0,0);
    t = T.block<3,1>(0,3);


    for (int x=0; x < I.cols; x ++){
        for (int y=0; y < I.rows; y++){
            p << x, y, 1;
            p = p * DRef.at<double>(y,x);
            pTrans = R*K.inverse()*p + t;

            if(pTrans(3) > 0 && DRef.at<double>(y,x) > 0){
                pTransProj = K*pTrans;
                xImg.at<double>(y,x) = pTransProj(1) / pTransProj(3);
                yImg.at<double>(y,x) = pTransProj(2) / pTransProj(3);

                xp.at<double>(y,x) = pTrans(1);
                yp.at<double>(y,x) = pTrans(2);
                zp.at<double>(y,x) = pTrans(3);
            }
        }
    }

    kernel_x = Mat::zeros(3, 3, CV_32F);
    kernel_y = Mat::zeros(3, 3, CV_32F);
    kernel_x.at<double>(1,0) = -0.5;
    kernel_x.at<double>(1,2) = 0.5;
    kernel_y.at<double>(0,1) = -0.5;
    kernel_y.at<double>(2,1) = 0.5;

    cv::filter2D(I, dxI, -1, kernel_x, Point(-1,-1), 0, BORDER_CONSTANT);
    cv::filter2D(I, dyI, -1, kernel_y, Point(-1,-1), 0, BORDER_CONSTANT);
    cv::filter2D(D, dxD, -1, kernel_x, Point(-1,-1), 0, BORDER_CONSTANT);
    cv::filter2D(D, dyD, -1, kernel_y, Point(-1,-1), 0, BORDER_CONSTANT);

    dxI.reshape(JacSize,1);
    dyI.reshape(JacSize,1);
    dxD.reshape(JacSize,1);
    dyD.reshape(JacSize,1);

    dxIK = K(1,1)*dxI;
    dyIK = K(2,2)*dyI;
    dxDK = K(1,1)*dxD;
    dyDK = K(2,2)*dyD;

    JacI<JacSize,1>(0,0) = dxIK.mul(zp);
    JacI<JacSize,1>(0,1) = dyIK.mul(zp);
    JacI<JacSize,1>(0,2) = -(dxIK.mul(xp)+dyIK.mul(yp))/zp.mul(zp);
    JacI<JacSize,1>(0,3) = -(dxIK.mul(xp.mul(yp)))/zp.mul(zp)-dyIK.mul(1+(yp/zp).mul(yp/zp));
    JacI<JacSize,1>(0,4) = dxIK.mul(1+1+(yp/zp).mul(yp/zp))+dyIK.mul(xp.mul(yp))/zp.mul(zp);
    JacI<JacSize,1>(0,5) = (-dxIK.mul(yp)+dyIK.mul(xp))/zp;

    JacD<JacSize,1>(0,0) = dxDK.mul(zp) ;
    JacD<JacSize,1>(0,1) = dyDK.mul(zp);
    JacD<JacSize,1>(0,2) = -(dxDK.mul(xp)+dyDK.mul(yp))/zp.mul(zp);
    JacD<JacSize,1>(0,3) = -(dxDK.mul(xp.mul(yp)))/zp.mul(zp)-dyDK.mul(1+(yp/zp).mul(yp/zp));
    JacD<JacSize,1>(0,4) = dxDK.mul(1+1+(yp/zp).mul(yp/zp))+dyDK.mul(xp.mul(yp))/zp.mul(zp);
    JacD<JacSize,1>(0,5) = (-dxDK.mul(yp)+dyDK.mul(xp))/zp;

    JacI = -JacI;
    JacD = -JacD;

    cv::Mat IR;
    cv::Mat DR;
    remap(I,IR,xImg,yImg,INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0));
    remap(D,DR,xImg,yImg,INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0));
    resI = IRef-IR;
    resD = Dref-DR;
    resI.reshape(JacSize,1);
    resD.reshape(JacSize,1);
}

void occacc::downScaleImage(cv::Mat &src_img, cv::Mat &dst_img, Matrix3f &dst_K, int num){
    double scale = pow(0.5,num-1);
    dst_K << K(0,0)*scale, 0, (K(0,2)+0.5)*scale-0.5,
            0, K(1,1)*scale, (K(1,2)+0.5)*scale-0.5,
            0, 0, 1;
    resize(src_img,dst_img,dst_img.size(),scale,scale,INTER_NEAREST);
    //imshow("In the down",dst_img);
    //waitKey(1000);
}

void occacc::init(){
    string rgb_topic_name = string("/camera/rgb/image_color");
    string depth_topic_name = string("/camera/depth_registered/image");
    int queue_size = 1;
    ros::NodeHandle nh;

    rgb_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, rgb_topic_name,queue_size);
    depth_sub = new message_filters::Subscriber<sensor_msgs::Image>(nh, depth_topic_name,queue_size);

    message_sync = new message_filters::Synchronizer<TimeSyncPolicy>(TimeSyncPolicy(3), *rgb_sub, *depth_sub);
    message_sync->registerCallback(boost::bind(&occacc::rgbdCallbackOcc, this,_1,_2));
}

void occacc::showimages(){
    imshow("depth1",this->depth_cur);
    //imshow("depth2",this->depth_prev);
    imshow("image1",this->rgb_cur);
    //imshow("image2",this->rgb_prev);
    waitKey(30);
}
void occacc::showtheimage(cv::Mat image){
    imshow("image", image);
    waitKey(30);
}

void occacc::rgbdCallbackOcc(const sensor_msgs::ImageConstPtr& rgbMsg, const sensor_msgs::ImageConstPtr& depthMsg)
{
  rgb_cur.copyTo(rgb_prev);
  depth_cur.copyTo(depth_prev);
  cv::Mat rgb = cv_bridge::toCvShare(rgbMsg, "bgr8")->image;
  cv::Mat depth = cv_bridge::toCvShare(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
  rgb.copyTo(rgb_cur);
  depth.copyTo(depth_cur);

  //downScaleImage(rgb_cur,img,K_proc, 2);
  //ROS_INFO_STREAM("\n" << K_proc);
  //showimages();
}

void occacc::xi2T(Matrix4f &T, VectorXf &xi){
    Matrix4f Temp;
    Temp << 0, -xi(6), xi(5), xi(1),
            xi(6), 0, -xi(4), xi(2),
            -xi(5), xi(4), 0, xi(3),
            0, 0, 0, 0;
    T = Temp.array().exp();

}
void occacc::T2xi(VectorXf &xi, Matrix4f &T){
    Matrix4f Temp;
    Temp = T.array().log();
    xi << Temp(1,4), Temp(2,4), Temp(3,4), Temp(3,2), Temp(1,3), Temp(2,1);
}

void imageCallback(const sensor_msgs::ImageConstPtr& rgbMsg)
{
  cv::Mat rgb = cv_bridge::toCvShare(rgbMsg, "bgr8")->image;
  imshow( "rgb", rgb );
  waitKey(30);
}

void depthCallback(const sensor_msgs::ImageConstPtr& depthMsg)
{
  cv::Mat depth = cv_bridge::toCvShare(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1)->image;
  imshow( "depth", depth );
  waitKey(30);
}

void rgbdCallback(const sensor_msgs::ImageConstPtr& rgbMsg, const sensor_msgs::ImageConstPtr& depthMsg)
{
  cv::Mat rgb = cv_bridge::toCvShare(rgbMsg, "bgr8")->image;
  cv::Mat depth = cv_bridge::toCvShare(depthMsg, sensor_msgs::image_encodings::TYPE_32FC1)->image;

  imshow( "rgb", rgb);
  imshow( "depth", depth );
  waitKey(30);
}


int main(int argc, char **argv)
{
  //initialize node
  ros::init(argc, argv, "cv_example");

  // node handler
  //ros::NodeHandle nh;
  occacc *OCC = new occacc();
  OCC->init();
  /*string rgb_topic_name = string("/camera/rgb/image_color");
  string depth_topic_name = string("/camera/depth_registered/image");
  
  // subsribe topic
  message_filters::Subscriber<sensor_msgs::Image> rgbImgSubs(nh, rgb_topic_name,1);
  message_filters::Subscriber<sensor_msgs::Image> depthImgSubs(nh, depth_topic_name,1);

  message_filters::Synchronizer<TimeSyncPolicy> sync(TimeSyncPolicy(3), rgbImgSubs, depthImgSubs);
  sync.registerCallback(boost::bind(&occacc::rgbdCallbackOcc,_1,_2),OCC);
*/
  //ros::Subscriber sub = nh.subscribe("/camera/rgb/image_color", 1000, imageCallback);
  //ros::Subscriber sub_depth = nh.subscribe("/camera/depth_registered/image", 1000, depthCallback);

  ros::spin();

  return 0;
}
