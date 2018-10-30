#include "ros/ros.h"
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

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
    Matrix4f T;
    VectorXf xi;

    void xi2T(VectorXf& xi, Matrix4f& T);
    void T2xi(Matrix4f& T, VectorXf& xi);
    void getWarp(cv::Mat xImg, cv::Mat yImg, cv::Mat xyz_point, cv::Mat DRef, VectorXf xi);
    void downScaleImage(cv::Mat &src_img, cv::Mat &dst_img, Matrix3f &dst_K, int num);
    void deriveErrAnalytic(cv::Mat &JacI, cv::Mat &JacD, cv::Mat &resI, cv::Mat &resD,
                           cv::Mat IRef, cv::Mat DRef, cv::Mat I, cv::Mat D, Matrix3f K);
};

occacc::occacc(){
    rgb_cur = cv::Mat::zeros(cv::Size(640,480), CV_8UC3);
    depth_cur = cv::Mat::zeros(cv::Size(640,480), CV_32FC1);
    rgb_prev = cv::Mat::zeros(cv::Size(640,480), CV_8UC3);
    depth_prev= cv::Mat::zeros(cv::Size(640,480), CV_32FC1);
    xi.resize(6);
    xi << 0,0,0,0,0,0;
    K << 544.3, 0, 326.9,
         0, 546.1, 236.1,
         0, 0, 1;
    ROS_INFO_STREAM("\n" << K);
}

void occacc::DVO(){
    cv::Mat IRef, I, DRef, D, IRefGray, IGray, resI, resD, res,
            weighI, weighD, weigh, maskI, maskD, mask, JacI, JacD, Jac;
    cv::cvtColor(rgb_cur, IGray, CV_BGR2GRAY);
    cv::cvtColor(rgb_prev, IRefGray, CV_BGR2GRAY);
    Matrix3f Klvl;
    Matrix4f T_upd, T_temp;
    VectorXf xi_upd, xi_temp;
    double weighDeltaI,weighDeltaD,errLast,err,lambda;
    xi_temp =  xi;
    int lvlmax = 5;
    ROS_INFO_STREAM("initialize DVO \n");
    for(int lvl = lvlmax;lvl>0;lvl--){
        errLast = 1e10;
        lambda = 0.1;

        downScaleImage(IGray,I,Klvl,lvl);
        downScaleImage(IRefGray,IRef,Klvl,lvl);
        downScaleImage(depth_cur,D,Klvl,lvl);
        downScaleImage(depth_prev,DRef,Klvl,lvl);
        ROS_INFO_STREAM("Image pyramid " << lvl << "\n");
        for(int iter = 0; iter < 40; iter ++){

            deriveErrAnalytic(JacI,JacD,resI,resD,IRef,DRef,I,D,Klvl);

            weighDeltaI = (64-exp(lvl-1)*48)/255;
            weighDeltaD = 0.5;
            weighI = Mat::zeros(resI.size(), CV_32F);
            weighD = Mat::zeros(resD.size(), CV_32F);
            maskI = abs(resI) <= weighDeltaI;
            maskD = abs(resD) <= weighDeltaD;
            resI = ((1-(resI/weighDeltaI).mul(resI/weighDeltaI))
                    .mul(1-(resI/weighDeltaI).mul(resI/weighDeltaI)));
            resD = ((1-(resD/weighDeltaD).mul(resD/weighDeltaD))
                    .mul(1-(resD/weighDeltaD).mul(resD/weighDeltaD)));

            weighI.copyTo(resI, maskI);
            weighD.copyTo(resD, maskD);
            hconcat(weighI,weighD,weigh);
            hconcat(JacI,JacD,Jac);
            hconcat(resI,resD,res);
            cv::Mat H = (Jac.t()*repeat(weigh,1,6).mul(Jac));
            cv2eigen(-(H + lambda* (Mat::diag(H.diag(0))).inv())*Jac.t()*(weigh*res),xi_upd);

            xi_temp = xi;
            xi2T(xi_upd, T_upd);
            xi2T(xi, T_temp);
            T = T_upd*T_temp;
            T2xi(T,xi);
            mask = res!=0;
            weigh.copyTo((Mat)(weigh.mul(res)).mul(res),mask);
            err = mean(weigh).val[0];
            //TODO: Jacobian, residual, IR
            if (err >= errLast){
                lambda = lambda * 3;
                xi = xi_temp;

                if(lambda > 5) break;
            }
            else{
                lambda = lambda / 1.5;
            }
            errLast = err;

        }
    }
    ROS_INFO_STREAM("\n " << xi);
}
void occacc::deriveErrAnalytic(cv::Mat &JacI, cv::Mat &JacD, cv::Mat &resI, cv::Mat &resD,
                               cv::Mat IRef, cv::Mat DRef, cv::Mat I, cv::Mat D, Matrix3f K){
    int JacSize = I.rows*I.cols;
    cv::Mat xImg, yImg, xp, yp, zp, dxI, dyI, dxD, dyD, dxIK, dyIK, dxDK, dyDK, kernel_x, kernel_y;
    Vector3f t, p, pTrans, pTransProj;
    Matrix3f R;
    Matrix4f T;

    JacI = Mat::zeros(Size(JacSize,1),CV_32F);
    JacD = Mat::zeros(Size(JacSize,1),CV_32F);
    xImg = Mat::zeros(Size(I.cols,I.rows),CV_32F);
    yImg = Mat::zeros(Size(I.cols,I.rows),CV_32F);
    xp = Mat::zeros(Size(I.cols,I.rows),CV_32F);
    yp = Mat::zeros(Size(I.cols,I.rows),CV_32F);
    zp = Mat::zeros(Size(I.cols,I.rows),CV_32F);

    xi2T(xi,T);
    R = T.block<3,3>(0,0);
    t = T.block<3,1>(0,3);

    ROS_INFO_STREAM("deriveErrAnalytic \n" << xi << "\n");

    for (int x=0; x < I.cols; x++){
        for (int y=0; y < I.rows; y++){
            p << x, y, 1;
            p = p * DRef.at<double>(y,x);
            pTrans = R*K.inverse()*p + t;
            if((pTrans(2) > 0) && (DRef.at<double>(y,x) > 1e-4)){

                ROS_INFO_STREAM("p DRef" << DRef.at<double>(y,x) <<" \n" << p << "\n");
                ROS_INFO_STREAM("p Trans \n" << pTrans << "\n");

                pTransProj = K*pTrans;
                xImg.at<double>(y,x) = pTransProj(0) / pTransProj(2);
                yImg.at<double>(y,x) = pTransProj(1) / pTransProj(2);

                xp.at<double>(y,x) = pTrans(0);
                yp.at<double>(y,x) = pTrans(1);
                zp.at<double>(y,x) = pTrans(2);
            }
        }
    }

    ROS_INFO_STREAM("kernel set \n");
    kernel_x = Mat::zeros(3, 3, CV_32F);
    kernel_y = Mat::zeros(3, 3, CV_32F);
    kernel_x.at<double>(1,0) = -0.5;
    kernel_x.at<double>(1,2) = 0.5;
    kernel_y.at<double>(0,1) = -0.5;
    kernel_y.at<double>(2,1) = 0.5;

    ROS_INFO_STREAM("Image filter \n");
    cv::filter2D(I, dxI, -1, kernel_x, Point(-1,-1), 0, BORDER_CONSTANT);
    cv::filter2D(I, dyI, -1, kernel_y, Point(-1,-1), 0, BORDER_CONSTANT);
    cv::filter2D(D, dxD, -1, kernel_x, Point(-1,-1), 0, BORDER_CONSTANT);
    cv::filter2D(D, dyD, -1, kernel_y, Point(-1,-1), 0, BORDER_CONSTANT);

    ROS_INFO_STREAM("reshape image \n");
    dxI.reshape(JacSize,1);
    dyI.reshape(JacSize,1);
    dxD.reshape(JacSize,1);
    dyD.reshape(JacSize,1);

    ROS_INFO_STREAM("dxIK set \n");
    dxIK = K(1,1)*dxI;
    dyIK = K(2,2)*dyI;
    dxDK = K(1,1)*dxD;
    dyDK = K(2,2)*dyD;

    // reshape and divide problem.

    ROS_INFO_STREAM("\n ok" << dxIK << " \n" << zp << "\n" << dxI);
    divide(dxIK, zp, zp,1,-1); // / is right? or divide(a,b,c) would be better? divide 0 problem
    ROS_INFO_STREAM("\n ok12");
    hconcat(JacI, dyIK/zp, JacI);
    ROS_INFO_STREAM("\n ok2");
    hconcat(JacI, -(dxIK.mul(xp)+dyIK.mul(yp))/zp.mul(zp), JacI);
    hconcat(JacI, -(dxIK.mul(xp.mul(yp)))/zp.mul(zp)-dyIK.mul(1+(yp/zp).mul(yp/zp)), JacI);
    hconcat(JacI, dxIK.mul(1+(yp/zp).mul(yp/zp))+dyIK.mul(xp.mul(yp))/zp.mul(zp), JacI);
    hconcat(JacI, (-dxIK.mul(yp)+dyIK.mul(xp))/zp, JacI);

    JacD = dxDK/zp; // / is right? or divide(a,b,c) would be better?
    hconcat(JacD, dyDK/zp, JacD);
    hconcat(JacD, -(dxDK.mul(xp)+dyDK.mul(yp))/zp.mul(zp), JacD);
    hconcat(JacD, -(dxDK.mul(xp.mul(yp)))/zp.mul(zp)-dyDK.mul(1+(yp/zp).mul(yp/zp)), JacD);
    hconcat(JacD, dxDK.mul(1+(yp/zp).mul(yp/zp))+dyDK.mul(xp.mul(yp))/zp.mul(zp), JacD);
    hconcat(JacD, (-dxDK.mul(yp)+dyDK.mul(xp))/zp, JacD);
    //JacD.block<JacSize,1>(0,0) = dxDK.mul(zp) ;
    //JacD.block<JacSize,1>(0,1) = dyDK.mul(zp);
    //JacD.block<JacSize,1>(0,2) = -(dxDK.mul(xp)+dyDK.mul(yp))/zp.mul(zp);
    //JacD.block<JacSize,1>(0,3) = -(dxDK.mul(xp.mul(yp)))/zp.mul(zp)-dyDK.mul(1+(yp/zp).mul(yp/zp));
    //JacD.block<JacSize,1>(0,4) = dxDK.mul(1+(yp/zp).mul(yp/zp))+dyDK.mul(xp.mul(yp))/zp.mul(zp);
    //JacD.block<JacSize,1>(0,5) = (-dxDK.mul(yp)+dyDK.mul(xp))/zp;

    JacI = -JacI;
    JacD = -JacD;

    cv::Mat IR;
    cv::Mat DR;
    remap(I,IR,xImg,yImg,INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0));
    remap(D,DR,xImg,yImg,INTER_NEAREST,BORDER_CONSTANT,Scalar(0,0,0));
    resI = IRef-IR;
    resD = DRef-DR;
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
  showimages();
  DVO();
}

void occacc::xi2T(VectorXf &xi, Matrix4f &T){
    Matrix4f Temp;
    Temp << 0, -xi(5), xi(4), xi(0),
            xi(5), 0, -xi(3), xi(1),
            -xi(4), xi(3), 0, xi(2),
            0, 0, 0, 0;
    T = Temp.array().exp();

}
void occacc::T2xi(Matrix4f &T, VectorXf &xi){
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
