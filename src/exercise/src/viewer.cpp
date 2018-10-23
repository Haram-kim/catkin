#include "ros/ros.h"
#include <iostream>
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/Image.h"
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> TimeSyncPolicy;

using namespace std;
using namespace cv;

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
  ros::NodeHandle nh;

  string rgb_topic_name = string("/camera/rgb/image_color");
  string depth_topic_name = string("/camera/depth_registered/image");
  
  // subsribe topic
  message_filters::Subscriber<sensor_msgs::Image> rgbImgSubs(nh, rgb_topic_name,1);
  message_filters::Subscriber<sensor_msgs::Image> depthImgSubs(nh, depth_topic_name,1);

  message_filters::Synchronizer<TimeSyncPolicy> sync(TimeSyncPolicy(3), rgbImgSubs, depthImgSubs);
  sync.registerCallback(boost::bind(&rgbdCallback,_1,_2));

  //ros::Subscriber sub = nh.subscribe("/camera/rgb/image_color", 1000, imageCallback);
  //ros::Subscriber sub_depth = nh.subscribe("/camera/depth_registered/image", 1000, depthCallback);

  ros::spin();

  return 0;
}
