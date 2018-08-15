#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <ros/ros.h>
#include <ros/console.h>
#include "nav_msgs/Odometry.h"
#include "nav_msgs/OccupancyGrid.h"
#include "nav_msgs/MapMetaData.h"
#include "geometry_msgs/Pose.h"
#include "std_msgs/Header.h"
#include "map_preprocessor/LocalMap.h"

using std::vector ;

typedef unsigned int UINT ;

class LocalMapState{
  public:
    LocalMapState(ros::NodeHandle) ;
    ~LocalMapState(){}
  private:
    ros::Subscriber subOdom ;
    ros::Subscriber subMap ;
    ros::Publisher pubLocalMap ;
    
    bool fOdom ;
    bool fMap ;
    
    geometry_msgs::Pose pose ;
    
    double resolution ;
    UINT width ;
    UINT height ;
    geometry_msgs::Pose origin ;
    nav_msgs::OccupancyGrid mapData ;
    nav_msgs::MapMetaData metaData ;
    std_msgs::Header header ;
    int frameSize ;
    
    void odomCallback(const nav_msgs::Odometry&) ;
    void mapCallback(const nav_msgs::OccupancyGrid&) ;
    
    void publishLocalMap() ;
};

LocalMapState::LocalMapState(ros::NodeHandle nh){
  std::string map_topic ;
  std::string odom_topic ;
  ros::param::param<std::string>("map_topic", map_topic, "map") ;
  subMap = nh.subscribe(map_topic, 10, &LocalMapState::mapCallback, this) ;
  ros::param::param<std::string>("odom_topic", odom_topic, "odom") ;
  
  ROS_INFO_STREAM("Subscribed to map: " << map_topic << ", and odom: " << odom_topic) ;
  
  subOdom = nh.subscribe(odom_topic, 10, &LocalMapState::odomCallback, this) ;
  pubLocalMap = nh.advertise<map_preprocessor::LocalMap>("local_map", 10, true) ;
  fMap = false ;
  fOdom = false ;
  
  ros::param::get("frame_size", frameSize) ;
}

void LocalMapState::odomCallback(const nav_msgs::Odometry& msg){
  fOdom = true ;
  pose.position.x = msg.pose.pose.position.x ;
  pose.position.y = msg.pose.pose.position.y ;
  pose.position.z = msg.pose.pose.position.z ;
  pose.orientation.x = msg.pose.pose.orientation.x ;
  pose.orientation.y = msg.pose.pose.orientation.y ;
  pose.orientation.z = msg.pose.pose.orientation.z ;
  pose.orientation.w = msg.pose.pose.orientation.w ;
}

void LocalMapState::mapCallback(const nav_msgs::OccupancyGrid& msg){
  if (fMap == false){
    // Initialise map meta data
    resolution = msg.info.resolution ;
    width = msg.info.width ;
    height = msg.info.height ;
    
    ROS_INFO_STREAM("Map meta data: resolution: " << resolution << ", width: " << width << ", height: " << height) ;
    
    origin.position.x = msg.info.origin.position.x ;
    origin.position.y = msg.info.origin.position.y ;
    origin.position.z = msg.info.origin.position.z ;
    origin.orientation.x = msg.info.origin.orientation.x ;
    origin.orientation.y = msg.info.origin.orientation.y ;
    origin.orientation.z = msg.info.origin.orientation.z ;
    origin.orientation.w = msg.info.origin.orientation.w ;
    
    fMap = true ;
  }
 
  header = msg.header ;
  metaData = msg.info ;
  mapData.data = msg.data ;
  
  if (fOdom)
    publishLocalMap() ;
}

void LocalMapState::publishLocalMap(){
  // compute robot's current cell (i,j)
  int i = (int) ((pose.position.y - origin.position.y)/resolution) ;
  int j = (int) ((pose.position.x - origin.position.x)/resolution) ;
  size_t c = i*width + j ;
//  ROS_INFO_STREAM("(x,y): (" << pose.position.x << "," << pose.position.y << ")") ;
//  ROS_INFO_STREAM("(i,j): (" << i << "," << j << ")") ;
//  ROS_INFO_STREAM("mapData.data[" << c << "]: " << (int)mapData.data[c]) ;
  
  // compute bottom left cell (min_i,min_j) and top right cell (max_i, max_j) of frame
  int min_i = i - frameSize/2 ;
  int min_j = j - frameSize/2 ;
  int max_i = min_i + frameSize ;
  int max_j = min_j + frameSize ;
  
  // Stay within map bounds
  if (min_i < 0)
    min_i = 0 ;
  
  if (min_j < 0)
    min_j = 0 ;
  
  if (max_i > height)
    min_i = height - frameSize ;
  
  if (max_j > width)
    min_j = width - frameSize ;
  
  // LocalMap object to store and publish data
  map_preprocessor::LocalMap local_map ;
  local_map.header = header;
  
  for (int ii = min_i; ii < max_i; ii++){
    for (int jj = min_j; jj < max_j; jj++){
      int cc = ii*width + jj ;
      local_map.data.push_back(mapData.data[cc]) ;
    }
  }
  
  local_map.info = metaData ;
  local_map.info.width = frameSize ;
  local_map.info.height = frameSize ;
  local_map.info.origin.position.x = (double)min_j*resolution + origin.position.x ;
  local_map.info.origin.position.y = (double)min_i*resolution + origin.position.y ;
  local_map.info.origin.orientation.w = 1.0 ;
  
  pubLocalMap.publish(local_map) ;
}


