#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <ros/ros.h>
#include <ros/console.h>
#include "nav_msgs/Odometry.h"
#include "nav_msgs/OccupancyGrid.h"
#include "geometry_msgs/Pose.h"

using std::vector ;
using std::string ;

typedef unsigned int UINT ;

class MapWriter{
  public:
    MapWriter(ros::NodeHandle) ;
    ~MapWriter(){}
  private:
    ros::Subscriber subOdom ;
    ros::Subscriber subMap ;
    
    bool fOdom ;
    bool fMap ;
    
    geometry_msgs::Pose pose ;
    
    double resolution ;
    UINT width ;
    UINT height ;
    geometry_msgs::Pose origin ;
    nav_msgs::OccupancyGrid mapData ;
    UINT seq ;
    char fileName[100] ;
    int frameSize ;
    
    void odomCallback(const nav_msgs::Odometry&) ;
    void mapCallback(const nav_msgs::OccupancyGrid&) ;
    
    void writeToFile() ;
};

MapWriter::MapWriter(ros::NodeHandle nh){
  subMap = nh.subscribe("map", 10, &MapWriter::mapCallback, this) ;
  subOdom = nh.subscribe("odom", 10, &MapWriter::odomCallback, this) ;
  fMap = false ;
  fOdom = false ;
  
  string fName ;
  ros::param::get("file_name", fName) ;
  std::strcpy(fileName, fName.c_str()) ;
  ROS_INFO_STREAM("Filename: " << fileName << "_X.txt") ;
  ros::param::get("frame_size", frameSize) ;
}

void MapWriter::odomCallback(const nav_msgs::Odometry& msg){
  fOdom = true ;
  pose.position.x = msg.pose.pose.position.x ;
  pose.position.y = msg.pose.pose.position.y ;
  pose.position.z = msg.pose.pose.position.z ;
  pose.orientation.x = msg.pose.pose.orientation.x ;
  pose.orientation.y = msg.pose.pose.orientation.y ;
  pose.orientation.z = msg.pose.pose.orientation.z ;
  pose.orientation.w = msg.pose.pose.orientation.w ;
}

void MapWriter::mapCallback(const nav_msgs::OccupancyGrid& msg){
  if (fMap == false){
    // Initialise map meta data
    resolution = msg.info.resolution ;
    width = msg.info.width ;
    height = msg.info.height ;
    
    origin.position.x = msg.info.origin.position.x ;
    origin.position.y = msg.info.origin.position.y ;
    origin.position.z = msg.info.origin.position.z ;
    origin.orientation.x = msg.info.origin.orientation.x ;
    origin.orientation.y = msg.info.origin.orientation.y ;
    origin.orientation.z = msg.info.origin.orientation.z ;
    origin.orientation.w = msg.info.origin.orientation.w ;
    
    fMap = true ;
  }
 
  seq = msg.header.seq ;
  mapData.data = msg.data ;
  
  if (fOdom)
    writeToFile() ;
}

void MapWriter::writeToFile(){
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
  
  ROS_INFO_STREAM("Writing map #" << seq << "...") ;
  
  // open file for writing
  int buffSize = 100 ;
  char oFile[buffSize] ;
  sprintf(oFile,"%s_%d.txt",fileName,seq) ;
  std::ofstream outputFile ;
  if (outputFile.is_open()){
    outputFile.close() ;
  }
  outputFile.open(oFile, std::ios::app) ;
  
  for (int ii = min_i; ii < max_i; ii++){
    for (int jj = min_j; jj < max_j; jj++){
      int cc = ii*width + jj ;
      outputFile << (int)mapData.data[cc] << "," ;
    }
    outputFile << "\n" ;
  }
  outputFile.close() ;
  ROS_INFO("Complete!") ;
}


