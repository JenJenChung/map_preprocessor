#include <ros/ros.h>
#include "MapWriter.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "map_writer") ;

  ros::NodeHandle nHandle ;
  
  MapWriter map_writer(nHandle) ;
  
  ros::spin();
  return 0;
}
