#include <ros/ros.h>
#include "LocalMapState.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "local_map_state") ;

  ros::NodeHandle nHandle ;
  
  LocalMapState local_map_state(nHandle) ;
  
  ros::spin();
  return 0;
}
