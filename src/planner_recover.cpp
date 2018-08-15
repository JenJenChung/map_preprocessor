#include <ros/ros.h>
#include "planner_recover.h"

int main(int argc, char **argv)
{
  ros::init(argc, argv, "planner_recover") ;

  ros::NodeHandle nHandle ;
  
  MoveBaseRecover recover(nHandle) ;
  
  ros::spin();
  return 0;
}
