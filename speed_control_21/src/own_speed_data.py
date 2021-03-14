#!/usr/bin/env python3

import rospy
from std_msgs.msg import Float64


rospy.init_node('own_speed', anonymous=True)
pub = rospy.Publisher('speed', Float64, queue_size=1)

r = rospy.Rate(5)

while not rospy.is_shutdown():
    speed = 1
    pub.publish(speed)
    r.sleep()
