#!/usr/bin/env python3

import rospy
import time
from std_msgs.msg import Float64
from sign_recognition_21.msg import Sign

class SpeedController:
    def __init__(self, sys_timeout=2, sys_max_speed=10, safety_c1=3, safety_c2=0.1, safety_c3=0.01):
        self.safety_c1 = safety_c1
        self.safety_c2 = safety_c2
        self.safety_c3 = safety_c3
        self.sys_timeout = sys_timeout
        self.sys_max_speed = sys_max_speed

        self.object_vel = None
        self.object_dist = None
        self.own_vel = None
        self.stop = None
        self.stop_distance = None
        self.speed_limit = None
        self.speed_limit_value = None
        self.lane_offset_time = time.time()
        self.object_dist_time = time.time()
        self.object_vel_time = time.time()
        self.own_vel_time = time.time()
        self.sign_data_time = time.time()

        rospy.init_node('calc_target_speed', anonymous=True)

        rospy.Subscriber("object_speed", Float64, self.set_object_speed_data)
        rospy.Subscriber("object_distance", Float64, self.set_object_dist_data)
        rospy.Subscriber("speed", Float64, self.set_speed_data)
        rospy.Subscriber("sign", Sign, self.set_sign_data)
        rospy.Subscriber("lane_offset", Float64, self.set_lane_offset_time)

        self.pub = rospy.Publisher('target_speed', Float64, queue_size=1)

    def __call__(self, rate=10):
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            # set values so they dont change during calculation
            target_speed = self.calc_speed(
                self.own_vel,
                self.object_vel,
                self.object_dist,
                self.own_vel_time,
                self.object_vel_time,
                self.object_dist_time,
                self.lane_offset_time
            )
            print(target_speed)
            self.pub.publish(target_speed)
            r.sleep()

    def calc_speed(self, own_vel, object_vel, object_dist, own_vel_time, object_vel_time, object_dist_time, lane_offset_time):

        # If we don't have data yet, don't move
        if None in (own_vel, object_dist):
            print("I dont have all the input data yet")
            return 0
        if not object_vel:
            object_vel = 0

        curr_time = time.time()
        if curr_time - own_vel_time > self.sys_timeout:
            print("own velocity too old!")
            return 0
        if curr_time - object_dist_time > self.sys_timeout:
            print("object distance too old!")
            return 0
        if curr_time - lane_offset_time > self.sys_timeout:
            print("lane offset too old!")
            #return 0

        # as speed reading get more stale, phase them out
        staleness = curr_time - object_vel_time
        if staleness > 0.5:
            object_vel = 0.5 ** staleness

        safety_dist = self.safety_c1 + (self.safety_c2 * own_vel) + (self.safety_c3 * own_vel ** 2)

        base = ((object_dist - safety_dist) / safety_dist) * abs(object_vel - own_vel)


        if object_dist >= safety_dist:
            adjustment = object_dist * object_vel / safety_dist
        else:
            adjustment = ((object_dist / safety_dist) ** 2) * object_vel

        target_speed = base + adjustment

        # check if target speed is greater that system max or speed limit
        if target_speed > self.sys_max_speed:
            target_speed = self.sys_max_speed
            print("speed limited by system max speed")
        if self.speed_limit:
            if target_speed > self.speed_limit_value:
                print("speed limited by speed limit")
                target_speed = self.speed_limit_value
        if target_speed < 0:
            target_speed = 0
            print("target speed below 0")

        return target_speed

    def set_lane_offset_time(self, message):
        self.lane_offset_time = time.time()

    def set_object_dist_data(self, message):
        self.object_dist = message.data
        self.object_dist_time = time.time()

    def set_object_speed_data(self, message):
        self.object_vel = message.data
        self.object_vel_time = time.time()

    def set_speed_data(self, message):
        self.own_vel = message.data
        self.own_vel_time = time.time()

    def set_sign_data(self, message):
        self.stop = message.stop
        self.stop_distance = message.stop_distance
        self.speed_limit = message.speed_limit
        self.speed_limit_value = message.speed_limit_value
        self.sign_data_time = time.time()


if __name__ == "__main__":
    speed_controller = SpeedController(sys_max_speed=10)
    speed_controller()
