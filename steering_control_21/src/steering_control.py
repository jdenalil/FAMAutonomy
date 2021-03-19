#!/usr/bin/env python3
import rospy
import math
from std_msgs.msg import Float64


class SteeringController:
    def __init__(self, lookahead_gain=0.25):
        self.lookahead_gain = lookahead_gain

        self.right_rad = None
        self.left_rad = None
        self.lane_offset = None
        self.current_speed = None
        self.last_angle = 0

        rospy.init_node('calc_steering_angle', anonymous=True)

        rospy.Subscriber("speed", Float64, self.set_current_speed)
        rospy.Subscriber("l_rad", Float64, self.set_left_rad)
        rospy.Subscriber("r_rad", Float64, self.set_right_rad)
        rospy.Subscriber("lane_offset", Float64, self.set_lane_offset)

        self.pub = rospy.Publisher('target_steering_angle', Float64, queue_size=1)

    def __call__(self, rate=10):
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            target_angle = self.calc_angle(
                self.right_rad,
                self.left_rad,
                self.lane_offset,
                self.current_speed
            )
            self.pub.publish(target_angle)
            r.sleep()

    def calc_angle(self, right_rad, left_rad, lane_offset, current_speed):
        # If we don't have all data yet, don't steer
        if None in [lane_offset, current_speed]:
            return 0


        # Extremly simple V0 version
        ld = self.lookahead_gain * self.current_speed
        alpha = math.asin(lane_offset/ld)
        target_angle = alpha
        '''
        wheelbase = 2.4

        # Angle to lookahead point - complex version using curvature
        avg_rad = (right_rad + left_rad) / 2
        if turning_left:
            r2 = avg_rad
            r1 = avg_rad - lane_offset
            direction = "left"
        elif turning_right:
            r2 = avg_rad
            r1 = avg_rad + lane_offset
            direction = "right"

        # Custom estimation of angle to lookahead point
        # Uses law of sines
        alpha = math.acos(((r2 ** 2) + (r1 ** 2) - (ld ** 2))/(2 * r1 * r2)) / 2

        # ultra-simple control using only lane offset:
        # alpha and target_angle will have the same sign as lane_offset
        # lane offset is positive if the car is too far right
        # and negative if the car is too far left
        alpha = math.asin(lane_offset/ld)
        # Pure Pursuit
        target_angle = math.atan(2 * wheelbase * math.sin(alpha) / ld)
        '''
        return target_angle

    def set_current_speed(self, message):
        self.current_speed = message

    def set_right_rad(self, message):
        self.right_rad = message

    def set_left_rad(self, message):
        self.left_rad = message

    def set_lane_offset(self, message):
        self.lane_offset = message


if __name__ == "__main__":
    steering_controller = SteeringController()
    steering_controller()
