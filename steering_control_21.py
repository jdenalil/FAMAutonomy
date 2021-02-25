import rospy
from std_msg import Float64


class SteeringController:
    def __init__(self, constants):
        self.constants = constants

        self.right_rad = None
        self.left_rad = None
        self.lane_offset = None

        rospy.init_node('calc_steering_angle', anonymous=True)

        rospy.Subscriber("/l_rad", Float64, self.set_left_rad)
        rospy.Subscriber("/r_rad", Float64, self.set_right_rad)
        rospy.Subscriber("/lane_offset", Float64, self.set_lane_offset)

        self.pub = rospy.Publisher('/target_steering_angle', Float64, queue_size=1)

    def __call__(self, rate=10):
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            target_angle = self.calc_angle(
                self.right_rad,
                self.left_rad,
                self.lane_offset
            )
            self.pub.publish(target_angle)
            r.sleep()

    def calc_angle(self, right_rad, left_rad, lane_offset):
        # If we don't have all data yet, don't steer
        if None in [right_rad, left_rad, lane_offset]:
            return 0
        # ------------------------------------------------------
        # do calculation here
        target_angle = 0
        # ------------------------------------------------------
        return target_angle

    def set_right_rad(self, message):
        self.right_rad = message

    def set_left_rad(self, message):
        self.left_rad = message

    def set_lane_offset(self, message):
        self.lane_offset = message


if __name__ == "__main__":
    steering_controller = SteeringController(None)
    steering_controller()
