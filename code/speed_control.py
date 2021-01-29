import rospy
from std_msgs.msg import Float64


class SpeedController:
    def __init__(self, sys_max_speed=10, safety_c1=3, safety_c2=0.1, safety_c3=0.01):
        self.safety_c1 = safety_c1
        self.safety_c2 = safety_c2
        self.safety_c3 = safety_c3
        self.sys_max_speed = sys_max_speed

        self.object_vel = None
        self.own_vel = None
        self.stop = None
        self.stop_distance = None
        self.speed_limit = None
        self.speed_limit_value = None

        rospy.init_node('calc_target_speed', anonymous=True)

        rospy.Subscriber("radar", Radar, self.set_object_data)
        rospy.Subscriber("speed", Float64, self.set_speed_data)
        rospy.Subscriber("sign", Sign, self.set_sign_data)

        self.pub = rospy.Publisher('target_speed', Float64, queue_size=1)

    def control(self, rate=10):
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            target_speed = self.calc_speed(
                self.own_vel,
                self.object_vel,
                self.object_dist
            )
            self.pub.publish(target_speed)
            r.sleep()

    def calc_speed(self, own_vel, object_vel, object_dist):

        # If we don't have data yet, don't move
        if not own_vel or not object_vel or not object_dist:
            return 0

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
        if self.speed_limit:
            if target_speed > self.speed_limit_value:
                target_speed = self.speed_limit_value

        return target_speed

    def set_object_data(self, message):
        self.object_vel = message.vel
        self.object_dist = message.dist

    def set_speed_data(self, message):
        self.own_vel = message

    def set_sign_data(self, message):
        self.stop = message.stop
        self.stop_distance = message.stop_distance
        self.speed_limit = message.speed_limit
        self.speed_limit_value = message.speed_limit_value

if __name__ == "__main__":
    speed_controller = SpeedController(sys_max_speed=10)
    speed_controller.control()