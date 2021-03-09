"""
PID controllers and Send Serial Messages
"""
import serial
import io


class SerialPublisher:
    def __init__(self, p, b):
        self.target_speed = None
        self.current_speed = None
        self.target_steering_angle = None

        self.s = serial.Serial(port=p, baudrate=b, timeout=3)

        rospy.Subscriber("/target_speed", Float64, self.set_target_speed)
        rospy.Subscriber("/speed", Float64, self.set_current_speed)
        rospy.Subscriber("/target_steering_angle", Float64, self.set_target_steering_angle)

    def __call__(self, rate):
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            steer, brake, throttle = self.calc_commands()
            command = f"{steer},{brake},{throttle}".encode()
            serial.write(command)
            r.sleep()

    def calc_commands(self):
        steer = self.target_steering_angle
        # calc everything here --------------------------------------------------------
        return steer, brake, throttle

    def set_target_speed(self, msg):
        self.target_speed = msg

    def set_current_speed(self, msg):
        self.current_speed = msg

    def set_target_steering_angle(self, msg):
        self.target_steering_angle = msg


if __name__ == "__main__":
    port = 10
    baud = 9600
    s = SerialPublisher(port, baud)
    s(10)
