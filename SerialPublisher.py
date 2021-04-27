#!/usr/bin/env python3
import serial
import rospy
import io
from std_msgs.msg import Float64, Bool



class SerialPublisher:
    def __init__(self, p, b):
        self.target_speed = None
        self.current_speed = None
        self.target_steering_angle = None
        self.drive_mode = False

        s = serial.Serial(port=p, baudrate=b, timeout=0.25)
        rospy.init_node('serial_publisher', anonymous=True)
        self.speed_pub = rospy.Publisher('/speed', Float64, queue_size=1)
        rospy.Subscriber("/target_speed", Float64, self.set_target_speed)
        rospy.Subscriber("/target_steering_angle", Float64, self.set_target_steering_angle)
        rospy.Subscriber("/drive_mode", Bool, self.set_drive_mode)

    def __call__(self, rate):
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            steer, brake, throttle = self.calc_commands()
            command = f"{steer},{brake},{throttle}".encode()
            if self.drive_mode:
                serial.write(command)
            else:
                print(steer, brake, throttle)
            current_speed = serial.readline()
            if current_speed != '':
                self.current_speed = current_speed
                self.speed_pub.publish(current_speed)
            r.sleep()

    def calc_commands(self):
        # Unfinished - still needs to be integrated with drive-by-wire system.
        if None in [self.target_speed, self.current_speed, self.target_steering_angle]:
            return 0,0,0
        steer = self.target_steering_angle
        if self.target_speed < self.current_speed:
            brake = 1
            throttle = 0
        else:
            throttle = int(self.target_speed)
            brake = 0
        return steer, brake, throttle

    def set_drive_mode(self, msg):
        self.drive_mode = msg.data

    def set_target_speed(self, msg):
        self.target_speed = msg.data

    def set_target_steering_angle(self, msg):
        self.target_steering_angle = msg.data


if __name__ == "__main__":
    port = 10
    baud = 9600
    s = SerialPublisher(port, baud)
    s(4)
