#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool


class Controller:
    def __init__(self):
        rospy.init_node('master_controller', anonymous=True)
        self.drive_mode_pub = rospy.Publisher('drive_mode', Bool, queue_size=1)

    def __call__(self):
        mode = input().lower()
        # change to shadow mode
        if mode == "0" or mode == "false" or mode == "off":
            self.drive_mode_pub.publish(False)
        # change to drive mode
        elif mode == "1" or mode == "true" or mode == "on":
            self.drive_mode_pub.publish(True)
        # quit the node
        elif mode == "Quit" or mode == "quit":
            return False
        else:
            print("Invalid mode", mode)
        return True


if __name__ == "__main__":
    c = Controller()
    q = True
    while q:
        q = c()
