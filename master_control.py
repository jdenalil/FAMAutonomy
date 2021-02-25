import rospy
from std_msgs import Bool


class Controller:
    def __init__(self):
        rospy.init_node('master_controller', anonymous=True)
        self.pub = rospy.Publisher('/drive_mode', Bool, queue_size=1)

    def __call__(self):
        mode = input()
        # change to shadow mode
        if mode == "0" or mode == "False":
            self.pub(False)
        # change to drive mode
        elif mode == "1" or mode == "True":
            self.pub(True)
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