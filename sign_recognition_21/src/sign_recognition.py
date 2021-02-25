# Code for sign recognition
"""
TODO: Stop sign distance recognition
"""

import rospy
import torch
import argparse
from torch.autograd import Variable
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import torchvision.transforms as transforms
import PIL

from utils import load_classes, non_max_suppression, pad_to_square, resize, Darknet
from sign_recognition_21.msg import Sign


class SignRecognition:
    def __init__(self, opt):
        # Options
        self.opt = opt

        # Initialize network
        self.device = 'cpu'
        self.classes = load_classes(self.opt.class_path)
        self.network = self.load_network()
        self.image = None
        self.depth_image = None

        # Initialize ROS things
        rospy.init_node('sign_detection', anonymous=True)
        self.pub = rospy.Publisher('sign', Sign, queue_size=1)
        self.sub_raw = rospy.Subscriber("/zed/zed_node/left_raw/image_raw_color", Image, self.set_img)
        self.sub_depth = rospy.Subscriber("/zed/zed_node/depth/depth_registered", Image, self.set_depth_img)
        self.bridge = CvBridge()
        print("setup complete")

    def __call__(self, rate=10):
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            msg = self.detect()
            self.pub.publish(msg)
            r.sleep()

    def set_img(self, msg):
        # read into OpenCV 2
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
    
    def set_depth_img(self, msg):
        # read into OpenCV 2
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, "32FC1")
        except CvBridgeError as e:
            print(e)

    def detect(self):
        msg = Sign()
        (
            msg.stop,
            msg.stop_distance,
            msg.speed_limit,
            msg.speed_limit_value,
        ) = self.process(self.image)
        return msg

    def process(self, image):
        if image is None:
            return(False, 0, False, 0)
        # read into PIL from OpenCV 2
        img = transforms.ToTensor()(PIL.Image.fromarray(image))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.opt.img_size)
        # Transform into tensor
        input_img = Variable(img.type(torch.FloatTensor))
        # detect things
        with torch.no_grad():
            detections = self.network(input_img.unsqueeze(0))
            detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)
        
        if detections is not None:
            labels = [:, -1]
        else:
            labels = []
        
        speed_limit = False
        speed_limit_value = 0
        stop_sign = False
        stop_sign_distance = 0

        for i, label in enumerate(labels):
            # avoid subscript errors
            if len(label) > 10:
                if label[:10] == "speedLimit":
                    speed_limit = True
                    # If unreadable
                    if label[-1] == "l":
                        # Stick with previous speed limit
                        speed_limit = False
                    else:
                        speed_limit_value = int(label[-2:])
            if label == "stop":
                stop_sign = True
                stop_sign_distance = self.calc_dist(detections[i], self.depth_image)

        return stop_sign, stop_sign_distance, speed_limit, speed_limit_value

    def load_network(self):
        model = Darknet(self.opt.model_def, img_size=self.opt.img_size).to(self.device)

        if self.opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(self.opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(self.opt.weights_path, map_location=self.device))

        model.eval()  # Set in evaluation mode
        return model

    def calc_dist(self, detection, depth_image):
        # find x and y coordinates of center of bounding box of detection
        x = int(detection[2] + detection[0] / 2)
        y = int(detection[3] + detection[1] / 2)
        # read into PIL from OpenCV 2
        img = transforms.ToTensor()(PIL.Image.fromarray(depth_image))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.opt.img_size)
        # take value from center for now
        dist = img[0,y,x].item()
        return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="yolov3-LISA.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="yolov3_LISA.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    options = parser.parse_args()
    sign = SignRecognition(options)
    sign()
