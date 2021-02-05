# Code for sign recognition
"""
TODO: Stop sign distance recognition
"""

import rospy
import torch
import argparse
from torch.autograd import Variable
from cv_bridge import CvBridge, CvBridgeError
import torchvision.transforms as transforms
from PIL import Image

from Autonomy.sign_recognition.utils import load_classes, non_max_suppression, pad_to_square, resize
from Autonomy.sign_recognition.models import Darknet
from Autonomy.msg import Sign


class SignRecognition:
    def __init__(self, opt):
        # Options
        self.opt = opt

        # Initialize network
        self.device = 'cpu'
        self.classes = load_classes(self.opt.class_path)
        self.network = self.load_network()

        # Initialize ROS things
        self.pub = rospy.Publisher('sign', Sign, queue_size=1)
        self.sub = rospy.Subscriber("/zed/zed_node/right_raw/image_raw_color", Image, self.detect)
        self.bridge = CvBridge()

    def detect(self, incoming_msg):
        msg = Sign()
        (
            msg.stop,
            msg.stop_distance,
            msg.speed_limit,
            msg.speed_limit_value,
        ) = self.process(incoming_msg)

        self.pub.publish(msg)

    def process(self, msg):
        # read into OpenCV 2
        try:
          image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
          print(e)
        # read into PIL from OpenCV 2
        img = transforms.ToTensor()(Image.fromarray(image))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.opt.img_size)
        # Transform into tensor
        input_img = Variable(image.type(torch.FloatTensor))
        # detect things
        with torch.no_grad():
            detections = self.network(input_img)
            detections = non_max_suppression(detections, self.opt.conf_thres, self.opt.nms_thres)

        labels = detections[:, -1]

        speed_limit = False
        speed_limit_value = 0
        stop_sign = False
        stop_sign_distance = 0

        for label in labels:
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
                stop_sign_distance = self.calc_dist(detections)

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

    @staticmethod
    def calc_dist(detections):
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou threshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    options = parser.parse_args()
    sign = SignRecognition(options)

