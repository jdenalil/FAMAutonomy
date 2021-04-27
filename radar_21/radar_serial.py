#!/usr/bin/env python3
from optparse import OptionParser
from time import time

from std_msgs.msg import Float64
import rospy

import serial
import serial.tools.list_ports
from serial.serialutil import SerialException

import sys
import select
import tty
import termios

THRESH = 250
curr_det_mag = 0


def main(speed_pub, dist_pub):
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-p", "--port", dest="port_name",
                      help="read data from PORTNAME")
    parser.add_option("-b", "--baud", dest="baudrate",
                      default="57600",
                      help="baud rate on serial port")
    parser.add_option("-t", "--timeToLive",
                      default=0,
                      dest="time_to_live")
    parser.add_option("-i", "--interval_time",
                      default=0,
                      dest="interval_time")
    parser.add_option("-s", "--interval_send",
                      default="",
                      dest="interval_send")
    (options, args) = parser.parse_args()

    baudrate_int = int(options.baudrate)
    if baudrate_int <= 0:
        baudrate_int = 57600
    serial_port = serial.Serial(
        timeout=0.1,
        writeTimeout=0.2,
        baudrate=baudrate_int
    )
    serial_port.port = "/dev/ttyACM0"
    serial_port.open()

    if not serial_port.is_open:
        print("Exiting.  Could not open serial port:", serial_port.port)
        sys.exit(1)

    # suppress echo on terminal.
    old_tty_settings = termios.tcgetattr(sys.stdin)
    old_port_attr = termios.tcgetattr(serial_port.fileno())
    new_port_attr = termios.tcgetattr(serial_port.fileno())
    new_port_attr[3] = new_port_attr[3] & ~termios.ECHO
    termios.tcdrain(serial_port.fileno())
    termios.tcsetattr(serial_port.fileno(), termios.TCSADRAIN, new_port_attr)
    try:
        tty.setcbreak(sys.stdin.fileno())
        serial_port.flushInput()
        serial_port.flushOutput()
        while serial_port.is_open:
            data_rx_bytes = serial_port.readline()
            data_rx_length = len(data_rx_bytes)
            if data_rx_length != 0:
                data_rx_str = str.rstrip(str(data_rx_bytes.decode('utf-8', 'strict')))
                data = data_rx_str.split(',')
                units = data[0]
                vals = data[1:]
                units = units.strip("\"")
                curr_det_mag = 0
                try:
                    if units == "mps":
                        # if the most recent distance detection was reputable
                        if curr_det_mag > THRESH:
                            # when false readings happen, they are always over 10 mps
                            # real readings over 10 mps are rare
                            if abs(float(vals[0])) < 10:
                                print("speed reading: ", vals[0])
                                speed_pub.publish(float(vals[0]))
                            else:
                                print("invalidated speed (too high value): ", vals[0])
                        else:
                            print("invalidated speed (magnitude): ", vals[0])
                    elif units == "m":
                        for i, val in enumerate(vals):
                            vals[i] = float(val)
                        curr_det_mag = vals[0]
                        # if detected distance is under 1 meter (not an actual object)
                        if vals[1] < 1:
                            # then look at second magnitude for real object
                            if vals[2] > THRESH:
                                print("second object: ", vals[3], "magnitude: ", vals[2])
                                dist_pub.publish(vals[3])
                            else:
                                reason = 1
                        elif vals[0] > THRESH:
                            print("first object: ", vals[1], "magnitude: ", vals[0])
                            dist_pub.publish(vals[1])
                        else:
                            print("no valid object detected")
                    else:
                        print("invalid unit:", units)

                except Exception as e:
                    print('something went wrong...', data, e)


    except SerialException:
        print("Serial Port closed. terminate.")
    except KeyboardInterrupt:
        print("Break received,  Serial Port closing.")
    finally:
        termios.tcsetattr(serial_port.fileno(), termios.TCSADRAIN, old_port_attr)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_settings)


if __name__ == "__main__":
    rospy.init_node('radar', anonymous=True)
    speed_pub = rospy.Publisher('object_speed', Float64, queue_size=5)
    dist_pub = rospy.Publisher('object_distance', Float64, queue_size=5)
    main(speed_pub, dist_pub)
