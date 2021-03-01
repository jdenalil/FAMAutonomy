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
                print(data_rx_str)
                data = data_rx_str.split(',')
                if len(data) == 2:
                    units, value = data
                    value = float(value)
                    units = units.strip("\"")
                    if units == "m":
                        dist_pub.publish(value)
                    elif units == "mps":
                        speed_pub.publish(value)
                    else:
                        print("invalid unit:", units)
                else:
                    print('something went wrong...', data)

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
