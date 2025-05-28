#!/bin/bash

# Parameters
TCP_IP="192.168.127.2"    # Change this to the correct IP address
TCP_PORT=4007             # Change this to the correct TCP port
TTY_DEVICE=/dev/ttyUSB0   # Change this to the desired ttyUSB device name

# Create the virtual serial device
sudo socat PTY,link=$TTY_DEVICE,raw TCP:$TCP_IP:$TCP_PORT
