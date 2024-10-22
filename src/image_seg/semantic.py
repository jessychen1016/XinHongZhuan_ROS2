import zmq
import cv2
import base64
import numpy as np
from ultralytics import YOLO

# Set up the ZeroMQ context and subscriber socket
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")  # Connect to the publisher
socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages\

zmq_send_context = zmq.Context()
send_socket = zmq_send_context.socket(zmq.PUB)
send_socket.bind("tcp://*:5556")  # Binding to port 5555 for sending images


model = YOLO("./yolo11x-seg.pt") 
init = True


while True:
    # Receive the base64 encoded image string
    img_as_text = socket.recv_string()
    if  init == True:
        cv2.waitKey(10)
        init = False
    # Decode the base64 string back to binary
    img_data = base64.b64decode(img_as_text)

    # Convert the binary data to a NumPy array and decode the JPEG
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    results = model(img)

    segmented_image = np.zeros_like(img)

    # Define a color map for different classes
    color_map = {
        0: (255, 0, 0),       # Class 0 - Red
        1: (0, 255, 0),       # Class 1 - Green
        2: (0, 0, 255),       # Class 2 - Blue
        3: (255, 255, 0),     # Class 3 - Cyan
        4: (255, 0, 255),     # Class 4 - Magenta
        5: (0, 255, 255),     # Class 5 - Yellow
        6: (128, 0, 0),       # Class 6 - Dark Red
        7: (0, 128, 0),       # Class 7 - Dark Green
        8: (0, 0, 128),       # Class 8 - Dark Blue
        9: (128, 128, 0),     # Class 9 - Olive
        10: (128, 0, 128),    # Class 10 - Purple
        11: (0, 128, 128),    # Class 11 - Teal
        12: (192, 192, 192),   # Class 12 - Silver
        13: (128, 128, 128),   # Class 13 - Gray
        14: (255, 165, 0),     # Class 14 - Orange
        15: (255, 20, 147),    # Class 15 - Deep Pink
        16: (0, 255, 127),     # Class 16 - Spring Green
        17: (75, 0, 130),      # Class 17 - Indigo
        18: (255, 105, 180),   # Class 18 - Hot Pink
        19: (0, 100, 0)        # Class 19 - Dark Green (Forest Green)
    }



    # Process the results
    for result in results:
        if result.masks is not None:
            for i, mask in enumerate(result.masks):
                # Convert the mask to a binary format (0s and 1s)
                binary_mask = mask.cpu().data.numpy()  # Get the mask as a NumPy array

                # Create a colored mask based on the class index
                color = color_map.get(i, (255, 255, 255))  # Default to white if class not in map
                colored_mask = np.zeros_like(img)  # Create an empty color mask
                if binary_mask.ndim > 2:
                    binary_mask = binary_mask.squeeze()
                resized_mask = cv2.resize(binary_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                # Apply the color to the mask
                # print(binary_mask.shape)
                colored_mask[resized_mask == 1] = color  # Color the pixels where the mask is 1

                # Combine the colored mask with the segmented image
                segmented_image = cv2.add(segmented_image, colored_mask)

    # Combine original image and segmented areas for visualization
    output_image = cv2.addWeighted(img, 1, segmented_image, 0.5, 0)
    #send the image through zmq to other scritps that does not support ROS2
    _, buffer = cv2.imencode('.jpg', output_image)
    img_as_text = base64.b64encode(buffer).decode('utf-8')
    send_socket.send_string(f"{img_as_text}")

 