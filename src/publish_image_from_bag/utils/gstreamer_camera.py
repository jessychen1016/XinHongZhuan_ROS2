# gstreamer_camera.py

import gi
import cv2
import numpy as np
import threading
import queue

gi.require_version('Gst', '1.0')
from gi.repository import Gst

class GStreamerCamera:
    def __init__(self, device='/dev/video2', width=1920, height=1080, framerate=30, show_img= False):
        Gst.init(None)

        # Create a GStreamer pipeline
        self.pipeline = Gst.parse_launch(
            f'v4l2src device={device} ! image/jpeg,width={width},height={height},framerate={framerate}/1 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink drop=true max-buffers=1'
        )

        # Get the appsink element
        self.appsink = self.pipeline.get_by_name('sink')
        self.appsink.set_property('emit-signals', True)
        self.appsink.set_property('sync', False)
        self.show_image = show_img

        # Queue for holding frames
        self.frame_queue = queue.Queue(maxsize=1)

        # Connect the new_sample function to the appsink
        self.appsink.connect('new-sample', self.new_sample)

        # Flag for processing
        self.running = True

        # Start the frame processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()

    def new_sample(self, sink):
        sample = sink.emit('pull-sample')
        buf = sample.get_buffer()
        caps = sample.get_caps()

        # Get width and height from caps
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')

        # Extract buffer data
        buf_size = buf.get_size()
        buf_data = buf.extract_dup(0, buf_size)

        # Create a NumPy array from the buffer data
        frame = np.frombuffer(buf_data, np.uint8)

        # Check if the frame is in BGRA format (4 channels)
        if frame.size == width * height * 4:  # BGRA
            frame = frame.reshape((height, width, 4))  # Reshape to (height, width, 4)
            bgr_frame = frame[:, :, :3]  # Discard the alpha channel
        elif frame.size == width * height * 3:  # BGR
            frame = frame.reshape((height, width, 3))  # Reshape to (height, width, 3)
            bgr_frame = frame
        else:
            print(f"Unexpected frame size: {frame.size}")
            return Gst.FlowReturn.ERROR

        # Put the frame into the queue for processing
        try:
            self.frame_queue.put_nowait(bgr_frame)
        except queue.Full:
            print("Queue is full, dropping frame.")

        return Gst.FlowReturn.OK

    def process_frames(self):
        while self.running:
            try:
                # Get a frame from the queue
                frame = self.frame_queue.get(timeout=1)  # Wait for 1 second for a frame
                
                if self.show_image: 
                    cv2.imshow('Video Stream', frame)
                    # Handle OpenCV events and exit condition
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

            except queue.Empty:
                continue

    def get_frame(self):
        """Retrieve the latest frame from the queue."""
        try:
            return self.frame_queue.get(timeout=1)  # Get the frame without blocking
        except queue.Empty:
            print("No frame available.")
            return None  # Return None if no frame is available

    def start(self):
        # Start playing the pipeline
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        # Clean up and stop the pipeline
        self.running = False
        self.pipeline.set_state(Gst.State.NULL)
        self.processing_thread.join()  # Wait for the processing thread to finish
        cv2.destroyAllWindows()

if __name__ == "__main__":
    camera = GStreamerCamera()
    camera.start()

    try:
        while True:
            pass
    except KeyboardInterrupt:
        camera.stop()
