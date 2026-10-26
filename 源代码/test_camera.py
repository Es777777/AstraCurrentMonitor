import queue
import threading
import time
import unittest
from unittest.mock import Mock,patch
import numpy as np
from camera import Camera
from main import scan_cameras

class FakeCapture:
    def __init__(self,can_open=True,read_ok=True):
        self.can_open=can_open;self.read_ok=read_ok;self.did_read=False;self.released=False
    def isOpened(self):return self.can_open
    def set(self,*args):return True
    def read(self):
        time.sleep(.01);self.did_read=True
        return self.read_ok,np.zeros((24,32,3),np.uint8) if self.read_ok else None
    def release(self):self.released=True

class CameraTests(unittest.TestCase):
    def test_camera_scan_only_lists_devices_that_open_and_return_a_frame(self):
        signals=Mock();devices=[]
        for index in range(10):
            cap=Mock();cap.isOpened.return_value=index in (1,4,7)
            cap.read.return_value=(index!=4,None);devices.append(cap)
        with patch('main.cv2.VideoCapture',side_effect=devices):scan_cameras('DSHOW',signals)
        signals.finished.emit.assert_called_once_with([1,7])
        for cap in devices:cap.release.assert_called_once()

    def test_pending_preserves_every_packet_and_one_frame_peak(self):
        camera=Camera.__new__(Camera);camera.frames=queue.Queue(maxsize=64)
        for t,v in [(1,.9),(2,1.),(3,.9)]:camera.frames.put((t,v))
        self.assertEqual(camera.pending(),[(1,.9),(2,1.),(3,.9)])
        self.assertEqual(camera.pending(),[])

    def test_connected_is_reported_only_after_successful_frame(self):
        fake=FakeCapture()
        with patch('camera.cv2.VideoCapture',return_value=fake):
            camera=Camera()
            event,message=camera.events.get(timeout=2)
            self.assertEqual(event,'connected');self.assertTrue(fake.did_read)
            camera.stop();camera.thread.join(timeout=2)
        self.assertTrue(fake.released);self.assertFalse(camera.thread.is_alive())

    def test_camera_open_failure_does_not_report_connected(self):
        fake=FakeCapture(can_open=False)
        with patch('camera.cv2.VideoCapture',return_value=fake):
            camera=Camera();camera.thread.join(timeout=2)
        events=[]
        while not camera.events.empty():events.append(camera.events.get()[0])
        self.assertIn('error',events);self.assertNotIn('connected',events);self.assertTrue(fake.released)

if __name__=='__main__':unittest.main()
