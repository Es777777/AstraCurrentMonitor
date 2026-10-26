import os
import unittest
from pathlib import Path
import cv2
import numpy as np
from recognition import RecognitionConfig,recognize,render_display,render_monitor_display
from smart_display import locate_current,locate_voltage,read_row

class SmartDisplayTests(unittest.TestCase):
    def test_auto_detect_and_decode_visible_numeric_combinations(self):
        for value in [0,.1,.2,.9,1.,2.,1.1,11.1,12.3,22.2,33.3,44.4,55.5,66.6,77.7,88.8,99.9,19.8,21.5]:
            with self.subTest(value=value):
                frame,_=render_display(value)
                box,result,reason=locate_current(frame,3)
                self.assertIsNotNone(result,reason)
                self.assertEqual(int(result.text)/10,value)

    def test_empty_frame_and_blue_digits_are_rejected(self):
        frame,_=render_display(12.3)
        blue=frame[:,:,[2,1,0]].copy()
        self.assertIsNone(locate_current(blue,3)[0])
        self.assertIsNone(locate_current(np.zeros((480,640,3),np.uint8),3)[0])

    def test_dual_color_display_reads_blue_voltage_and_red_current(self):
        frame,voltage_roi,current_roi=render_monitor_display(12.6,.9,noise=2)
        voltage=recognize(frame,RecognitionConfig(roi=voltage_roi,digits=3,decimals=1,
            maximum=999999,color='blue',auto_align=True))
        current=recognize(frame,RecognitionConfig(roi=current_roi,digits=3,decimals=1,
            maximum=100,color='auto',auto_align=True))
        self.assertEqual(voltage.value,12.6)
        self.assertEqual(current.value,.9)
        self.assertIsNone(locate_voltage(frame[:,:,[2,1,0]],3)[1])

    def test_migrated_photo_set_when_supplied(self):
        root=os.environ.get('CURRENT_MONITOR_TEST_IMAGE_DIR')
        if not root:self.skipTest('Set CURRENT_MONITOR_TEST_IMAGE_DIR to the local migrated photo directory')
        expected={
            '000d95063ff2d306adf40b95e01ac7c0.jpg':'030',
            '03bd650eb26ace2ec5d3df351d55856b.jpg':'054',
            '29fc518e35a231aa3ca01e2a347d2c23.png':'078',
            '2ab68d690d8f0e404af96089c9e55fa8.jpg':'064',
            '2c25b31395683e12ee0f426620dd1ac7.jpg':'069',
            '36bfa7913c6863d3a55400bcad427c71.jpg':'036',
            '4f5e68d4415372ce1a873969b0c80f92.jpg':'016',
            '50fe3c7c41cd030da93998f25c45a385.jpg':'021',
            '6aafaa226397bd94873b80bce5fb8966.jpg':'073',
            'ca9b885c374beec7e004047e3ea684cf.jpg':'049',
        }
        for name,current_text in expected.items():
            path=Path(root)/name
            with self.subTest(image=name):
                frame=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
                self.assertIsNotNone(frame)
                _,voltage,_=locate_voltage(frame,3)
                self.assertIsNotNone(voltage)
                self.assertEqual(voltage.text,'126')
                _,current,_=locate_current(frame,3)
                if current_text is None:self.assertIsNone(current)
                else:
                    self.assertIsNotNone(current)
                    self.assertEqual(current.text,current_text)

    def test_deleted_digit_box_cannot_reappear_or_produce_a_reading(self):
        frame,roi=render_display(12.3)
        cfg=RecognitionConfig(roi=roi,cells=[(.1,.2,.2,.5),(.4,.2,.2,.5)])
        reading=recognize(frame,cfg)
        self.assertIsNone(reading.value)

    def test_live_photo_if_supplied(self):
        path=os.environ.get('CURRENT_MONITOR_TEST_IMAGE')
        if not path:self.skipTest('No private camera image supplied')
        frame=cv2.imread(path)
        box,result,reason=locate_current(frame,3)
        self.assertIsNotNone(result,reason)
        self.assertEqual(result.text,'002')
        x,y,xx,yy=map(int,box);h,w=frame.shape[:2]
        cfg=RecognitionConfig(roi=(x/w,y/h,(xx-x)/w,(yy-y)/h),color='auto',auto_align=True)
        self.assertEqual(recognize(frame,cfg).value,.2)

    def test_actual_one_with_decimal_is_not_stretched_into_two(self):
        fixtures=Path(__file__).parent/'fixtures'
        for filename,expected in [('actual_010_a.png','010'),('actual_010_b.png','010'),('actual_009.png','009')]:
            with self.subTest(filename=filename):
                frame=cv2.imdecode(np.fromfile(fixtures/filename,np.uint8),cv2.IMREAD_COLOR)
                self.assertIsNotNone(frame)
                result=read_row(frame,3)
                self.assertEqual(result.text,expected,result.reason)

    def test_bloom_at_roi_edge_does_not_discard_complete_current(self):
        path=Path(__file__).parent/'fixtures'/'actual_009_bloom.png'
        frame=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
        cfg=RecognitionConfig(roi=(0,0,1,1),auto_align=True,color='auto')
        result=recognize(frame,cfg)
        self.assertEqual(result.value,.9,result.reason)

    def test_cut_digit_is_still_rejected(self):
        frame,roi=render_display(.9)
        x,y,w,h=roi
        cfg=RecognitionConfig(roi=(x+w*.12,y,w*.88,h),auto_align=True,color='auto')
        result=recognize(frame,cfg)
        self.assertIsNone(result.value)
        self.assertIn('截断',result.reason)

if __name__=='__main__':unittest.main()
