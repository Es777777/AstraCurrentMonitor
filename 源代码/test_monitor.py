"""Run: python -m unittest -v test_monitor.py"""
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from recognition import RecognitionConfig,Reading,recognize,render_display,digit_boxes
from measurement import MeasurementEngine,RecordStore

def good(value):return Reading(value,str(value),.9,'有效')

class RecognizerTests(unittest.TestCase):
    def test_independent_polygon_led_geometry(self):
        # Independent beveled LED geometry (not recognition sampling rectangles).
        polygons=[[(17,6),(81,6),(73,15),(25,15)],
                  [(84,9),(92,18),(92,68),(83,77),(78,70),(78,20)],
                  [(83,83),(92,92),(92,142),(84,151),(78,140),(78,93)],
                  [(25,145),(73,145),(81,154),(17,154)],
                  [(15,83),(21,93),(21,140),(14,151),(7,142),(7,92)],
                  [(14,9),(21,20),(21,70),(15,77),(7,68),(7,18)],
                  [(24,75),(74,75),(81,80),(74,85),(24,85),(17,80)]]
        active={'0':'abcdef','1':'bc','2':'abdeg','3':'abcdg','4':'bcfg',
                '5':'acdfg','6':'acdefg','7':'abc','8':'abcdefg','9':'abcdfg'}
        for text in ['000','123','456','789','981','610','888','777']:
            image=np.full((200,340,3),18,np.uint8)
            for index,char in enumerate(text):
                for letter,poly in zip('abcdefg',polygons):
                    if letter in active[char]:
                        points=np.array(poly,np.int32)+[20+index*100,20]
                        cv2.fillConvexPoly(image,points,(18,25,220))
            image=cv2.GaussianBlur(image,(3,3),.7)
            cfg=RecognitionConfig(roi=(20/340,20/200,300/340,160/200))
            self.assertEqual(recognize(image,cfg).value,int(text)/10)

    def test_all_three_digit_combinations(self):
        for number in range(1000):
            frame,roi=render_display(number/10,size=(480,270))
            reading=recognize(frame,RecognitionConfig(roi=roi,maximum=100))
            self.assertEqual(reading.value,number/10,f'{number}: {reading.reason}')

    def test_blur_noise_and_brightness(self):
        for value in (0,1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,12.3,88.8):
            for noise in (4,15,25):
                f,roi=render_display(value,noise=noise)
                f=cv2.GaussianBlur(f,(3,3),.7)
                self.assertEqual(recognize(f,RecognitionConfig(roi=roi)).value,value)

    def test_fixed_decimal_and_individual_boxes(self):
        f,roi=render_display(123.4,4,1)
        cfg=RecognitionConfig(roi=roi,digits=4,decimals=1,maximum=1000)
        cfg.cells=digit_boxes(cfg)
        self.assertEqual(recognize(f,cfg).value,123.4)
        cfg.decimals=2
        self.assertEqual(recognize(f,cfg).value,12.34)

    def test_blank_occlusion_and_range_rejected(self):
        f,roi=render_display(12.3)
        self.assertIsNone(recognize(f,RecognitionConfig(roi=roi,maximum=10)).value)
        self.assertIsNone(recognize(np.zeros_like(f),RecognitionConfig(roi=roi)).value)
        self.assertIsNone(recognize(f,RecognitionConfig()).value)
        # Remove a vertical strip through a complete digit, producing an invalid pattern.
        f,roi=render_display(88.8);x,y,w,h=roi;cw=w/3
        x1=int((x+.72*cw)*f.shape[1]);x2=int((x+.99*cw)*f.shape[1])
        f[int((y+.45*h)*f.shape[0]):int((y+.86*h)*f.shape[0]),x1:x2]=0
        self.assertIsNone(recognize(f,RecognitionConfig(roi=roi)).value)

    def test_green_light_is_not_red(self):
        f,roi=render_display(88.8);f=f[:,:,[0,2,1]].copy()
        self.assertIsNone(recognize(f,RecognitionConfig(roi=roi)).value)

class MeasurementTests(unittest.TestCase):
    def test_maximum_confirms_and_discards_isolated_spike(self):
        engine=MeasurementEngine(period=10,repeats=2);engine.start(100,datetime(2026,1,1))
        for t,v in [(100.1,1.2),(100.2,1.2),(101,99.9),(102,2.3),(102.2,2.3),(104.9,1.1)]:engine.observe(t,good(v))
        result=engine.advance(105)[0][0]
        self.assertEqual(result.maximum,2.3);self.assertEqual(result.valid_frames,6)
        self.assertEqual(result.ended,'2026-01-01T00:00:05.000')
        self.assertEqual(engine.phase(105),('等待下一轮',5))

    def test_confirmation_does_not_cross_windows_or_timeout(self):
        e=MeasurementEngine(period=10,repeats=2);e.start(0)
        e.observe(.1,good(9));e.observe(.9,good(9))
        self.assertIsNone(e.advance(5)[0][0].maximum)
        e.advance(10);e.observe(10.1,good(9))
        self.assertIsNone(e.advance(15)[0][0].maximum)

    def test_boundaries_and_fixed_cadence(self):
        e=MeasurementEngine(period=10,repeats=1);e.start(50)
        e.observe(49.99,good(99));e.observe(50.1,good(1));e.observe(54.99,good(2));e.observe(55,good(88))
        self.assertEqual(e.advance(55)[0][0].maximum,2)
        e.advance(59.9);self.assertIsNone(e.window)
        e.advance(60.2);self.assertEqual(e.window['start'],60)
        e.observe(59.9,good(99));e.observe(60.3,good(3))
        self.assertEqual(e.advance(65)[0][0].maximum,3)

    def test_invalid_only_and_disconnect(self):
        e=MeasurementEngine(period=10);e.start(0)
        for t in (.1,.2,.3):e.observe(t,Reading(None,'',0,'无效'))
        result=e.advance(5)[0][0];self.assertIsNone(result.maximum);self.assertEqual(result.status,'无有效读数')
        e.advance(10);e.observe(10.1,good(1));e.observe(10.2,good(1))
        rec=e.stop(11)[0][0];self.assertEqual(rec.status,'已停止，窗口未完成');self.assertFalse(e.running)

    def test_large_time_jump_marks_missing_windows(self):
        e=MeasurementEngine(period=10);e.start(0)
        results=e.advance(31)
        self.assertEqual(len(results),3);self.assertEqual(e.window['start'],30)
        self.assertEqual(results[1][0].status,'调度中断，未采集')

    def test_equal_period_and_duration(self):
        e=MeasurementEngine(period=5,duration=5,repeats=1);e.start(0);e.observe(4.9,good(2))
        self.assertEqual(e.advance(5)[0][0].maximum,2);self.assertEqual(e.window['start'],5)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):MeasurementEngine(3,5)
        with self.assertRaises(ValueError):MeasurementEngine(repeats=0)
        with self.assertRaises(ValueError):MeasurementEngine(interval=-1)

    def test_five_second_capture_then_ten_second_idle(self):
        e=MeasurementEngine();e.start(0,datetime(2026,1,1))
        self.assertEqual((e.duration,e.interval,e.period,e.repeats),(5,10,15,1))
        e.observe(0,good(.9));e.observe(4.999,good(1))
        first=e.advance(5)[0][0]
        self.assertEqual(first.maximum,1)
        self.assertEqual(e.phase(5),('等待下一轮',10))
        self.assertFalse(e.observe(9,good(99)))
        self.assertEqual(e.advance(14.999),[]);self.assertIsNone(e.window)
        e.advance(15);self.assertEqual(e.window['start'],15)
        e.observe(15,good(.9))
        second=e.advance(20)[0][0]
        self.assertEqual(second.maximum,.9)
        self.assertEqual(second.started,'2026-01-01T00:00:15.000')
        e.advance(30);self.assertEqual(e.window['start'],30)

    def test_single_brief_high_is_preserved_even_after_lower_readings(self):
        e=MeasurementEngine();e.start(0)
        for t,v in [(.1,.9),(.2,.9),(.21,1),(.3,.9),(4.9,.9)]:
            e.observe(t,good(v))
        result=e.advance(5)[0][0]
        self.assertEqual(result.maximum,1)
        self.assertEqual(result.confirmed_frames,5)

    def test_interval_is_independent_of_duration(self):
        e=MeasurementEngine(duration=5,interval=2);e.start(0)
        e.advance(5);self.assertEqual(e.phase(5),('等待下一轮',2))
        e.advance(7);self.assertEqual(e.window['start'],7)

    def test_dropped_samples_are_flagged_without_hiding_observed_maximum(self):
        e=MeasurementEngine();e.start(0);e.observe(.1,good(1));e.mark_dropped(2)
        result=e.advance(5)[0][0]
        self.assertEqual(result.maximum,1);self.assertEqual(result.dropped_frames,2)
        self.assertEqual(result.status,'采集帧丢失，请复核')

    def test_durable_store_and_csv(self):
        # Use an ordinary workspace directory to accommodate sandbox ACL inheritance.
        root=Path(__file__).resolve().parents[2]/'work'/'test_records';root.mkdir(parents=True,exist_ok=True)
        e=MeasurementEngine(repeats=1);e.start(0);e.observe(.1,good(2.5));rec=e.advance(5)[0][0]
        s=RecordStore(root);prior=len(s.records());s.add(rec);s.export(root/'out.csv');s.close()
        s=RecordStore(root);self.assertEqual(len(s.records()),prior+1);s.close()
        content=(root/'out.csv').read_text(encoding='utf-8-sig');self.assertIn('最大电流(A)',content);self.assertIn('2.5',content)

if __name__=='__main__':unittest.main()
