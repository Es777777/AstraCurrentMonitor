"""Capture on a background thread; retain every sample for peak measurement."""
import queue
import threading
import time
import cv2
from recognition import render_monitor_display

_CAPTURE_LOCK=threading.Lock()

class Camera:
    def __init__(self,index=0,backend='DSHOW',resolution=(1280,720),demo=False):
        self.index,self.backend,self.resolution,self.demo=index,backend,resolution,demo
        self.frames=queue.Queue(maxsize=64)
        self.dropped_frames=0
        self.events=queue.Queue()
        self.cancel=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)
        self.thread.start()

    def run(self):
        cap=None;owns_capture=False;announced=False
        try:
            if not self.demo:
                owns_capture=_CAPTURE_LOCK.acquire(timeout=3)
                if not owns_capture:raise RuntimeError('上一相机连接尚未释放，请稍等后重新连接。')
                if self.cancel.is_set():return
                api=cv2.CAP_DSHOW if self.backend=='DSHOW' else cv2.CAP_MSMF
                cap=cv2.VideoCapture(self.index,api)
                if not cap.isOpened(): raise RuntimeError('无法打开相机。请仅保留一个电流监测窗口，并关闭其它占用相机的软件，再检查相机编号、USB 连接及权限。')
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.resolution[1])
                cap.set(cv2.CAP_PROP_FPS,30)
            failures=0; started=time.monotonic()
            while not self.cancel.is_set():
                if self.demo:
                    values=[.9,.9,1.0,.9,.9,.9,.9]
                    frame,_,_=render_monitor_display(voltage=12.6,current=values[int((time.monotonic()-started)*4)%len(values)],noise=3)
                    self.cancel.wait(1/25)
                    ok=True
                else:
                    ok,frame=cap.read()
                captured=time.monotonic()
                if self.cancel.is_set(): break
                if not ok or frame is None:
                    failures+=1
                    if failures>=15: raise RuntimeError('相机连续读取失败，连接可能已断开。')
                    self.cancel.wait(.05); continue
                failures=0
                if not announced:
                    self.events.put(('connected','模拟' if self.demo else f'摄像头 {self.index} · {self.backend}'))
                    announced=True
                try:self.frames.put_nowait((captured,frame))
                except queue.Full:
                    try:self.frames.get_nowait();self.dropped_frames+=1
                    except queue.Empty:pass
                    self.frames.put_nowait((captured,frame))
        except Exception as exc:
            self.events.put(('error',str(exc)))
        finally:
            if cap is not None: cap.release()
            if owns_capture:_CAPTURE_LOCK.release()
            self.events.put(('closed',''))

    def latest(self):
        try: return self.frames.get_nowait()
        except queue.Empty: return None

    def pending(self):
        """Return every queued sample, in capture order, so a brief peak is retained."""
        result=[]
        for _ in range(self.frames.qsize()):
            try:result.append(self.frames.get_nowait())
            except queue.Empty:break
        return result

    def stop(self): self.cancel.set()
