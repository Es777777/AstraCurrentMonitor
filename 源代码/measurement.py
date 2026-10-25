"""Five-second capture windows separated by a configurable idle interval."""
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import csv
import json
import sqlite3
from pathlib import Path

@dataclass
class Record:
    number: int
    started: str
    ended: str
    maximum: float | None
    valid_frames: int
    total_frames: int
    confirmed_frames: int
    quality: float
    status: str
    source: str
    evidence: str = ''
    config: str = ''
    dropped_frames: int = 0
    voltage_at_max: float | None = None
    maximum_voltage: float | None = None

class MeasurementEngine:
    def __init__(self, period=None, duration=5., repeats=1, confirm_span=.6, interval=10.):
        # Explicit period remains supported for callers that schedule start-to-start.
        # The application passes interval, which means END-to-next-START idle time.
        if period is None:period=duration+interval
        if not 0 < duration <= period or interval<0: raise ValueError('测量时长必须大于 0，测量间隔不能小于 0')
        if repeats < 1: raise ValueError('确认帧数至少为 1')
        self.period,self.duration,self.repeats,self.confirm_span=period,duration,repeats,confirm_span
        self.interval=self.period-self.duration
        self.running=False; self.window=None; self.number=0

    def start(self, now, wall=None, source='camera'):
        self.origin=now; self.wall_origin=wall or datetime.now()
        self.source=source; self.running=True; self.index=0; self.number=0
        self._new_window(0)

    def _new_window(self,index):
        self.index=index; self.number+=1
        self.window={'start':self.origin+index*self.period,'max':None,'total':0,'valid':0,
                     'confirmed':0,'quality':0.,'recent':deque(),'evidence':None,'dropped':0,
                     'voltage_at_max':None,'max_voltage':None,'voltage_valid':0}

    def _finish(self,status=None):
        w=self.window
        start=self.wall_origin+timedelta(seconds=w['start']-self.origin)
        end=start+timedelta(seconds=self.duration)
        fraction=w['valid']/w['total'] if w['total'] else 0
        if status is None:
            status='成功' if w['max'] is not None else '无有效读数'
            if w['max'] is not None and (fraction < .5 or w['confirmed'] < 3): status='有效帧偏少，请复核'
            if w['dropped']:status='采集帧丢失，请复核'
        rec=Record(self.number,start.isoformat(timespec='milliseconds'),end.isoformat(timespec='milliseconds'),
                   w['max'],w['valid'],w['total'],w['confirmed'],round(w['quality'],3),status,self.source)
        rec.dropped_frames=w['dropped']
        rec.voltage_at_max=w['voltage_at_max'];rec.maximum_voltage=w['max_voltage']
        evidence=w['evidence']; self.window=None
        return rec,evidence

    def advance(self, now):
        completed=[]
        if not self.running: return completed
        if self.window and now >= self.window['start']+self.duration:
            completed.append(self._finish())
        index=int(max(0.,now-self.origin)//self.period)
        if index > self.index:
            # A delayed UI must not fabricate measurements for missed periods.
            for missing in range(self.index+1,index):
                self._new_window(missing); completed.append(self._finish('调度中断，未采集'))
            self._new_window(index)
            if now >= self.window['start']+self.duration:
                completed.append(self._finish('调度中断，未采集'))
        return completed

    def observe(self, timestamp, reading, evidence=None, voltage=None):
        if not self.running or self.window is None: return False
        w=self.window
        if not w['start'] <= timestamp < w['start']+self.duration: return False
        w['total']+=1
        if voltage is not None and voltage.value is not None:
            w['voltage_valid']+=1
            if w['max_voltage'] is None or voltage.value>w['max_voltage']:
                w['max_voltage']=voltage.value
        q=w['recent']
        while q and timestamp-q[0][0] > self.confirm_span: q.popleft()
        if reading.value is None: return False
        w['valid']+=1
        q.append((timestamp,reading.value))
        if sum(v==reading.value for _,v in q) < self.repeats: return False
        w['confirmed']+=1
        if w['max'] is None or reading.value > w['max']:
            w['max']=reading.value; w['quality']=reading.confidence
            w['voltage_at_max']=voltage.value if voltage is not None else None
            w['evidence']=None if evidence is None else evidence.copy()
        return True

    def mark_dropped(self,count):
        if self.running and self.window:self.window['dropped']+=max(0,int(count))

    def stop(self, now=None):
        self.running=False
        if self.window:
            result=self._finish('已停止，窗口未完成')
            if now is not None:
                result[0].ended=(self.wall_origin+timedelta(seconds=now-self.origin)).isoformat(timespec='milliseconds')
            return [result]
        return []

    def phase(self,now):
        if not self.running: return '已停止',0
        if self.window: return '正在测量',max(0.,self.window['start']+self.duration-now)
        return '等待下一轮',max(0.,self.origin+(self.index+1)*self.period-now)

class RecordStore:
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'evidence').mkdir(exist_ok=True)
        self.db=sqlite3.connect(self.root/'records.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        self.db.commit()

    def add(self,record):
        with self.db:
            cursor=self.db.execute('INSERT INTO records(payload) VALUES(?)',(json.dumps(asdict(record),ensure_ascii=False),))
        return cursor.lastrowid

    def records(self):
        return [json.loads(v[0]) for v in self.db.execute('SELECT payload FROM records ORDER BY id')]

    def export(self,path):
        fields=list(Record.__dataclass_fields__)
        labels=['轮次','开始时间','结束时间','最大电流(A)','有效识别帧','采集帧','通过确认帧数','分段质量','状态','来源','证据图片','识别配置','缓冲丢失帧','峰值时电压(V)','最高电压(V)']
        with open(path,'w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.writer(stream); writer.writerow(labels)
            for record in self.records(): writer.writerow([record.get(k,'') for k in fields])

    def close(self): self.db.close()
