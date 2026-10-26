"""Camera-based current and voltage display monitor for Windows."""
import argparse
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback

os.environ.setdefault('OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS','0')
import cv2
import numpy as np
from PyQt5.QtCore import Qt,QRectF,QPointF,QTimer,pyqtSignal as Signal,QUrl,QLockFile,QObject
from PyQt5.QtGui import QColor,QFont,QFontDatabase,QImage,QPainter,QPen,QPixmap,QDesktopServices
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,
    QLabel,QPushButton,QComboBox,QSpinBox,QDoubleSpinBox,QFormLayout,QGroupBox,
    QCheckBox,QScrollArea,QTableWidget,QTableWidgetItem,QHeaderView,QFileDialog,QMenu,
    QMessageBox,QPlainTextEdit,QProgressBar,QSplitter,QAbstractItemView,QFrame,QSizePolicy)
from recognition import RecognitionConfig,recognize,digit_boxes,render_display,render_monitor_display,box_pixels
from camera import Camera
from measurement import MeasurementEngine,RecordStore
from smart_display import locate_voltage

VERSION='1.3.3'

class CameraScanSignals(QObject):
    finished=Signal(list)

def scan_cameras(backend,signals):
    api=cv2.CAP_DSHOW if backend=='DSHOW' else cv2.CAP_MSMF
    found=[]
    for index in range(10):
        cap=cv2.VideoCapture(index,api)
        try:
            if cap.isOpened():
                ok,_=cap.read()
                if ok:found.append(index)
        finally:cap.release()
    signals.finished.emit(found)

STYLE='''
QWidget {font-family: "Microsoft YaHei UI", "Microsoft YaHei"; font-size: 13px; color:#25334a;}
QMainWindow, #root {background:#f1f4f8;}
QGroupBox {background:white; border:1px solid #dce3ed; border-radius:9px; margin-top:15px; padding:14px 10px 10px; font-weight:600;}
QGroupBox::title {subcontrol-origin:margin; left:12px; padding:0 5px;}
QPushButton {background:white; border:1px solid #ccd6e3; border-radius:6px; padding:8px 12px;}
QPushButton:hover {background:#edf5ff; border-color:#6eabdd;}
QPushButton:disabled {color:#a5adba; background:#eff2f5; border-color:#e0e5eb;}
QPushButton#primary {background:#087f8c;color:white;border:0;font-weight:600;}
QPushButton#primary:hover {background:#066673;}
QPushButton#stop {color:#b43c3c;}
QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox {background:white;border:1px solid #cfd8e5;border-radius:4px;padding:5px;min-height:21px;}
QComboBox::drop-down {width:22px;border:0;}
QTableWidget {background:white;alternate-background-color:#f6f8fb;border:0;gridline-color:#edf0f5;selection-background-color:#dff2f2;selection-color:#25334a;}
QHeaderView::section {background:#eaf0f6;color:#617086;border:0;padding:9px;font-weight:600;}
QPlainTextEdit {background:#f8fafc;border:1px solid #e1e6ee;border-radius:6px;font-size:12px;}
QProgressBar {border:0;border-radius:4px;background:#e2e9f0;text-align:center;min-height:8px;max-height:8px;}
QProgressBar::chunk {background:#16a1a3;border-radius:4px;}
QScrollArea {border:0;background:transparent;}
QCheckBox {spacing:7px;}
QLabel#hint {color:#6d7e94;font-size:12px;}
'''

def image_pixmap(bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    return QPixmap.fromImage(QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format_RGB888).copy())

def single_instance(app,data_dir=None):
    """Prevent duplicate camera windows from the same portable installation."""
    folder=Path(data_dir) if data_dir else (Path(sys.executable).parent/'data' if getattr(sys,'frozen',False) else Path(__file__).parent/'data')
    folder.mkdir(parents=True,exist_ok=True)
    lock=QLockFile(str(folder/'application.lock'));lock.setStaleLockTime(0)
    if lock.tryLock(0):return lock
    if lock.error()==QLockFile.LockError.LockFailedError:return None
    raise RuntimeError('无法建立启动保护，请将软件解压到可写入的文件夹后重试。')

class Preview(QWidget):
    selected=Signal(tuple)
    delete_requested=Signal(int)
    clear_requested=Signal()
    cancel_requested=Signal()
    def __init__(self):
        super().__init__(); self.setMinimumSize(480,250)
        self.pix=None; self.boxes=[]; self.roi=None; self.voltage_roi=None; self.drag=None; self.mode=''; self.demo=False
        self.selected_index=None
        self.setMouseTracking(True);self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_OpaquePaintEvent,True)

    def display_rect(self):
        if self.pix is None: return QRectF()
        scale=min(self.width()/self.pix.width(),self.height()/self.pix.height())
        w,h=self.pix.width()*scale,self.pix.height()*scale
        return QRectF((self.width()-w)/2,(self.height()-h)/2,w,h)

    def set_frame(self,frame): self.pix=image_pixmap(frame); self.update()

    def paintEvent(self,event):
        p=QPainter(self);p.setClipRect(self.rect()); p.fillRect(self.rect(),QColor('#101b2d'))
        r=self.display_rect()
        if self.pix is None:
            p.setPen(QColor('#afc0d6'));p.setFont(QFont('Microsoft YaHei UI',14))
            p.drawText(self.rect(),Qt.AlignCenter,'连接任意可用摄像头查看画面\n或点击「模拟演示」体验测量流程')
            return
        p.drawPixmap(r,self.pix,QRectF(self.pix.rect()))
        if self.voltage_roi:
            x,y,w,h=self.voltage_roi;p.setPen(QPen(QColor('#68b7ff'),2));area=QRectF(r.x()+x*r.width(),r.y()+y*r.height(),w*r.width(),h*r.height())
            p.drawRect(area);p.drawText(area.topLeft()+QPointF(3,15),'上排电压 V')
        if self.roi:
            x,y,w,h=self.roi;p.setPen(QPen(QColor('#ffd578' if self.selected_index==-1 else '#ff7e69'),3 if self.selected_index==-1 else 2));area=QRectF(r.x()+x*r.width(),r.y()+y*r.height(),w*r.width(),h*r.height())
            p.drawRect(area);p.drawText(area.topLeft()+QPointF(3,15),'下排电流 A')
        for i,(x,y,w,h) in enumerate(self.boxes):
            p.setPen(QPen(QColor('#ffd578' if self.selected_index==i else '#66d6ff'),3 if self.selected_index==i else 1)); area=QRectF(r.x()+x*r.width(),r.y()+y*r.height(),w*r.width(),h*r.height())
            p.drawRect(area);p.drawText(area.topLeft()+QPointF(3,15),str(i+1))
        if self.drag:
            p.setPen(QPen(QColor('#ffd578'),2,Qt.DashLine));p.drawRect(QRectF(self.drag[0],self.drag[1]).normalized())
        if self.demo:
            p.fillRect(12,12,213,32,QColor('#a56515'));p.setPen(QColor('white'));p.drawText(22,34,'模拟演示 · 非真实测量')
        if self.mode:
            p.fillRect(0,self.height()-35,self.width(),35,QColor('#203c57'));p.setPen(QColor('white'));p.drawText(12,self.height()-12,self.mode)

    def mousePressEvent(self,event):
        self.setFocus()
        if self.mode and self.display_rect().contains(event.localPos()):self.drag=[event.localPos(),event.localPos()]
        elif not self.mode:
            r=self.display_rect();point=event.localPos();self.selected_index=None
            for i,box in reversed(list(enumerate(self.boxes))):
                x,y,w,h=box
                if QRectF(r.x()+x*r.width(),r.y()+y*r.height(),w*r.width(),h*r.height()).contains(point):self.selected_index=i;break
            if self.selected_index is None and self.roi:
                x,y,w,h=self.roi
                if QRectF(r.x()+x*r.width(),r.y()+y*r.height(),w*r.width(),h*r.height()).contains(point):self.selected_index=-1
            if self.selected_index is None and self.voltage_roi:
                x,y,w,h=self.voltage_roi
                if QRectF(r.x()+x*r.width(),r.y()+y*r.height(),w*r.width(),h*r.height()).contains(point):self.selected_index=-2
            self.update()

    def mouseMoveEvent(self,event):
        if self.drag: self.drag[1]=event.localPos();self.update()

    def mouseReleaseEvent(self,event):
        if not self.drag:return
        self.drag[1]=event.localPos(); area=QRectF(*self.drag).normalized().intersected(self.display_rect()); self.drag=None
        r=self.display_rect()
        if area.width()>8 and area.height()>12:
            self.selected.emit(((area.x()-r.x())/r.width(),(area.y()-r.y())/r.height(),area.width()/r.width(),area.height()/r.height()))
        self.update()

    def keyPressEvent(self,event):
        if event.key() in (Qt.Key_Delete,Qt.Key_Backspace):
            if self.selected_index is not None:self.delete_requested.emit(self.selected_index)
        elif event.key()==Qt.Key_Escape:self.cancel_requested.emit()
        else:super().keyPressEvent(event)

    def contextMenuEvent(self,event):
        menu=QMenu(self);delete=menu.addAction('删除选中的框');delete.setEnabled(self.selected_index is not None)
        clear=menu.addAction('清空所有框');action=menu.exec_(event.globalPos())
        if action==delete and self.selected_index is not None:self.delete_requested.emit(self.selected_index)
        elif action==clear:self.clear_requested.emit()

class MainWindow(QMainWindow):
    def __init__(self,data_dir=None,scan_on_start=True):
        super().__init__()
        self.data_dir=Path(data_dir or (Path(sys.executable).parent/'data' if getattr(sys,'frozen',False) else Path(__file__).parent/'data'))
        self.store=RecordStore(self.data_dir)
        self.cfg=RecognitionConfig(auto_align=True,color='auto',row='current',maximum=999999)
        self.voltage_cfg=RecognitionConfig(digits=3,decimals=1,maximum=999999,auto_align=True,color='blue',row='voltage')
        self.camera=None; self.retiring=[]; self.frame=None;self.last_frame=0.; self.connected=False
        self.engine=MeasurementEngine();self.calibrating=None;self.last_shape=None; self.last_reading=None;self.last_voltage_reading=None;self.last_drop_count=0
        self.setWindowTitle(f'通用摄像头电流监测 · {VERSION}');self.resize(1380,940);self.setMinimumSize(900,680)
        self.build_ui();self.load_settings();self.refresh_records()
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(50)
        if scan_on_start:QTimer.singleShot(150,self.refresh_cameras)
        self.log('就绪。默认每轮采集 5 秒，测量结束后等待 10 秒；记录区间内所有有效读数的最大值。')

    def build_ui(self):
        root=QWidget();root.setObjectName('root');self.setCentralWidget(root)
        layout=QVBoxLayout(root);layout.setContentsMargins(20,15,20,15);layout.setSpacing(10)
        title=QHBoxLayout();heading=QLabel('通用摄像头 / 电流监测');heading.setStyleSheet('font-size:24px;font-weight:700;color:#16334b;')
        title.addWidget(heading);title.addStretch();self.connection=QLabel('●  相机未连接');title.addWidget(self.connection);layout.addLayout(title)
        self.notice=QLabel('通用摄像头 · 上排蓝色为电压 / 下排红色为电流 · 1 位小数    |    首次使用请核对实物读数')
        self.notice.setStyleSheet('background:#e1edf6;color:#365c7e;padding:10px;border-radius:6px;');layout.addWidget(self.notice)
        stats=QHBoxLayout();self.voltage=self.stat(stats,'上排电压','— V','#1769aa');self.current=self.stat(stats,'下排电流','— A','#b74836');self.peak=self.stat(stats,'本轮最大电流','— A')
        self.phase_label=self.stat(stats,'测量状态','已停止');self.count_label=self.stat(stats,'已保存记录','0');layout.addLayout(stats)
        self.progress=QProgressBar();self.progress.setRange(0,1000);self.progress.setTextVisible(False);layout.addWidget(self.progress)
        self.main_splitter=QSplitter(Qt.Horizontal);self.main_splitter.setChildrenCollapsible(False);layout.addWidget(self.main_splitter,1)
        left=QWidget();left.setMinimumWidth(0);left.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred)
        ll=QVBoxLayout(left);ll.setContentsMargins(0,0,8,0)
        self.preview=Preview();self.preview.setMinimumHeight(220);self.preview.selected.connect(self.on_selected)
        self.preview.delete_requested.connect(self.delete_box);self.preview.clear_requested.connect(self.clear_boxes);self.preview.cancel_requested.connect(self.cancel_selection)
        ll.addWidget(self.preview,3)
        tools=QGridLayout();tools.setHorizontalSpacing(6);tools.setVerticalSpacing(6)
        self.auto_btn=QPushButton('① 自动定位上下两行');self.auto_btn.clicked.connect(self.auto_locate)
        self.roi_btn=QPushButton('② 框选电流大致区域');self.roi_btn.clicked.connect(self.select_roi)
        self.voltage_roi_btn=QPushButton('③ 框选电压区域');self.voltage_roi_btn.clicked.connect(self.select_voltage_roi)
        self.cell_btn=QPushButton('逐位框选');self.cell_btn.clicked.connect(self.select_cells)
        self.demo_btn=QPushButton('模拟演示');self.demo_btn.clicked.connect(self.start_demo)
        for index,button in enumerate((self.auto_btn,self.roi_btn,self.voltage_roi_btn,self.cell_btn,self.demo_btn)):
            button.setMinimumHeight(34);tools.addWidget(button,index//3,index%3)
        for column in range(3):tools.setColumnStretch(column,1)
        ll.addLayout(tools)
        edit=QHBoxLayout();self.delete_btn=QPushButton('删除选中框');self.delete_btn.clicked.connect(self.delete_selected)
        self.clear_btn=QPushButton('清空所有框');self.clear_btn.clicked.connect(self.clear_boxes)
        edit.addWidget(self.delete_btn);edit.addWidget(self.clear_btn)
        hint=QLabel('单击框使其变黄，按 Delete 删除；Esc 取消框选');hint.setObjectName('hint');edit.addWidget(hint,1);ll.addLayout(edit)
        self.read_status=QLabel('识别区域尚未设置');self.read_status.setWordWrap(True);ll.addWidget(self.read_status)
        masks=QHBoxLayout();self.crop_view=QLabel('电流局部');self.mask_view=QLabel('红色提取预览')
        for widget in (self.crop_view,self.mask_view):
            widget.setAlignment(Qt.AlignCenter);widget.setFixedHeight(62);widget.setStyleSheet('background:#172338;color:#b5c3d6;border-radius:5px;');masks.addWidget(widget,1)
        ll.addLayout(masks)
        row=QHBoxLayout();rh=QLabel('测量记录');rh.setStyleSheet('font-size:16px;font-weight:600;');row.addWidget(rh);row.addStretch()
        export=QPushButton('导出 CSV');export.clicked.connect(self.export_csv);row.addWidget(export)
        folder=QPushButton('打开记录目录');folder.clicked.connect(lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.data_dir))));row.addWidget(folder);ll.addLayout(row)
        self.table=QTableWidget(0,9);self.table.setHorizontalHeaderLabels(['开始时间','最大电流 / A','峰值时电压 / V','最高电压 / V','有效 / 采集帧','确认帧','质量','状态','来源 / 证据'])
        self.table.setAlternatingRowColors(True);self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().hide()
        header=self.table.horizontalHeader();header.setSectionResizeMode(QHeaderView.Interactive)
        for column,width in enumerate((142,88,100,100,92,58,55,90)):
            self.table.setColumnWidth(column,width)
        header.setStretchLastSection(True);self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded);self.table.setMinimumHeight(120)
        self.table.setMinimumWidth(0);self.table.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Expanding)
        self.table.cellDoubleClicked.connect(self.open_evidence);ll.addWidget(self.table,2)
        self.logs=QPlainTextEdit();self.logs.setReadOnly(True);self.logs.setMaximumBlockCount(250);self.logs.setFixedHeight(52);ll.addWidget(self.logs)
        left_scroll=QScrollArea();left_scroll.setWidgetResizable(True);left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);left_scroll.setWidget(left)
        self.main_splitter.addWidget(left_scroll)
        scroll=QScrollArea();scroll.setWidgetResizable(True);scroll.setMinimumWidth(320);scroll.setMaximumWidth(385)
        settings=QWidget();sl=QVBoxLayout(settings);sl.setContentsMargins(2,0,2,0)
        camera_group=QGroupBox('相机连接');form=QFormLayout(camera_group)
        self.camera_index=QComboBox()
        for index in range(10):self.camera_index.addItem(f'摄像头 {index}',index)
        form.addRow('摄像头',self.camera_index)
        self.camera_index.setCurrentIndex(0)
        self.scan_btn=QPushButton('刷新摄像头列表');self.scan_btn.clicked.connect(self.refresh_cameras);form.addRow(self.scan_btn)
        self.backend=QComboBox();self.backend.addItems(['DSHOW','MSMF']);form.addRow('接口',self.backend)
        self.resolution=QComboBox();self.resolution.addItems(['1280 × 720','640 × 480']);form.addRow('请求分辨率',self.resolution)
        self.resolution.setCurrentIndex(1)
        self.connect_btn=QPushButton('连接相机');self.connect_btn.clicked.connect(self.toggle_camera);form.addRow(self.connect_btn)
        tip=QLabel('扫描结果会列出当前可打开的摄像头。\n连接失败时可切换 DSHOW / MSMF 接口。');tip.setObjectName('hint');tip.setWordWrap(True);form.addRow(tip);sl.addWidget(camera_group)
        recognition=QGroupBox('数显识别设置');rf=QFormLayout(recognition)
        self.digits=QSpinBox();self.digits.setRange(2,8);self.digits.setValue(3);rf.addRow('总数字位数',self.digits)
        self.decimals=QSpinBox();self.decimals.setRange(0,4);self.decimals.setValue(1);rf.addRow('小数点后位数',self.decimals)
        self.color=QComboBox();self.color.addItems(['自动去光晕（推荐）','红色数码管','灰度亮字']);rf.addRow('识别颜色',self.color)
        self.auto_align=QCheckBox('自动分离数字并校正倾斜');self.auto_align.setChecked(True);rf.addRow(self.auto_align)
        self.brightness=QSpinBox();self.brightness.setRange(15,250);self.brightness.setValue(90);rf.addRow('手动亮度阈值',self.brightness)
        self.red_diff=QSpinBox();self.red_diff.setRange(5,180);self.red_diff.setValue(35);rf.addRow('红色差值阈值',self.red_diff)
        self.lower=QDoubleSpinBox();self.lower.setRange(0,999999);self.lower.setDecimals(3);rf.addRow('最小电流 / A',self.lower)
        self.upper=QDoubleSpinBox();self.upper.setRange(.001,999999);self.upper.setDecimals(3);self.upper.setValue(999999);rf.addRow('最大电流 / A',self.upper)
        self.upper.setToolTip('这是识别结果筛选阈值，不是仪表量程或电流峰值上限；请按实际设备需要设置。')
        self.repeats=QSpinBox();self.repeats.setRange(1,10);self.repeats.setValue(1);rf.addRow('读数确认帧数',self.repeats)
        self.blanks=QCheckBox('允许最左侧空白位补 0');self.blanks.setChecked(True);rf.addRow(self.blanks)
        tip=QLabel('①自动定位上排蓝色电压与下排红色电流。\n定位失败时分别框选上排电压、下排电流。\n总位数不含小数点；小数位按设置解释。\n确认前请核对 V、A 单位及实物小数点。\n单击框选择，按 Delete 删除。');tip.setObjectName('hint');tip.setWordWrap(True);rf.addRow(tip);sl.addWidget(recognition)
        measure=QGroupBox('自动测量');mf=QFormLayout(measure)
        self.interval=QSpinBox();self.interval.setRange(0,3600);self.interval.setValue(10);mf.addRow('两次测量间隔 / 秒',self.interval)
        self.duration=QSpinBox();self.duration.setRange(1,3600);self.duration.setValue(5);mf.addRow('采集时长 / 秒',self.duration)
        self.verified=QCheckBox('已核对电压 V、电流 A 和小数点');mf.addRow(self.verified)
        self.start_btn=QPushButton('开始自动测量');self.start_btn.setObjectName('primary');self.start_btn.clicked.connect(self.start_measure);mf.addRow(self.start_btn)
        self.stop_btn=QPushButton('停止测量');self.stop_btn.setObjectName('stop');self.stop_btn.setEnabled(False);self.stop_btn.clicked.connect(self.stop_measure);mf.addRow(self.stop_btn)
        tip=QLabel('默认：测 5 秒 → 等待 10 秒 → 再测。\n记录电流峰值、峰值时电压与最高电压。\n默认 1 帧确认，保留区间内的有效瞬态峰值。');tip.setObjectName('hint');tip.setWordWrap(True);mf.addRow(tip)
        sl.addWidget(measure);sl.addStretch();scroll.setWidget(settings)
        right=QWidget();right.setMinimumWidth(0);right.setMaximumWidth(385)
        rl=QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);rl.addWidget(scroll,1)
        self.main_splitter.addWidget(right);self.main_splitter.setSizes([970,350])
        self.lock_controls=[self.camera_index,self.scan_btn,self.backend,self.resolution,self.digits,self.decimals,self.color,self.auto_align,self.brightness,self.red_diff,self.lower,self.upper,self.repeats,self.blanks,self.interval,self.duration,self.auto_btn,self.roi_btn,self.voltage_roi_btn,self.cell_btn,self.delete_btn,self.clear_btn,self.demo_btn,self.verified]
        self.camera_scan_signals=CameraScanSignals(self);self.camera_scan_signals.finished.connect(self.camera_scan_finished)
        for control in (self.digits,self.decimals,self.brightness,self.red_diff,self.lower,self.upper,self.repeats,self.interval,self.duration):control.valueChanged.connect(self.settings_changed)
        self.color.currentIndexChanged.connect(self.settings_changed);self.blanks.toggled.connect(self.settings_changed)
        self.auto_align.toggled.connect(self.settings_changed)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if not hasattr(self,'main_splitter'):return
        orientation=Qt.Horizontal if self.centralWidget().width()>=1080 else Qt.Vertical
        if self.main_splitter.orientation()!=orientation:
            self.main_splitter.setOrientation(orientation)
            if orientation==Qt.Vertical:
                self.main_splitter.setSizes([max(310,int(self.main_splitter.height()*.68)),max(170,int(self.main_splitter.height()*.32))])
            else:self.main_splitter.setSizes([max(600,int(self.main_splitter.width()*.73)),max(320,int(self.main_splitter.width()*.27))])

    def stat(self,layout,label,value,color='#143e55'):
        box=QWidget();box.setStyleSheet('background:white;border-radius:8px;');v=QVBoxLayout(box);v.setContentsMargins(16,10,16,10)
        caption=QLabel(label);caption.setStyleSheet('color:#728399;font-size:12px;');number=QLabel(value);number.setStyleSheet(f'color:{color};font-size:25px;font-weight:600;')
        v.addWidget(caption);v.addWidget(number);layout.addWidget(box,1);return number

    def log(self,text):self.logs.appendPlainText(datetime.now().strftime('%H:%M:%S')+'  '+text)

    def reset_reading_views(self):
        self.last_reading=None;self.current.setText('— A')
        self.last_voltage_reading=None
        if hasattr(self,'voltage'):self.voltage.setText('— V')
        self.crop_view.clear();self.crop_view.setText('电流局部')
        self.mask_view.clear();self.mask_view.setText('亮段提取预览')

    def cancel_selection(self):
        self.calibrating=None;self.preview.mode='';self.preview.drag=None
        self.preview.selected_index=None;self.preview.update()

    def clear_boxes(self):
        if self.engine.running:return self.alert('请先停止测量，再编辑识别区域。')
        self.cancel_selection();self.cfg.roi=None;self.cfg.cells=[];self.cfg.voltage_roi=None
        self.verified.setChecked(False);self.reset_reading_views();self.sync_config();self.save_settings()
        self.read_status.setText('所有框已清空，点击“自动定位上下两行”重新识别')
        self.log('已清空所有识别框及旧预览。')

    def delete_selected(self):
        if self.preview.selected_index is None:return self.alert('先在画面中单击一个框，框变黄后即可删除。')
        self.delete_box(self.preview.selected_index)

    def delete_box(self,index):
        if self.engine.running:return self.alert('请先停止测量，再编辑识别区域。')
        if index==-1:return self.clear_boxes()
        if index==-2:
            self.cfg.voltage_roi=None;self.verified.setChecked(False);self.preview.selected_index=None
            self.reset_reading_views();self.sync_config();self.save_settings();self.read_status.setText('上排电压框已删除，请重新定位或框选。');return
        cells=list(digit_boxes(self.cfg))
        if 0<=index<len(cells):
            del cells[index];self.cfg.cells=cells
            if not cells:self.cfg.roi=None
            self.verified.setChecked(False);self.preview.selected_index=None
            self.reset_reading_views();self.sync_config();self.save_settings()
            self.read_status.setText('单字框已删除，请重新定位或重新逐位框选')
            self.log(f'已删除第 {index+1} 个单字框。')

    def auto_locate(self):
        if self.engine.running:return
        if self.frame is None:return self.alert('请先连接相机，获得实际画面。')
        from smart_display import locate_current,locate_voltage
        self.cancel_selection();self.verified.setChecked(False)
        box,result,message=locate_current(self.frame,self.digits.value())
        voltage_box,voltage_result,voltage_message=locate_voltage(self.frame,self.digits.value())
        h,w=self.frame.shape[:2]
        self.auto_align.setChecked(True);self.color.setCurrentIndex(0)
        if box:
            x,y,xx,yy=map(int,box);self.cfg.roi=(x/w,y/h,(xx-x)/w,(yy-y)/h)
        else:self.cfg.roi=None
        self.cfg.cells=[]
        if voltage_box:
            vx,vy,vxx,vyy=map(int,voltage_box)
            self.preview.voltage_roi=(vx/w,vy/h,(vxx-vx)/w,(vyy-vy)/h)
            self.cfg.voltage_roi=self.preview.voltage_roi
        else:
            self.preview.voltage_roi=None;self.cfg.voltage_roi=None
        self.sync_config();self.save_settings()
        if box and voltage_box:self.log('已自动定位上排蓝色电压与下排红色电流；请核对两行实物读数。')
        else:
            missing=[]
            if not voltage_box:missing.append('电压：'+voltage_message)
            if not box:missing.append('电流：'+message)
            text='；'.join(missing)+'。可分别手动框选。'
            self.read_status.setText(text);self.log(text)

    def select_voltage_roi(self):
        if self.frame is None:return self.alert('先连接相机并获得画面。')
        self.calibrating='voltage';self.preview.mode='框住上排蓝色电压一行；Esc 取消'

    def settings_changed(self,*args):
        if hasattr(self,'verified'):self.verified.setChecked(False)
        if hasattr(self,'cfg'):
            if self.cfg.digits!=self.digits.value():self.cfg.cells=[]
            self.sync_config();self.save_settings()

    def sync_config(self):
        self.cfg.digits=self.digits.value();self.cfg.decimals=self.decimals.value();self.cfg.brightness=self.brightness.value()
        self.cfg.red_difference=self.red_diff.value();self.cfg.color=['auto','red','gray'][self.color.currentIndex()]
        self.cfg.auto_align=self.auto_align.isChecked()
        self.cfg.minimum=self.lower.value();self.cfg.maximum=self.upper.value();self.cfg.allow_blank=self.blanks.isChecked()
        self.preview.roi=self.cfg.roi;self.preview.voltage_roi=getattr(self.cfg,'voltage_roi',None);self.preview.boxes=digit_boxes(self.cfg);self.preview.update()

    def save_settings(self):
        if self.camera and self.camera.demo:return
        settings={'schema':3,'recognition':asdict(self.cfg),'voltage_roi':self.cfg.voltage_roi,'interval':self.interval.value(),'duration':self.duration.value(),'repeats':self.repeats.value(),
                  'camera_index':self.camera_index.currentData(),'backend':self.backend.currentIndex(),'resolution':self.resolution.currentIndex()}
        try:(self.data_dir/'settings.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
        except OSError as exc:self.log('配置保存失败：'+str(exc))

    def load_settings(self):
        path=self.data_dir/'settings.json'
        if not path.exists():return
        try:
            data=json.loads(path.read_text(encoding='utf-8'));c=RecognitionConfig(**data['recognition'])
            controls=[self.digits,self.decimals,self.brightness,self.red_diff,self.lower,self.upper,self.blanks,self.color,self.auto_align,self.interval,self.duration,self.repeats]
            for widget in controls:widget.blockSignals(True)
            self.digits.setValue(c.digits);self.decimals.setValue(c.decimals);self.brightness.setValue(c.brightness);self.red_diff.setValue(c.red_difference)
            self.lower.setValue(c.minimum);self.upper.setValue(c.maximum);self.blanks.setChecked(c.allow_blank)
            self.color.setCurrentIndex(['auto','red','gray'].index(c.color));self.auto_align.setChecked(c.auto_align)
            self.interval.setValue(data.get('interval',10));self.duration.setValue(data.get('duration',5))
            self.repeats.setValue(data.get('repeats',1) if data.get('schema')==2 else 1)
            self.camera_index.setCurrentIndex(max(0,self.camera_index.findData(data.get('camera_index',0))))
            self.backend.setCurrentIndex(data.get('backend',0));self.resolution.setCurrentIndex(data.get('resolution',0))
            for widget in controls:widget.blockSignals(False)
            c.voltage_roi=data.get('voltage_roi');self.cfg=c;self.sync_config()
        except Exception as exc:self.log('旧配置未能载入，使用默认值：'+str(exc))

    def toggle_camera(self):
        if self.camera:self.disconnect_camera()
        else:self.open_camera(False)

    def refresh_cameras(self):
        if self.camera:
            self.log('请先断开当前摄像头，再刷新设备列表。');return
        backend=self.backend.currentText();self.scan_btn.setEnabled(False);self.connect_btn.setEnabled(False);self.scan_btn.setText('正在扫描…')
        threading.Thread(target=scan_cameras,args=(backend,self.camera_scan_signals),daemon=True).start()

    def camera_scan_finished(self,indices):
        previous=self.camera_index.currentData()
        self.camera_index.blockSignals(True);self.camera_index.clear()
        for index in indices:self.camera_index.addItem(f'摄像头 {index}',index)
        if previous in indices:self.camera_index.setCurrentIndex(indices.index(previous))
        elif indices:self.camera_index.setCurrentIndex(0)
        else:self.camera_index.addItem('未检测到摄像头',0)
        self.camera_index.blockSignals(False);self.scan_btn.setEnabled(True);self.connect_btn.setEnabled(True);self.scan_btn.setText('刷新摄像头列表')
        self.log('摄像头列表已刷新：'+(', '.join(str(i) for i in indices) if indices else '未找到可用设备'))

    def start_demo(self):
        if not (self.camera and self.camera.demo):self.save_settings()
        self.disconnect_camera();self.open_camera(True)

    def open_camera(self,demo):
        self.frame=None;self.last_frame=0;self.last_shape=None;self.connected=False;self.verified.setChecked(False)
        self.last_drop_count=0
        self.reset_reading_views()
        self.preview.demo=demo;self.preview.pix=None;self.preview.update();self.last_reading=None
        self.camera=Camera(self.camera_index.currentData() or 0,self.backend.currentText(),(1280,720) if self.resolution.currentIndex()==0 else (640,480),demo)
        self.connect_btn.setText('断开相机');self.connection.setText('●  正在连接…')
        if demo:
            self.digits.setValue(3);self.decimals.setValue(1);self.brightness.setValue(90);self.red_diff.setValue(35)
            self.color.setCurrentIndex(0);self.auto_align.setChecked(True);self.lower.setValue(0);self.upper.setValue(999999)
            _,self.cfg.voltage_roi,self.cfg.roi=render_monitor_display();self.cfg.cells=[];self.sync_config()
            self.notice.setText('模拟演示 · 所有数值均为生成的测试信号，记录中会标记“模拟”')
        else:self.notice.setText('摄像头模式 · 上排蓝色为电压 / 下排红色为电流；先核对实物读数再测量')

    def disconnect_camera(self):
        self.stop_measure()
        was_demo=bool(self.camera and self.camera.demo)
        if self.camera:self.camera.stop();self.retiring.append(self.camera);self.camera=None
        self.connected=False;self.last_frame=0;self.current.setText('— A');self.last_reading=None
        self.connect_btn.setText('连接相机');self.connection.setText('●  相机未连接')
        self.preview.mode='';self.calibrating=None
        if was_demo:self.load_settings()

    def select_roi(self):
        if self.frame is None:return self.alert('先连接相机并获得画面。')
        self.calibrating='roi';self.preview.mode='框住完整电流一行即可，可留少量边缘；Esc 取消'
        self.auto_align.setChecked(True);self.color.setCurrentIndex(0)

    def select_voltage_roi(self):
        if self.frame is None:return self.alert('先连接相机并取得画面。')
        self.calibrating='voltage';self.preview.mode='框住上排蓝色电压一行；Esc 取消'

    def select_cells(self):
        if self.frame is None:return self.alert('先连接相机并获得画面。')
        self.auto_align.setChecked(False)
        if self.color.currentIndex()==0:self.color.setCurrentIndex(2);self.brightness.setValue(140)
        self.calibrating=[];self.preview.mode='请框选第 1 位的完整七段位置（从左到右）；Esc 取消'

    def on_selected(self,box):
        if self.engine.running:return
        if self.calibrating=='roi':
            self.cfg.roi=box;self.cfg.cells=[];self.calibrating=None;self.preview.mode=''
            self.log('已框选下排电流整行。')
        elif self.calibrating=='voltage':
            self.cfg.voltage_roi=box;self.calibrating=None;self.preview.mode=''
            self.log('已框选上排电压整行。')
        elif isinstance(self.calibrating,list):
            self.calibrating.append(box)
            if len(self.calibrating)==self.digits.value():
                self.cfg.cells=self.calibrating.copy();xs=[b[0] for b in self.cfg.cells]
                if xs!=sorted(xs):self.alert('请从左到右框选数字。');self.cfg.cells=[]
                else:
                    x=min(b[0] for b in self.cfg.cells);y=min(b[1] for b in self.cfg.cells)
                    self.cfg.roi=(x,y,max(b[0]+b[2] for b in self.cfg.cells)-x,max(b[1]+b[3] for b in self.cfg.cells)-y)
                self.calibrating=None;self.preview.mode='';self.log('逐位框选完成。请核对实时识别和实物读数。')
            else:self.preview.mode=f'请框选第 {len(self.calibrating)+1} 位的完整七段位置'
        self.verified.setChecked(False);self.sync_config();self.save_settings()

    def alert(self,text):QMessageBox.information(self,'电流监测',text)

    def start_measure(self):
        if not self.connected or self.frame is None:return self.alert('先连接相机或启动模拟演示。')
        if not self.cfg.roi:return self.alert('请先定位或框选下排红色电流。')
        if not self.cfg.voltage_roi:return self.alert('请先定位或框选上排蓝色电压。')
        if self.calibrating is not None:return self.alert('请先完成框选。')
        if self.cfg.decimals>=self.cfg.digits:return self.alert('总位数必须大于小数位数。')
        if self.cfg.minimum>=self.cfg.maximum:return self.alert('最大电流必须大于最小电流。')
        if not self.last_reading or self.last_reading.value is None:return self.alert('当前尚无有效读数，请先调整框选与阈值。')
        if not self.last_voltage_reading or self.last_voltage_reading.value is None:return self.alert('当前尚无有效电压读数，请调整上排电压框选。')
        if not self.camera.demo and not self.verified.isChecked():return self.alert('请先核对上排电压（V）、下排电流（A）及小数点，再勾选核对项。')
        self.sync_config();self.save_settings()
        self.engine=MeasurementEngine(duration=self.duration.value(),interval=self.interval.value(),repeats=self.repeats.value())
        self.engine.start(time.monotonic(),source='模拟' if self.camera.demo else f'相机 {self.camera.index}/{self.camera.backend}')
        self.set_measuring(True);self.log(f'开始：连续采集 {self.duration.value()} 秒，结束后等待 {self.interval.value()} 秒；记录区间内最大值，确认帧数 {self.repeats.value()}。')

    def set_measuring(self,active):
        for widget in self.lock_controls:widget.setEnabled(not active)
        self.start_btn.setEnabled(not active);self.stop_btn.setEnabled(active)

    def stop_measure(self):
        if self.engine.running:
            self.persist(self.engine.stop(time.monotonic()));self.log('测量已停止。未完成的窗口会单独标记。')
        if hasattr(self,'stop_btn'):self.set_measuring(False)

    def persist(self,completed):
        for record,evidence in completed:
            try:
                record.config=json.dumps({**asdict(self.cfg),'measurement':{'duration':self.duration.value(),'interval':self.interval.value(),'repeats':self.repeats.value()}},ensure_ascii=False)
                if evidence is not None:
                    filename=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+f'_{record.number}.png'
                    target=self.data_dir/'evidence'/filename
                    ok,encoded=cv2.imencode('.png',evidence)
                    if not ok:raise OSError('证据图片编码失败')
                    encoded.tofile(str(target));record.evidence=str(Path('evidence')/filename)
                self.store.add(record);self.store.export(self.data_dir/'全部测量记录.csv')
                value='—' if record.maximum is None else f'{record.maximum:.{self.cfg.decimals}f} A'
                at_peak='—' if record.voltage_at_max is None else f'{record.voltage_at_max:.{self.cfg.decimals}f} V'
                voltage_max='—' if record.maximum_voltage is None else f'{record.maximum_voltage:.{self.cfg.decimals}f} V'
                self.log(f'[{record.source}] 第 {record.number} 轮：最大电流 {value}，峰值时电压 {at_peak}，最高电压 {voltage_max}，{record.status}，有效 {record.valid_frames}/{record.total_frames} 帧')
            except Exception as exc:
                self.engine.running=False;self.set_measuring(False)
                self.alert('记录保存失败，自动测量已停止。请检查记录目录权限或磁盘空间：\n'+str(exc))
        if completed:self.refresh_records()

    def refresh_records(self):
        records=self.store.records();visible=list(reversed(records[-500:]));self.table.setRowCount(len(visible))
        for i,rec in enumerate(visible):
            decimals=json.loads(rec.get('config') or '{}').get('decimals',1)
            values=[rec['started'].replace('T',' '),'—' if rec['maximum'] is None else f"{rec['maximum']:.{decimals}f}",
                    '—' if rec.get('voltage_at_max') is None else f"{rec['voltage_at_max']:.{decimals}f}",
                    '—' if rec.get('maximum_voltage') is None else f"{rec['maximum_voltage']:.{decimals}f}",
                    f"{rec['valid_frames']} / {rec['total_frames']}",str(rec['confirmed_frames']),f"{rec['quality']:.2f}",rec['status'],rec['source']+(' · 查看证据' if rec['evidence'] else '')]
            for j,value in enumerate(values):
                item=QTableWidgetItem(value);item.setData(Qt.UserRole,rec.get('evidence',''));self.table.setItem(i,j,item)
        self.count_label.setText(str(len(records)))

    def open_evidence(self,row,column):
        target=self.table.item(row,column).data(Qt.UserRole)
        if target:QDesktopServices.openUrl(QUrl.fromLocalFile(str((self.data_dir/target).resolve())))

    def export_csv(self):
        filename,_=QFileDialog.getSaveFileName(self,'导出测量记录',str(self.data_dir/'测量记录.csv'),'CSV 文件 (*.csv)')
        if filename:
            try:self.store.export(filename);self.log('已导出：'+filename)
            except OSError as exc:self.alert('导出失败：'+str(exc))

    def process_packet(self,timestamp,frame,display=True):
        if self.last_shape and frame.shape!=self.last_shape:
            self.stop_measure();self.cfg.roi=None;self.cfg.cells=[];self.verified.setChecked(False);self.log('分辨率发生变化，请重新框选。')
        self.last_shape=frame.shape;self.last_frame=timestamp
        reading=recognize(frame,self.cfg);self.last_reading=reading
        # Advance by CAPTURE timestamp before consuming the frame. This preserves
        # a last-frame peak and includes the first frame of the next window.
        self.persist(self.engine.advance(timestamp))
        voltage_reading=recognize(frame,RecognitionConfig(roi=self.cfg.voltage_roi,digits=self.cfg.digits,decimals=self.cfg.decimals,
            minimum=0,maximum=999999,color='blue',auto_align=True)) if self.cfg.voltage_roi else None
        self.last_voltage_reading=voltage_reading
        self.engine.observe(timestamp,reading,reading.crop,voltage_reading)
        if getattr(self,'diagnostic_dir',None) and time.monotonic()<self.diagnostic_until:
            with (self.diagnostic_dir/'readings.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'time':timestamp,'value':reading.value,'quality':reading.confidence,'reason':reading.reason},ensure_ascii=False)+'\n')
        if not display:return
        self.frame=frame
        if self.calibrating is None:self.preview.set_frame(frame)
        self.current.setText(f'{reading.value:.{self.cfg.decimals}f} A' if reading.value is not None else '— A')
        self.voltage.setText(f'{voltage_reading.value:.{self.cfg.decimals}f} V' if voltage_reading and voltage_reading.value is not None else '— V')
        voltage_status=voltage_reading.reason if voltage_reading else '未定位上排电压'
        self.read_status.setText(f'电压：{voltage_status}    |    电流：{reading.reason}    |    电流分段质量 {reading.confidence:.2f}    |    画面 {frame.shape[1]}×{frame.shape[0]}')
        if reading.crop is not None and reading.crop.size:
            self.crop_view.setPixmap(image_pixmap(reading.crop).scaled(self.crop_view.size(),Qt.KeepAspectRatio,Qt.SmoothTransformation))
            if reading.mask is not None and reading.mask.size:
                mask=cv2.cvtColor(reading.mask,cv2.COLOR_GRAY2BGR)
                self.mask_view.setPixmap(image_pixmap(mask).scaled(self.mask_view.size(),Qt.KeepAspectRatio))
            else:self.mask_view.clear()

    def tick(self):
        now=time.monotonic()
        self.retiring=[c for c in self.retiring if c.thread.is_alive()]
        if self.camera:
            while True:
                try:event,text=self.camera.events.get_nowait()
                except queue.Empty:break
                if event=='connected':self.connected=True;self.connection.setText('●  '+text);self.log('已连接：'+text)
                elif event=='error':
                    self.log(text);self.connection.setText('●  相机连接失败');self.disconnect_camera();self.read_status.setText(text);break
            if self.camera:
                dropped=self.camera.dropped_frames
                if dropped>self.last_drop_count:
                    self.engine.mark_dropped(dropped-self.last_drop_count)
                    self.log('采集缓冲发生丢帧，本轮结果需要复核。');self.last_drop_count=dropped
                packets=self.camera.pending()
                for index,(timestamp,frame) in enumerate(packets):
                    self.process_packet(timestamp,frame,display=index==len(packets)-1)
        # Do not finalize ahead of frames that are still waiting to be analyzed.
        now=time.monotonic()
        if self.camera is None or self.camera.frames.empty():self.persist(self.engine.advance(now))
        if self.connected and self.last_frame and now-self.last_frame>2:
            self.current.setText('— A');self.voltage.setText('— V');self.last_reading=None;self.last_voltage_reading=None;self.read_status.setText('超过 2 秒没有新画面，本轮记录将标记中断。')
            self.stop_measure();self.connected=False;self.connection.setText('●  相机无新画面，请重新连接')
        phase,remaining=self.engine.phase(now)
        self.phase_label.setText(f'{phase} {remaining:.1f}s' if self.engine.running else phase)
        w=self.engine.window
        self.peak.setText(f"{w['max']:.{self.cfg.decimals}f} A" if w and w['max'] is not None else '— A')
        self.progress.setValue(round(1000*(1-remaining/(self.engine.duration if w else max(.001,self.engine.period-self.engine.duration)))) if self.engine.running else 0)

    def closeEvent(self,event):
        self.timer.stop();self.disconnect_camera()
        # Demo calibration is not saved over a real camera calibration at shutdown.
        self.store.close();event.accept()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--demo',action='store_true');parser.add_argument('--data-dir')
    parser.add_argument('--connect',action='store_true',help='Open the selected camera and try one automatic localization')
    parser.add_argument('--diagnostic-output',help='Local diagnostic output for the first 120 seconds')
    parser.add_argument('--smoke-test',metavar='SCREENSHOT');parser.add_argument('--ui-screenshot',metavar='PNG',help='Capture the running Qt window with an explicitly simulated signal; no camera is opened');args=parser.parse_args()
    app=QApplication(sys.argv)
    # Offscreen/remote Windows sessions may not enumerate system fonts. Load the
    # installed Windows fonts explicitly; no proprietary font is redistributed.
    fonts=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    for filename in ('msyh.ttc','msyhbd.ttc','segoeui.ttf'):
        if (fonts/filename).exists():QFontDatabase.addApplicationFont(str(fonts/filename))
    app.setStyle('Fusion');app.setStyleSheet(STYLE)
    app.setFont(QFont('Microsoft YaHei UI',10))
    try:instance_lock=single_instance(app,args.data_dir)
    except (RuntimeError,OSError) as exc:
        QMessageBox.warning(None,'电流监测',str(exc));return 1
    if instance_lock is None:
        if not args.smoke_test:QMessageBox.information(None,'电流监测','软件已经在运行。请切换到已有窗口，避免同时打开多个相机预览。')
        return 0
    try:window=MainWindow(args.data_dir,not bool(args.ui_screenshot))
    except Exception as exc:
        QMessageBox.critical(None,'启动失败','无法打开记录目录，请将软件解压到有写入权限的文件夹。\n'+str(exc));return 1
    def exception_hook(kind,value,tb):
        text=''.join(traceback.format_exception(kind,value,tb));window.log(text)
        (window.data_dir/'error.log').write_text(text,encoding='utf-8')
        window.stop_measure()
    sys.excepthook=exception_hook;window.show()
    if args.diagnostic_output:
        window.diagnostic_dir=Path(args.diagnostic_output);window.diagnostic_dir.mkdir(parents=True,exist_ok=True)
        window.diagnostic_until=time.monotonic()+120
        diagnostic_timer=QTimer(window)
        def report_diagnostic():
            if time.monotonic()>window.diagnostic_until:diagnostic_timer.stop();return
            reading=window.last_reading
            status={'value':reading.value if reading else None,'reason':reading.reason if reading else '',
                    'measuring':window.engine.running,'roi':window.cfg.roi,'connected':window.connected}
            (window.diagnostic_dir/'status.json').write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8')
            if window.frame is not None:
                _,encoded=cv2.imencode('.png',window.frame);encoded.tofile(str(window.diagnostic_dir/'last-frame.png'))
        diagnostic_timer.timeout.connect(report_diagnostic);diagnostic_timer.start(1000)
    if args.demo or args.smoke_test:QTimer.singleShot(100,window.start_demo)
    elif args.connect:
        QTimer.singleShot(100,lambda:window.open_camera(False))
        QTimer.singleShot(2200,lambda:window.auto_locate() if window.frame is not None else None)
    if args.smoke_test:
        def begin():window.start_measure()
        def finish():
            window.grab().save(args.smoke_test)
            records=window.store.records()
            (window.data_dir/'smoke_result.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
            window.close();app.quit()
        QTimer.singleShot(900,begin);QTimer.singleShot(11900,finish)
    if args.ui_screenshot:
        def capture_ui():
            from recognition import render_monitor_display
            frame,voltage_roi,current_roi=render_monitor_display(voltage=12.6,current=2.5,noise=0)
            window.camera_index.clear();window.camera_index.addItem('offscreen 模拟画面')
            window.connection.setText('●  模拟')
            window.notice.setText('通用摄像头 · 上排蓝色为电压 / 下排红色为电流 · 当前画面为模拟信号')
            window.preview.demo=True;window.cfg.roi=current_roi;window.cfg.voltage_roi=voltage_roi
            window.cfg.color='auto';window.cfg.auto_align=True;window.sync_config()
            window.process_packet(time.monotonic(),frame)
            window.peak.setText('— A');window.phase_label.setText('演示 / 未测量')
            window.log('offscreen 界面截图：PyQt5 实际窗口；识别画面与数值为软件模拟数据，不来自实体仪表。')
            if not window.grab().save(args.ui_screenshot):raise OSError('无法保存 Qt 界面截图。')
            window.close();app.quit()
        QTimer.singleShot(250,capture_ui)
    return app.exec_()

if __name__=='__main__':sys.exit(main())
