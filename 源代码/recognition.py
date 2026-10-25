"""Offline seven-segment recognition. No cloud service or OCR model download."""
from dataclasses import dataclass, field
import cv2
import numpy as np

# a, b, c, d, e, f, g; clockwise from top, then middle.
PATTERNS = {
    '0': (1,1,1,1,1,1,0), '1': (0,1,1,0,0,0,0),
    '2': (1,1,0,1,1,0,1), '3': (1,1,1,1,0,0,1),
    '4': (0,1,1,0,0,1,1), '5': (1,0,1,1,0,1,1),
    '6': (1,0,1,1,1,1,1), '7': (1,1,1,0,0,0,0),
    '8': (1,1,1,1,1,1,1), '9': (1,1,1,1,0,1,1),
}
REVERSE = {v:k for k,v in PATTERNS.items()}
SEGMENTS = [(0.26,.04,.74,.14),(.76,.18,.94,.42),(.76,.58,.94,.82),
            (.26,.86,.74,.96),(.06,.58,.24,.82),(.06,.18,.24,.42),
            (.26,.45,.74,.55)]

@dataclass
class RecognitionConfig:
    roi: tuple | None = None                 # normalized x,y,w,h
    cells: list = field(default_factory=list) # optional individual normalized boxes
    digits: int = 3
    decimals: int = 1
    brightness: int = 90
    red_difference: int = 35
    color: str = 'red'
    on_threshold: float = .30
    off_threshold: float = .15
    allow_blank: bool = True
    minimum: float = 0.0
    maximum: float = 100.0
    confidence: float = .65
    auto_align: bool = False
    row: str = 'current'
    voltage_roi: tuple | None = None

@dataclass
class Reading:
    value: float | None
    text: str
    confidence: float
    reason: str
    mask: np.ndarray | None = None
    crop: np.ndarray | None = None
    occupancies: list = field(default_factory=list)

def box_pixels(box, shape):
    h,w = shape[:2]
    x,y,bw,bh = box
    x1,y1 = max(0,round(x*w)),max(0,round(y*h))
    x2,y2 = min(w,round((x+bw)*w)),min(h,round((y+bh)*h))
    return x1,y1,x2,y2

def digit_boxes(cfg):
    if cfg.cells:
        return cfg.cells
    if cfg.auto_align:return []
    if cfg.roi is None:
        return []
    x,y,w,h = cfg.roi
    return [(x+i*w/cfg.digits,y,w/cfg.digits,h) for i in range(cfg.digits)]

def make_mask(frame, cfg):
    b,g,r = cv2.split(frame)
    if cfg.color == 'red':
        # Int16 avoids unsigned subtraction wraparound.
        difference = r.astype(np.int16)-np.maximum(g,b).astype(np.int16)
        active = (r >= cfg.brightness) & (difference >= cfg.red_difference)
    else:
        active = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) >= cfg.brightness
    return active.astype(np.uint8)*255

def recognize(frame, cfg):
    if cfg.auto_align:
        return recognize_smart(frame,cfg)
    if cfg.cells and len(cfg.cells)!=cfg.digits:
        return Reading(None,'',0,f'还剩 {len(cfg.cells)} 个单字框，请重新框选或点击自动定位')
    boxes = digit_boxes(cfg)
    if not boxes:
        return Reading(None,'',0,'请先框选电流数字区域')
    if cfg.digits <= cfg.decimals:
        return Reading(None,'',0,'总位数必须大于小数位数')
    mask = make_mask(frame,cfg)
    if cfg.roi:
        x1,y1,x2,y2 = box_pixels(cfg.roi,frame.shape)
    else:
        pixels=[box_pixels(b,frame.shape) for b in boxes]
        x1,y1=min(p[0] for p in pixels),min(p[1] for p in pixels)
        x2,y2=max(p[2] for p in pixels),max(p[3] for p in pixels)
    crop=frame[y1:y2,x1:x2].copy()
    crop_mask=mask[y1:y2,x1:x2].copy()
    chars=[]; scores=[]; diagnostics=[]; has_digit=False
    for index, box in enumerate(boxes):
        a,b,c,d = box_pixels(box,frame.shape)
        if c-a < 12 or d-b < 24:
            return Reading(None,'',0,'数字太小：请靠近或提高分辨率',crop_mask,crop)
        cell=cv2.resize(mask[b:d,a:c],(100,160),interpolation=cv2.INTER_NEAREST)
        occ=[float(np.mean(cell[round(v*160):round(z*160),round(u*100):round(t*100)]))/255
             for u,v,t,z in SEGMENTS]
        diagnostics.append(occ)
        bits=tuple(int(v>=cfg.on_threshold) for v in occ)
        ambiguous=any(cfg.off_threshold<v<cfg.on_threshold for v in occ)
        if not any(bits) and cfg.allow_blank and not has_digit and index < cfg.digits-cfg.decimals-1:
            chars.append('0'); scores.append(.7); continue
        char=REVERSE.get(bits)
        if char is None or ambiguous:
            return Reading(None,''.join(chars)+'?',0,f'第 {index+1} 位不清晰，请调整框选或亮度阈值',crop_mask,crop,diagnostics)
        has_digit=True
        # This is a segment separation quality score, not a statistical probability.
        score=min(min(1.,.65+(v-cfg.on_threshold)*.7) if bit else
                  min(1.,.65+(cfg.off_threshold-v)*2) for v,bit in zip(occ,bits))
        chars.append(char); scores.append(score)
    digits=''.join(chars)
    text=digits[:-cfg.decimals]+'.'+digits[-cfg.decimals:] if cfg.decimals else digits
    value=int(digits)/(10**cfg.decimals)
    quality=min(scores)
    if not cfg.minimum <= value <= cfg.maximum:
        return Reading(None,text,quality,'超出设定电流范围',crop_mask,crop,diagnostics)
    if quality < cfg.confidence:
        return Reading(None,text,quality,'识别质量低于阈值',crop_mask,crop,diagnostics)
    return Reading(value,text,quality,'有效',crop_mask,crop,diagnostics)

def recognize_smart(frame,cfg):
    from smart_display import robust_read_row,core_mask
    if cfg.roi is None:return Reading(None,'',0,'请先定位或框选目标数显区域')
    x,y,xx,yy=box_pixels(cfg.roi,frame.shape)
    crop=frame[y:yy,x:xx].copy()
    if not crop.size:return Reading(None,'',0,'识别区域为空，请重新定位')
    if not 0<=cfg.decimals<cfg.digits:return Reading(None,'',0,'总位数必须大于小数位数')
    # Changing LED bloom can reach a previously located rectangle's edge even
    # though the physical digit remains complete. Check luminous cores, not halo.
    support=core_mask(crop,cfg.color,cfg.brightness)
    edges=[support[:,:2],support[:,-2:],support[:2,:],support[-2:,:]]
    if any(np.count_nonzero(edge)>4 for edge in edges):
        return Reading(None,'',0,'数字被区域边缘截断，请扩大框或将显示屏移到画面中央',support,crop)
    mode='blue' if cfg.color=='blue' else 'red'
    result=robust_read_row(crop,cfg.digits,mode,cfg.brightness)
    if not result.text or '?' in result.text:
        return Reading(None,'',0,result.reason,result.mask,crop)
    value=int(result.text)/(10**cfg.decimals)
    text=result.text[:-cfg.decimals]+'.'+result.text[-cfg.decimals:] if cfg.decimals else result.text
    if not cfg.minimum<=value<=cfg.maximum:return Reading(None,text,result.quality,'超出设定电流范围',result.mask,crop)
    if result.quality<cfg.confidence:return Reading(None,text,result.quality,'识别质量不足，请改善画面',result.mask,crop)
    return Reading(value,text,result.quality,result.reason,result.mask,crop)

def render_display(value=12.3, digits=3, decimals=1, size=(960,540), noise=0, seed=1):
    """Synthetic fixture and clearly labelled demonstration feed."""
    w,h=size; image=np.full((h,w,3),22,dtype=np.uint8)
    scale=min(w/(digits*105+120),h/320)
    cw,ch=int(90*scale),int(160*scale)
    x0,y0=(w-digits*cw)//2,(h-ch)//2
    text=f'{round(value*10**decimals):0{digits}d}'[-digits:]
    for i,char in enumerate(text):
        for active,(x,y,r,b) in zip(PATTERNS[char],SEGMENTS):
            if active:
                # Extend beyond sampling zones, like real LED bars.
                p=(x0+i*cw+round((x-.025)*cw),y0+round((y-.018)*ch))
                q=(x0+i*cw+round((r+.025)*cw),y0+round((b+.018)*ch))
                cv2.rectangle(image,p,q,(30,38,235),-1)
        if decimals and i==digits-decimals-1:
            cv2.circle(image,(x0+(i+1)*cw-round(.02*cw),y0+round(.94*ch)),max(2,int(3*scale)),(30,38,235),-1)
    if noise:
        rng=np.random.default_rng(seed)
        image=np.clip(image.astype(np.int16)+rng.normal(0,noise,image.shape),0,255).astype(np.uint8)
    roi=(x0/w,y0/h,digits*cw/w,ch/h)
    return image,roi

def render_monitor_display(voltage=12.6,current=.9,digits=3,decimals=1,size=(960,540),noise=0,seed=1):
    """Generate a clearly synthetic two-row panel: blue volts above red amps."""
    w,h=size;row_h=max(80,h//2)
    # Keep the blue row clean in the demo stream: random chroma speckles can
    # mimic saturated blue pixels and correctly trigger the live edge guard.
    upper,_=render_display(voltage,digits,decimals,(w,row_h),0,seed)
    lower,_=render_display(current,digits,decimals,(w,row_h),noise,seed+1)
    top=np.full_like(upper,22);blue=upper[:,:,2].astype(np.int16)>upper[:,:,0].astype(np.int16)*1.35
    top[blue]=(255,155,45)
    frame=np.vstack((top,lower))
    _,vr=render_display(voltage,digits,decimals,(w,row_h),0,seed)
    _,cr=render_display(current,digits,decimals,(w,row_h),0,seed)
    # Leave a small halo margin so the same edge-truncation guard used for live
    # camera frames does not reject the generated demo glyphs.
    def padded(box,dy):
        x,y,w0,h0=box; px=.03; py=max(dy,.06)
        return (max(0,x-px),max(0,y-py),min(1-x+px,w0+2*px),min(1-y+py,h0+2*py))
    voltage_roi=padded((vr[0],vr[1]*.5,vr[2],vr[3]*.5),.025)
    current_roi=padded((cr[0],.5+cr[1]*.5,cr[2],cr[3]*.5),.025)
    return frame,voltage_roi,current_roi
