"""Geometry-based LED localization and reading; never substitutes an expected number."""
from dataclasses import dataclass
import cv2
import numpy as np

PATTERNS={
    (1,1,1,1,1,1,0):'0',(0,1,1,0,0,0,0):'1',
    (1,1,0,1,1,0,1):'2',(1,1,1,1,0,0,1):'3',
    (0,1,1,0,0,1,1):'4',(1,0,1,1,0,1,1):'5',
    (1,0,1,1,1,1,1):'6',(1,1,1,0,0,0,0):'7',
    (1,1,1,1,1,1,1):'8',(1,1,1,1,0,1,1):'9'}
ZONES=[(.27,.03,.73,.13),(.78,.19,.96,.40),(.78,.60,.96,.81),
       (.27,.87,.73,.97),(.04,.60,.22,.81),(.04,.19,.22,.40),(.24,.41,.76,.59)]

@dataclass
class SmartResult:
    text: str=''
    quality: float=0.
    reason: str=''
    mask: np.ndarray|None=None
    boxes: list|None=None
    slant: float=0.

def red_support(frame):
    b,g,r=cv2.split(frame)
    delta=r.astype(np.int16)-np.maximum(b,g).astype(np.int16)
    hue=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)[:,:,0]
    return ((r>95)&(delta>35)&((hue<12)|(hue>168))).astype(np.uint8)*255

def blue_support(frame):
    """Extract saturated blue LED strokes; white centers are recovered later."""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    hue,saturation,value=cv2.split(hsv)
    return ((hue>=88)&(hue<=142)&(saturation>=65)&(value>=55)).astype(np.uint8)*255

def locate_color_rows(frame,mode='red'):
    """Return candidate LED rows for a color without reading their numeric value."""
    h,w=frame.shape[:2]
    mask=blue_support(frame) if mode=='blue' else red_support(frame)
    # Join a row across the spaces between its digits, but not across separate rows.
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(7,round(w*.085))|1,max(3,round(h*.065))|1))
    joined=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(joined)
    candidates=[list(map(int,item)) for item in stats[1:] if item[3]>=max(12,h*.02) and item[2]>=2]
    # Narrow 1s must be joined with their row before applying aspect-ratio checks.
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(candidates):
            for j in range(i+1,len(candidates)):
                b=candidates[j]
                overlap=min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1])
                gap=max(a[0],b[0])-min(a[0]+a[2],b[0]+b[2])
                if overlap>min(a[3],b[3])*.45 and gap<max(a[3],b[3])*1.2:
                    x,y=min(a[0],b[0]),min(a[1],b[1])
                    candidates[i]=[x,y,max(a[0]+a[2],b[0]+b[2])-x,max(a[1]+a[3],b[1]+b[3])-y,a[4]+b[4]]
                    del candidates[j];changed=True;break
            if changed:break
    rows=[]
    for x,y,bw,bh,area in candidates:
        if bh<14 or bw<bh*1.1 or bw>bh*12 or area<bh*bw*.08:continue
        # Do not accept only a fragment that touches the sensor image boundary.
        clipped=x<=1 or y<=1 or x+bw>=w-1 or y+bh>=h-1
        pad=max(3,round(bh*.15))
        box=(max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad))
        rows.append((box,float(area),clipped))
    return sorted(rows,key=lambda item:-item[1])

def locate_red_rows(frame):
    """Return candidate red rows, without reading or inferring their numeric value."""
    return locate_color_rows(frame,'red')

def core_mask(frame, mode='auto', brightness=140):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    if mode=='auto':
        red=red_support(frame)
        # Bright LED cores can be completely white, with a red ring around them.
        # Fill red envelopes, then use luminance to keep only their bright strokes.
        contours,_=cv2.findContours(red,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        envelope=np.zeros_like(red)
        cv2.drawContours(envelope,contours,-1,255,cv2.FILLED)
        vicinity=cv2.dilate(envelope,np.ones((5,5),np.uint8))>0
        if np.count_nonzero(vicinity)<12:return np.zeros_like(gray)
        # White centers are retained if supported by red pixels nearby. Pure red
        # bloom is darker in luminance and is separated from the luminous core.
        samples=gray[vicinity]
        otsu,_=cv2.threshold(samples.reshape(-1,1),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        peak=float(np.percentile(samples,98))
        threshold=max(40,min(195,max(float(otsu)*1.1,peak*.60)))
        b,g,r=cv2.split(frame)
        red_or_white=r.astype(np.float32)>=np.maximum(b,g)*.95
        mask=((gray>=threshold)&vicinity&red_or_white).astype(np.uint8)*255
    elif mode=='blue':
        support=blue_support(frame)
        contours,_=cv2.findContours(support,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        envelope=np.zeros_like(support)
        cv2.drawContours(envelope,contours,-1,255,cv2.FILLED)
        vicinity=cv2.dilate(envelope,np.ones((5,5),np.uint8))>0
        if np.count_nonzero(vicinity)<12:return np.zeros_like(gray)
        samples=gray[vicinity]
        otsu,_=cv2.threshold(samples.reshape(-1,1),0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        peak=float(np.percentile(samples,98))
        threshold=max(35,min(195,max(float(otsu)*1.1,peak*.60)))
        b,g,r=cv2.split(frame)
        blue_or_white=b.astype(np.float32)>=np.maximum(g,r)*.95
        mask=((gray>=threshold)&vicinity&blue_or_white).astype(np.uint8)*255
    elif mode=='gray':mask=(gray>=brightness).astype(np.uint8)*255
    else:
        red=red_support(frame)
        mask=((gray>=max(40,brightness*.75))&(cv2.dilate(red,np.ones((3,3),np.uint8))>0)).astype(np.uint8)*255
    return mask

def remove_small(mask):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    result=np.zeros_like(mask)
    h=mask.shape[0]
    for i,(x,y,w,bh,area) in enumerate(stats[1:],1):
        if bh>=max(3,h*.12) and area>=max(3,h*h*.002):result[labels==i]=255
    return result

def estimate_slant(mask):
    h,w=mask.shape
    edges=cv2.Canny(mask,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(5,round(h*.12)),
                         minLineLength=max(6,round(h*.22)),maxLineGap=2)
    slopes=[]
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            dy=y2-y1
            if abs(dy)<max(6,h*.2):continue
            slope=(x2-x1)/dy
            if abs(slope)<.5:slopes.extend([slope]*max(1,abs(dy)//4))
    return float(np.median(slopes)) if slopes else 0.

def level_row(mask):
    h,w=mask.shape
    lines=cv2.HoughLinesP(cv2.Canny(mask,50,150),1,np.pi/180,max(6,round(h*.12)),
                         minLineLength=max(8,round(h*.3)),maxLineGap=2)
    angles=[]
    if lines is not None:
        for x,y,xx,yy in np.asarray(lines).reshape(-1,4):
            if abs(xx-x)<max(8,h*.3):continue
            slope=(yy-y)/(xx-x)
            if abs(slope)<.22:angles.extend([float(np.degrees(np.arctan(slope)))]*max(1,abs(xx-x)//4))
    angle=float(np.median(angles)) if angles else 0.
    if abs(angle)<.3:return mask
    matrix=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    return cv2.warpAffine(mask,matrix,(w,h),flags=cv2.INTER_NEAREST)

def split_columns(mask):
    h,w=mask.shape
    profile=np.count_nonzero(mask,axis=0)
    active=profile>=max(2,round(h*.055))
    # Bridge one-column gaps caused by anti-aliasing; retain actual digit spaces.
    active=cv2.morphologyEx(active.astype(np.uint8)[None,:],cv2.MORPH_CLOSE,np.ones((1,2),np.uint8))[0]>0
    intervals=[];start=None
    for i,v in enumerate(np.r_[active,False]):
        if v and start is None:start=i
        elif not v and start is not None:
            if i-start>=2:intervals.append((start,i))
            start=None
    merged=[]
    for left,right in intervals:
        if (merged and left-merged[-1][1]<=h*.14 and right-merged[-1][0]<=h*.67
                and min(right-left,merged[-1][1]-merged[-1][0])<h*.4):
            merged[-1]=(merged[-1][0],right)
        else:merged.append((left,right))
    return merged

def decode_digit(cell,on_threshold=.32,off_threshold=.16):
    ys,xs=np.nonzero(cell)
    if not len(xs):return None,0.,[]
    x1,x2=xs.min(),xs.max()+1;y1,y2=ys.min(),ys.max()+1
    # A decimal dot may touch the preceding digit's bottom-right stroke. Infer
    # its right boundary from the upper body so the dot cannot stretch the glyph.
    upper=xs[ys<y1+(y2-y1)*.82]
    if len(upper):x2=min(x2,upper.max()+1)
    width,height=x2-x1,y2-y1
    if height<12:return None,0.,[]
    if width/height<.44:
        # Never stretch a narrow 1 to a full-width digit: an italic stem plus its
        # decimal dot can otherwise acquire the normalized seven-segment shape 2.
        # Check the physical stroke width before any horizontal normalization.
        col=cell[y1:y2,x1:x2]
        spans=[]
        for line in col[:max(1,round(height*.82))]:
            positions=np.flatnonzero(line)
            if len(positions):spans.append(positions[-1]-positions[0]+1)
        thin_stem=bool(spans) and float(np.percentile(spans,90))<=max(3,height*.27)
        if thin_stem and np.mean(col[:height//2]>0)>.25 and np.mean(col[height//2:]>0)>.25:
            return '1',.86,[]
        if width/height<.28:return None,0.,[]
    fitted=cv2.resize(cell[y1:y2,x1:x2],(100,160),interpolation=cv2.INTER_NEAREST)
    occupancies=[float(np.mean(fitted[round(v*160):round(b*160),round(u*100):round(r*100)]>0)) for u,v,r,b in ZONES]
    bits=tuple(int(v>=on_threshold) for v in occupancies)
    char=PATTERNS.get(bits)
    # A gray zone is evidence of an incomplete/blurred segment, not a license to
    # choose the nearest digit (which could turn a noisy frame into a false max).
    if char is None or any(off_threshold<v<on_threshold for v in occupancies):return None,0.,occupancies
    quality=min(min(1.,.68+(v-on_threshold)*.5) if bit else min(1.,.68+(off_threshold-v)*1.5)
                for v,bit in zip(occupancies,bits))
    return char,quality,occupancies

def evaluate_alignment(work,digits,slant,on_threshold=.32,off_threshold=.16):
    h,w=work.shape;pad=round(abs(slant)*h)+3
    matrix=np.float32([[1,-slant,pad+slant*h/2],[0,1,0]])
    aligned=cv2.warpAffine(work,matrix,(w+2*pad,h),flags=cv2.INTER_NEAREST)
    aligned=remove_small(aligned)
    intervals=split_columns(aligned)
    # Drop decimal points by height. They never become numeric positions.
    parts=[]
    for left,right in intervals:
        yy,xx=np.nonzero(aligned[:,left:right])
        if len(yy) and yy.max()-yy.min()+1>=h*.45:parts.append((left,right))
    if len(parts)!=digits:
        return SmartResult(reason=f'定位到 {len(parts)} 位，设置为 {digits} 位；请让整行数字完整入镜',mask=aligned,slant=slant)
    chars=[];scores=[]
    for i,(left,right) in enumerate(parts):
        char,score,occ=decode_digit(aligned[:,left:right],on_threshold,off_threshold)
        if char is None:
            return SmartResult(text=''.join(chars)+'?',reason=f'第 {i+1} 位亮段不完整，请改善清晰度或曝光',mask=aligned,boxes=parts,slant=slant)
        chars.append(char);scores.append(score)
    return SmartResult(''.join(chars),min(scores),'有效 · 已自动校正',aligned,parts,slant)

def read_row(frame,digits,mode='auto',brightness=140,on_threshold=.32,off_threshold=.16):
    if min(frame.shape[:2])<16:return SmartResult(reason='数字区域太小，请靠近显示屏')
    binary=level_row(remove_small(core_mask(frame,mode,brightness)))
    ys,xs=np.nonzero(binary)
    if len(xs)<12:return SmartResult(reason='未提取到完整亮段，请调整拍摄或亮度',mask=binary)
    # Tighten vertically; the user's coarse rectangle may include substantial margin.
    top,bottom=max(0,int(ys.min())-1),min(binary.shape[0],int(ys.max())+2)
    work=binary[top:bottom]
    estimated=estimate_slant(work)
    hypotheses=sorted(set(round(float(np.clip(s,-.4,.4)),3) for s in
                          [estimated,estimated-.025,estimated+.025,estimated-.08,estimated+.08,-.28,-.14,0,.14,.28]))
    results=[evaluate_alignment(work,digits,s,on_threshold,off_threshold) for s in hypotheses]
    valid=[r for r in results if len(r.text)==digits and '?' not in r.text]
    alternatives={r.text for r in valid}
    if len(alternatives)>1:
        return SmartResult(reason='不同校正角度读数不一致，请改善拍摄清晰度',mask=valid[0].mask)
    if len(valid)>=2:
        return max(valid,key=lambda r:r.quality)
    # One geometric fit alone is insufficient evidence for a numeric maximum.
    best=max(results,key=lambda r:(len(r.text.replace('?','')),r.quality))
    if valid:best.reason='仅一个校正角度有效，请改善清晰度或稍微靠近'
    best.text='';best.quality=0.
    return best

def _recover_mask(mask,digits):
    mask=remove_small(mask);ys,xs=np.nonzero(mask)
    if len(xs)<12:return None
    top,bottom=max(0,int(ys.min())-1),min(mask.shape[0],int(ys.max())+2)
    work=mask[top:bottom];estimated=estimate_slant(work)
    hypotheses=sorted(set(round(float(np.clip(s,-.4,.4)),3) for s in
                          [estimated,estimated-.025,estimated+.025,estimated-.08,estimated+.08,-.28,-.14,0,.14,.28]))
    results=[evaluate_alignment(work,digits,s) for s in hypotheses]
    valid=[r for r in results if len(r.text)==digits and '?' not in r.text]
    texts={r.text for r in valid}
    if len(texts)==1 and len(valid)>=2:return max(valid,key=lambda r:r.quality)
    if len(texts)==1 and len(valid)==1:return valid[0]
    return None

def robust_read_row(frame,digits,mode='red',brightness=90):
    """Use a small threshold ensemble only when the primary color read rejects."""
    primary_mode='blue' if mode=='blue' else 'auto'
    primary=read_row(frame,digits,primary_mode,brightness)
    if primary.text and '?' not in primary.text:return primary
    alternatives=[]
    if mode=='blue':
        for threshold in (90,110):
            alternatives.append(read_row(frame,digits,'gray',threshold))
    else:
        for threshold in (95,110,125):
            alternatives.append(read_row(frame,digits,'red',threshold))
    # A tiny morphological close/dilate ensemble recovers anti-aliased LED
    # strokes without allowing arbitrary nearest-digit guesses.
    support=core_mask(frame,primary_mode,brightness)
    for operation,size in ((cv2.MORPH_CLOSE,2),(cv2.MORPH_DILATE,2)):
        merged=cv2.morphologyEx(support,operation,np.ones((size,size),np.uint8))
        recovered=_recover_mask(merged,digits)
        if recovered:alternatives.append(recovered)
    if mode!='blue':
        for threshold in (90,120):
            merged=remove_small(cv2.dilate(core_mask(frame,'auto',threshold),np.ones((2,2),np.uint8)))
            recovered=_recover_mask(merged,digits)
            if recovered:alternatives.append(recovered)
    accepted=[r for r in alternatives if len(r.text)==digits and '?' not in r.text and r.quality>=.68]
    by_text={}
    for result in accepted:by_text.setdefault(result.text,[]).append(result)
    if by_text:
        ranked=sorted(by_text.items(),key=lambda item:(len(item[1]),max(r.quality for r in item[1])),reverse=True)
        minimum_votes=2 if mode!='blue' else 2
        if len(ranked[0][1])>=minimum_votes and (len(ranked)==1 or len(ranked[0][1])>len(ranked[1][1])):
            chosen=max(ranked[0][1],key=lambda r:r.quality)
            chosen.reason='有效 · 多阈值复核' if len(ranked[0][1])>1 else '有效 · 单阈值复核'
            return chosen
    if len(by_text)>1:
        return SmartResult(reason='不同颜色阈值读数不一致，请改善拍摄清晰度',mask=primary.mask)
    return primary

def locate_current(frame,digits):
    return locate_row(frame,digits,'red')

def locate_voltage(frame,digits):
    # The monitor layout reserves the upper row for voltage.  Restricting the
    # semantic voltage locator to that half prevents a red current row that has
    # been color-swapped by a camera/channel conversion from becoming a false
    # voltage reading.
    h = frame.shape[0]
    upper = frame[:max(1, int(h * .58))]
    box, result, reason = locate_row(upper, digits, 'blue')
    if box is not None:
        x1, y1, x2, y2 = box
        return (x1, y1, x2, y2), result, reason
    return box, result, reason

def locate_row(frame,digits,mode='red'):
    valid=[];failures=[]
    for box,area,clipped in locate_color_rows(frame,mode):
        if clipped:
            failures.append(('蓝色' if mode=='blue' else '红色')+'数字靠近画面边缘，请将整行数字移到画面中央');continue
        x1,y1,x2,y2=box
        row=frame[y1:y2,x1:x2]
        result=robust_read_row(row,digits,mode)
        if result.text and len(result.text)==digits and '?' not in result.text:
            valid.append((box,result))
        else:failures.append(result.reason)
    if len(valid)==1:return valid[0][0],valid[0][1],''
    label='蓝色电压' if mode=='blue' else '红色电流'
    if len(valid)>1:return None,None,f'找到多行{label}数字，请框选目标行的大致区域'
    return None,None,failures[0] if failures else f'未找到{label}数字，请正对显示屏并让整行数字完整入镜'
