import numpy as np, wave, sys
def rd(p):
    d=open(p,'rb').read(); i=12; fmt=sr=None
    while i<len(d):
        cid=d[i:i+4]; n=int.from_bytes(d[i+4:i+8],'little'); body=d[i+8:i+8+n]
        if cid==b'fmt ': fmt=int.from_bytes(body[0:2],'little'); sr=int.from_bytes(body[4:8],'little'); bits=int.from_bytes(body[14:16],'little')
        if cid==b'data': x=np.frombuffer(body,dtype=np.float32 if fmt==3 else np.int16).astype(np.float64); return (x if fmt==3 else x/32768), sr
        i+=8+n+(n&1)
def mel(x,sr=24000,nfft=1024,hop=256,nm=80):
    win=np.hanning(nfft); fr=[x[i:i+nfft]*win for i in range(0,len(x)-nfft,hop)]
    S=np.abs(np.fft.rfft(np.array(fr),axis=1))**2
    f=np.linspace(0,sr/2,nfft//2+1); m=lambda h:2595*np.log10(1+h/700); im=lambda m_:700*(10**(m_/2595)-1)
    pts=im(np.linspace(m(0),m(sr/2),nm+2)); fb=np.zeros((nm,len(f)))
    for i in range(nm):
        l,c,r=pts[i:i+3]; fb[i]=np.clip(np.minimum((f-l)/(c-l),(r-f)/(r-c)),0,None)
    return np.log10(S@fb.T+1e-9)
a,sr=rd(sys.argv[1]); b,_=rd(sys.argv[2]); sec=float(sys.argv[3]) if len(sys.argv)>3 else None
print(f"{sys.argv[1]}: {len(a)/sr:.3f}s  {sys.argv[2]}: {len(b)/sr:.3f}s  sr {sr}")
n=min(len(a),len(b)) if sec is None else int(sec*sr)
A,B=a[:n],b[:n]; print(f"window {n/sr:.2f}s: waveform corr {np.corrcoef(A,B)[0,1]:.4f}, max|diff| {np.abs(A-B).max():.4g}")
MA,MB=mel(A),mel(B); cs=np.sum(MA*MB,1)/np.linalg.norm(MA,axis=1)/np.linalg.norm(MB,axis=1)
print(f"log-mel: mean frame cosine {cs.mean():.4f}, mean |dB diff| {10*np.abs(MA-MB).mean():.3f} dB, corr {np.corrcoef(MA.ravel(),MB.ravel())[0,1]:.4f}")
