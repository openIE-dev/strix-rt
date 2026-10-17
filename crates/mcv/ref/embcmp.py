import sys, struct, math, array
a=array.array('f', open(sys.argv[1],'rb').read()); b=array.array('f', open(sys.argv[2],'rb').read()); d=2048; n=len(a)//d
cs=[]
for j in range(n):
    x=a[j*d:(j+1)*d]; y=b[j*d:(j+1)*d]
    ab=sum(p*q for p,q in zip(x,y)); aa=sum(p*p for p in x); bb=sum(q*q for q in y); cs.append((ab/math.sqrt(aa*bb),j))
cs.sort(); print("n",n,"mean",sum(c for c,_ in cs)/n,"worst",[(j,round(c,5)) for c,j in cs[:6]])
