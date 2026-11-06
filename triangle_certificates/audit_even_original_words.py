"""Independent uniform audit: expand ORIGINAL sextic words as deterministic paths.
No imports of the proposed proof's code, Jacobian, band formula, or certificates.
Symbolic boundary classes cover every m>=9; exact rational matrices cover m=5..8.
"""
from collections import defaultdict
from itertools import permutations,combinations
from functools import lru_cache
import sympy as S,json,time
from flint import fmpq_mat
m=S.symbols('m',integer=True,positive=True);j=S.symbols('j',integer=True)
M=m-1

def pmul(a,b):
 out=defaultdict(int)
 for w,c in a.items():
  for v,d in b.items():out[w+v]+=c*d
 return {w:c for w,c in out.items() if c}
def pcomm(a,b):
 out=defaultdict(int,pmul(a,b))
 for w,c in pmul(b,a).items():out[w]-=c
 return {w:c for w,c in out.items() if c}
x=[{(i,):1} for i in range(3)];cc=[pcomm(x[0],x[1]),pcomm(x[0],x[2]),pcomm(x[1],x[2])]
WORDS=defaultdict(int)
for p in permutations(range(3)):
 sign=(-1)**sum(p[a]>p[b] for a in range(3) for b in range(a+1,3))
 for w,c in pmul(pmul(cc[p[0]],cc[p[1]]),cc[p[2]]).items():WORDS[w]+=sign*c
WORDS={w:c for w,c in WORDS.items() if c}
assert len(WORDS)==48
# Indices are (0,k)=k, (1,k)=m+k, or (2,k)=j+k. Boundary classes
# use distinct indices for m>=9; generic j lies in 5,...,m-4.
def norm(i):
 typ,k=i
 if typ==0 and k<=0:return (1,k)
 if typ==1 and k>0:return (0,k)
 return i
def shift(i,d):return norm((i[0],i[1]+d))
def expr(i):return [S.Integer(0),m,j][i[0]]+i[1]
def eq(i,typ,k):return i==(typ,k)

@lru_cache(None)
def step(kind,var,par,i):
 z=S.Integer(0);one=S.Integer(1);ii=expr(i)
 if kind=='base':
  if var==0:
   return (1-par,i,one) if par==1 else (1,shift(i,-1),z if i==(0,1) else one)
  if var==1:
   return (0,shift(i,1),one if i==(1,0) else ii) if par==1 else (1,i,ii-1)
  return (0,shift(i,1),z if i==(1,0) else ii) if par==1 else (1,i,z if i==(1,0) else ii)
 if var==0:return(par,i,z)
 if kind[0]=='D':
  typ=int(kind[1]);w=z
  if typ==0:
   if var==1:w=(m*int(i==(0,1))-M*int(i==(0,2))) if par==0 else M*int(i==(0,1))
   else:w=(M if par==0 else one)*int(i==(0,1))
  else:
   if (var,par) in [(1,0),(2,1)]:w=one*int(i==(0,2))
  return par,i,w
 typ=int(kind[1]);c=-S.Rational(2)/(m*M) if typ==0 else 1/m
 if var==2:return par,shift(i,1),one*int(par==typ and i==(0,1))
 u=ii*c;v=ii*c
 if i==(1,-1):u=-(m+1)*c
 if i==(1,0):u=-M*c;v=-m*c
 if i==(0,1):
  u=(2 if typ==0 else 1)+c
  if typ==0:v=1+c
 if typ==0 and i==(0,2):u=2*c-2
 return par,shift(i,1),u if par==0 else v

def path(w,marks,par,i):
 v=S.Integer(1)
 for k in range(5,-1,-1):
  par,i,c=step(marks.get(k,'base'),w[k],par,i)
  if c==0:return None,0
  v*=c
 return (par,i),v

def coeff(i,par,kinds):
 ans=defaultdict(lambda:S.Integer(0))
 for w,wc in WORDS.items():
  poss=[k for k,x in enumerate(w) if x!=0]
  if len(kinds)==1:placements=[{k:kinds[0]} for k in poss]
  elif kinds[0]==kinds[1]:placements=[{k:kinds[0],h:kinds[0]} for k,h in combinations(poss,2)]
  else:placements=[{k:kinds[0],h:kinds[1]} for k in poss for h in poss if h!=k]
  for marks in placements:
   target,v=path(w,marks,par,i)
   if v!=0:ans[target]+=wc*v
 return {k:S.factor(v) for k,v in ans.items() if S.factor(v)!=0}

def aval(i):return 1-m if i==(1,0) else S.Integer(1)
def pval(i):return S.Integer(1) if i==(1,0) else expr(i)
def cval(i):return -aval(shift(i,1))*pval(i)
def wt0(i):return 1/(cval(i)*aval(i)*aval(shift(i,1)))
def wt1(i):return 1/(cval(i)*cval(shift(i,1))*(aval(i)+aval(shift(i,2))-aval(shift(i,1))))

BOUND=[(0,k) for k in range(1,5)]+[(1,k) for k in range(-3,1)]+[(2,0)]
def symbolic():
 result={'method':'48 original associative sextic words, marked positions, symbolic path expansion','uniform_range':'m>=9; boundary classes 1..4,m-3..m and interior 5..m-4','kernel_checks':[],'cross':{},'self':{}}
 for d in ['D0','D1','E0','E1']:
  for par in [0,1]:
   for i in BOUND:
    a=coeff(i,par,[d])
    assert a=={},(d,par,i,a)
  result['kernel_checks'].append(d)
 print('UNIFORM KERNEL ALL FOUR: PASS',flush=True)
 for a in [0,1]:
  for b in [0,1]:
   terms=[]
   for i in BOUND:
    z=coeff(i,1,['D'+str(a),'E'+str(b)])
    assert set(z).issubset({(1,shift(i,1))}),(i,z)
    val=S.factor(z.get((1,shift(i,1)),0)*wt0(i))
    if val!=0:terms.append([str(expr(i)),str(val)])
   total=S.factor(sum(S.sympify(v,locals={'m':m,'j':j}) for _,v in terms))
   result['cross'][str(a)+str(b)]={'terms':terms,'sum':str(total)}
   print('C'+str(a)+str(b),terms,total,flush=True)
 for typ in ['D0','D1']:
  for i in BOUND:
   z=coeff(i,1,[typ,typ]);assert (1,shift(i,1)) not in z
 for a in [0,1]:
  for b in [a,1]:
   for i in BOUND:
    z=coeff(i,1,['E'+str(a),'E'+str(b)]);assert (1,shift(i,1)) not in z
 terms=[]
 for i in BOUND:
  z=coeff(i,1,['E1','E1'])
  assert set(z).issubset({(1,shift(i,2))}),(i,z)
  val=S.factor(z.get((1,shift(i,2)),0)*wt1(i))
  if val!=0:terms.append([str(expr(i)),str(val)])
 total=S.factor(sum(S.sympify(v,locals={'m':m,'j':j}) for _,v in terms))
 result['self']['E1']={'terms':terms,'sum':str(total)}
 assert S.factor(total+2/(m*(m+1)))==0
 print('Q1 E1',terms,total,flush=True)
 return result

def finite(mm):
 # Actual finite rational matrices; direct original word expansion with coefficient extraction.
 nn=2*mm;I=S.eye(mm);J=S.zeros(mm);Sh=S.zeros(mm);P=S.zeros(mm)
 for i in range(mm-1):J[i,i+1]=1;Sh[i+1,i]=i+1;P[i+1,i]=i+1
 P[0,mm-1]=1;O=S.zeros(mm)
 def block(ul,ur,ll,lr):return S.BlockMatrix([[ul,ur],[ll,lr]]).as_explicit()
 A=[block(O,I,J,O),block(O,P,P*J,O),block(O,Sh,J*Sh,O)]
 dirs=[]
 for d in ['D0','D1','E0','E1']:
  ds=[S.zeros(nn) for _ in range(3)]
  for var in [1,2]:
   for par in [0,1]:
    for jj in range(1,mm+1):
     # Build directions from the displayed formulas using exact finite indices.
     c=(-S.Rational(2,mm*(mm-1)) if d=='E0' else S.Rational(1,mm))
     if d[0]=='D':
      t=int(d[1]);v=0
      if t==0:
       if var==1:v=(mm*(jj==1)-(mm-1)*(jj==2)) if par==0 else (mm-1)*(jj==1)
       else:v=((mm-1) if par==0 else 1)*(jj==1)
      elif (var,par) in [(1,0),(2,1)]:v=int(jj==2)
      ds[var][par*mm+jj-1,par*mm+jj-1]=v
     else:
      t=int(d[1]);u=jj*c;v=jj*c
      if jj==mm-1:u=-(mm+1)*c
      if jj==mm:u=-(mm-1)*c;v=-mm*c
      if jj==1:
       u=(2 if t==0 else 1)+c
       if t==0:v=1+c
      if t==0 and jj==2:u=2*c-2
      val=(u if par==0 else v) if var==1 else int(par==t and jj==1)
      ds[var][par*mm+jj%mm,par*mm+jj-1]=val
  dirs.append(ds)
 def toq(a):return fmpq_mat([[str(a[i,k]) for k in range(nn)] for i in range(nn)])
 A=[toq(a) for a in A];dirs=[[toq(a) for a in d] for d in dirs]
 def original_coefficient(ds,degree):
  out=fmpq_mat(nn,nn)
  for w,wc in WORDS.items():
   pol=[fmpq_mat([[int(i==k) for k in range(nn)] for i in range(nn)])]+[fmpq_mat(nn,nn) for _ in range(degree)]
   for var in w:
    nxt=[]
    for k in range(degree+1):
     r=pol[k]*A[var]
     if k:r+=pol[k-1]*ds[var]
     nxt.append(r)
    pol=nxt
   out+=wc*pol[degree]
  return out
 def norms(F):
  q0=0;q1=0
  aa=[1]*(mm-1)+[1-mm]
  pp=list(range(1,mm))+[1];cc=[-aa[(z+1)%mm]*pp[z] for z in range(mm)]
  for z in range(mm):
   q0+=S.Rational(str(F[mm+(z+1)%mm,mm+z]))/(cc[z]*aa[z]*aa[(z+1)%mm])
   q1+=S.Rational(str(F[mm+(z+2)%mm,mm+z]))/(cc[z]*cc[(z+1)%mm]*(aa[z]+aa[(z+2)%mm]-aa[(z+1)%mm]))
  return S.simplify(q0),S.simplify(q1)
 qq=[]
 for d in dirs:
  assert original_coefficient(d,1)==fmpq_mat(nn,nn)
  qq.append(norms(original_coefficient(d,2)))
 C=S.zeros(2)
 for a in [0,1]:
  for b in [0,1]:
   C[a,b]=norms(original_coefficient([dirs[a][k]+dirs[2+b][k] for k in range(3)],2))[0]-qq[a][0]-qq[2+b][0]
 expected=S.Matrix([[-S.Rational((mm-2)*(mm*mm+mm+2),(mm-1)**2),S.Rational((mm-2)*(mm+1),mm-1)],[S.Rational(2*mm*(mm-2),(mm-1)**2),0]])
 assert C==expected,(mm,C,expected)
 assert qq[3][0]==0 and qq[3][1]==-S.Rational(2,mm*(mm+1))
 assert all(q[0]==0 for q in qq)
 print('FINITE',mm,'C',C.tolist(),'det',C.det(),'Q1',qq[3][1],flush=True)
 return {'m':mm,'all_four_in_original_kernel':True,'cross':[[str(z) for z in row] for row in C.tolist()],'cross_det':str(C.det()),'Q1_E1':str(qq[3][1])}
if __name__=='__main__':
 start=time.time();out=symbolic();out['finite_boundary_checks']=[finite(mm) for mm in range(5,9)];out['elapsed_seconds']=time.time()-start
