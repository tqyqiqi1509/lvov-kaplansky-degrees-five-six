"""Independent uniform odd-Hessian audit using commutator Taylor algebra.
No author symbolic path/48-word implementation is imported.  Sparse vertices
are affine functions k*m+b, disjoint for every m>=14.  Every base multiplication
is a direct multiplication by the six specified band matrices X,Y,Z,A,B,C.
"""
from pathlib import Path
from functools import lru_cache
import sympy as s,json,itertools,time
m=s.symbols('m',integer=True,positive=True); M=m*(m+1)/2
OUT=Path(__file__).parent
DATA=json.loads((OUT/'odd_compact_certificate.json').read_text())

def clean(d):return {k:s.factor(v) for k,v in d.items() if s.expand(v)!=0}
def add(*ds):
 d={}
 for z in ds:
  for k,v in z.items():d[k]=d.get(k,0)+v
 return clean(d)
def neg(d):return {k:-v for k,v in d.items()}
def ix(v):return v[1]*m+v[2]
def shift(v,a,block=None):return (v[0] if block is None else block,v[1],v[2]+a)
def ge(a):
 a=s.Poly(s.expand(a),m); c=a.coeff_monomial(m); b=a.coeff_monomial(1)
 assert a.degree()<=1
 if c==0:return b>=0
 if c>0 and 14*c+b>=0:return True
 if c<0 and 14*c+b<0:return False
 raise AssertionError(('range changes beyond m14',a.as_expr()))
def valid(v):return ge(ix(v)-1) and ge(m+1-v[0]-ix(v))
def edge(v,c):return [(v,s.factor(c))] if valid(v) and s.expand(c)!=0 else []

@lru_cache(None)
def base_col(name,v):
 p,k,b=v;i=ix(v)
 if not valid(v):return []
 if name=='X':return edge((0,k,b),1) if p else edge((1,k,b-1),1)
 if name=='Y':return edge((0,k,b+1),i*(i+1)/2-M) if p else edge((1,k,b),(i-1)*i/2-M)
 if name=='Z':return edge((0,k,b+1),i) if p else edge((1,k,b),i)
 if name=='A':return edge(v,i) if p else (edge(v,-M) if (k,b)==(0,1) else [])
 if name=='B':return [] if p else edge(v,-m if (k,b)==(1,1) else 1)
 if name=='C':return edge((p,k,b+1),M-i*(i+1)/2 if p else i*i)
 raise ValueError(name)

@lru_cache(None)
def base_row(name,v):
 p,k,b=v;i=ix(v)
 if not valid(v):return []
 if name=='X':return edge((0,k,b+1),1) if p else edge((1,k,b),1)
 if name=='Y':return edge((0,k,b),(i-1)*i/2-M) if p else edge((1,k,b-1),(i-1)*i/2-M)
 if name=='Z':return edge((0,k,b),i) if p else edge((1,k,b-1),i-1)
 if name in ['A','B']:return base_col(name,v)
 if name=='C':return edge((p,k,b-1),M-(i-1)*i/2 if p else (i-1)**2)
 raise ValueError(name)

def mul(a,b):
 d={}
 if isinstance(a,str):
  assert not isinstance(b,str)
  for (r,c),v in b.items():
   for u,w in base_col(a,r):d[(u,c)]=d.get((u,c),0)+w*v
 elif isinstance(b,str):
  for (r,c),v in a.items():
   for u,w in base_row(b,c):d[(r,u)]=d.get((r,u),0)+v*w
 else:
  for (r,c),v in a.items():
   for (u,w),t in b.items():
    if c==u:d[(r,w)]=d.get((r,w),0)+v*t
 return clean(d)
def triple(a,b,c):
 return mul(a,mul(b,c)) if isinstance(a,str) and isinstance(b,str) else mul(mul(a,b),c)
def alt(a,b,c):
 return add(triple(a,b,c),neg(triple(a,c,b)),neg(triple(b,a,c)),triple(b,c,a),triple(c,a,b),neg(triple(c,b,a)))
def comm(a,b):return add(mul(a,b),neg(mul(b,a)))
def differentials(D):
 x,y,z=D
 return [add(comm(x,'Y'),comm('X',y)),add(comm(x,'Z'),comm('X',z)),add(comm(y,'Z'),comm('Y',z))]
def derivative(D):
 a,b,c=differentials(D)
 return add(alt(a,'B','C'),alt('A',b,'C'),alt('A','B',c))
def mixed(D,E):
 a,b,c=differentials(D);u,v,w=differentials(E)
 x,y,z=D;xx,yy,zz=E
 aa=add(comm(x,yy),comm(xx,y));bb=add(comm(x,zz),comm(xx,z));cc=add(comm(y,zz),comm(yy,z))
 return add(alt(aa,'B','C'),alt('A',bb,'C'),alt('A','B',cc),alt(a,v,'C'),alt(u,b,'C'),alt(a,'B',w),alt(u,'B',c),alt('A',b,w),alt('A',v,c))
def affine(text):
 z=s.Poly(s.sympify(text,locals={'m':m}),m)
 return (int(z.coeff_monomial(m)),int(z.coeff_monomial(1)))
def direction(rows):
 d=[{}, {}, {}]
 for var,block,r,c,val in rows:
  rr=(block,*affine(r));cc=(block,*affine(c));assert valid(rr) and valid(cc)
  d[var][(rr,cc)]=s.sympify(val,locals={'m':m})
 return d
def stated(name):return s.Matrix([[s.sympify(v,locals={'m':m})for v in row]for row in DATA[name]])
def assert_matrix(a,b):assert a.shape==b.shape and all(s.expand(v)==0 for v in a-b),(a,b)
def linear_matrix(ds):
 cs=[derivative(d)for d in ds];rr=sorted(set().union(*(set(c)for c in cs)))
 return rr,s.Matrix([[c.get(r,0)for c in cs]for r in rr])
def entry(z,col):return z.get(((1,0,1),(1,1,col)),0)

start=time.time();D=[direction(DATA['D0']),direction(DATA['D1'])]
assert all(derivative(d)=={}for d in D)
E=[direction([r])for r in DATA['Eco']];F=[direction([r])for r in DATA['Fco']]
er,ea=linear_matrix(E);fr,fa=linear_matrix(F)
assert_matrix(ea,stated('EA'));assert_matrix(fa,stated('FA'))
eb=s.Matrix([[entry(mixed(d,e),0)for e in E]for d in D])
fb=s.Matrix([[entry(mixed(d,f),-2)for f in F]for d in D])
assert_matrix(eb,stated('EB'));assert_matrix(fb,stated('FB'))
assert all(entry(mixed(d,e),0)==0 for d in D for e in D)
assert all(entry(mixed(d,e),0)==0 for d in E[1:]for e in E[1:])
assert all(entry(mixed(d,f),0)==0 for d in D for f in F)
em=s.factor(ea.col_join(eb)[:,1:7].det(method='domain-ge'))
fm=s.factor(fa.col_join(fb[0,:])[:,:9].det(method='domain-ge'))
assert s.expand(em-s.sympify(DATA['E_minors'][-1][1],locals={'m':m}))==0
assert s.expand(fm-s.sympify(DATA['Fminor0']['det'],locals={'m':m}))==0
P7=s.cancel(em/(m**4*(m-1)*(m+1)**2*(m*m+m+2)))
P13=s.cancel(fm/(m*m*(m**3-2*m*m-15*m-60)))
c7=s.Poly(s.expand(P7.subs(m,m+2)),m).all_coeffs();c13=s.Poly(s.expand(P13.subs(m,m+4)),m).all_coeffs()
assert all(x>0 for x in c7+c13)
R=m**3-2*m*m-15*m-60
assert all(R.subs(m,k)<0 for k in [4,5,6]) and all(x>0 for x in s.Poly(R.subs(m,m+7),m).all_coeffs())
out={'method':'Independent structured-band commutator Taylor algebra; no original author code imports','uniform_range':'m>=14; all affine vertex branches proven constant on this range','original_diagonal_kernel_verified':True,'EA_verified':True,'EB_verified':True,'FA_verified':True,'FB_verified':True,'all_unlisted_derivative_entries_zero':True,'Q0_DD_zero':True,'Q0_EE_zero':True,'Q0_DF_zero':True,'E_output_rows':str(er),'F_output_rows':str(fr),'E_determinant':str(em),'F_determinant':str(fm),'P7_shift2_positive_coefficients':list(map(str,c7)),'P13_shift4_positive_coefficients':list(map(str,c13)),'seconds':time.time()-start}
print(json.dumps(out,indent=2))
