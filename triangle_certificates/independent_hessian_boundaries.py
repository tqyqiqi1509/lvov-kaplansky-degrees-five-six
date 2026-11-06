"""Exact small-boundary check by independent commutator Taylor formulas."""
from pathlib import Path
from flint import fmpq_mat
import sympy as s,json,time
OUT=Path(__file__).parent;data=json.loads((OUT/'odd_compact_certificate.json').read_text());m=s.symbols('m')
def comm(a,b):return a*b-b*a
def alt(a,b,c):return a*b*c-a*c*b-b*a*c+b*c*a+c*a*b-c*b*a
def diff(A,D):
 x,y,z=A;u,v,w=D
 return [comm(u,y)+comm(x,v),comm(u,z)+comm(x,w),comm(v,z)+comm(y,w)]
def derivative(A,D):
 x,y,z=A;a,b,c=comm(x,y),comm(x,z),comm(y,z);aa,bb,cc=diff(A,D)
 return alt(aa,b,c)+alt(a,bb,c)+alt(a,b,cc)
def mixed(A,D,E):
 x,y,z=A;a,b,c=comm(x,y),comm(x,z),comm(y,z);u,v,w=diff(A,D);uu,vv,ww=diff(A,E)
 dx,dy,dz=D;ex,ey,ez=E
 ab=comm(dx,ey)+comm(ex,dy);bb=comm(dx,ez)+comm(ex,dz);cb=comm(dy,ez)+comm(ey,dz)
 return alt(ab,b,c)+alt(a,bb,c)+alt(a,b,cb)+alt(u,vv,c)+alt(uu,v,c)+alt(u,b,ww)+alt(uu,b,w)+alt(a,v,ww)+alt(a,vv,w)
out=[];start=time.time()
for mm in range(5,14):
 n=2*mm+1;l=mm+1;M=mm*(mm+1)//2;A=[fmpq_mat(n,n) for _ in range(3)]
 for j in range(mm):
  A[0][j,l+j]=1;A[0][l+j,j+1]=1
  A[1][j+1,l+j]=(j+1)*(j+2)//2-M;A[1][l+j,j]=j*(j+1)//2-M
  A[2][j+1,l+j]=j+1;A[2][l+j,j]=j+1
 def num(x):return str(s.sympify(x).subs(m,mm))
 def direction(rows):
  D=[fmpq_mat(n,n) for _ in range(3)]
  for v,p,r,c,a in rows:D[v][int(num(r))-1+p*l,int(num(c))-1+p*l]=num(a)
  return D
 Ds=[direction(data['D0']),direction(data['D1'])]
 assert all(derivative(A,d)==fmpq_mat(n,n) for d in Ds)
 for label,ofs,rows in [('E',0,[(i,l+mm-2+i)for i in range(2)]+[(l+i,mm-1+i)for i in range(2)]),('F',-2,[(i,l+mm-4+i)for i in range(4)]+[(l+i,mm-3+i)for i in range(4)])]:
  ca=data[label+'A'];cb=data[label+'B']
  for k,r in enumerate(data[label+'co']):
   e=direction([r]);de=derivative(A,e);expected=fmpq_mat(n,n)
   for j,(ri,ci)in enumerate(rows):expected[ri,ci]=num(ca[j][k])
   assert de==expected,(mm,label,k,'derivative')
   for j,d in enumerate(Ds):
    value=mixed(A,d,e)[l,l+mm-1+ofs]
    assert value==s.Rational(num(cb[j][k])),(mm,label,k,j,'mixed',value,num(cb[j][k]))
 out.append({'m':mm,'n':n,'D_kernel':True,'all_E_F_derivative_coordinates':True,'all_E_F_mixed_quadratic_entries':True})
 print('PASS exact commutator boundary',mm,flush=True)
