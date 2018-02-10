from sympy import symbols, conjugate, derive_by_array, pi, exp
S, l, m, n = symbols('S l m n', real=True)
u, v, w, wt = symbols('u v w wt', real=True)
Vobs = symbols('Vobs', complex=True)
#n = sqrt(1-l*l - m*m) - 1
phase = - 2 * pi * (u*l+v*m)
V = S * exp(- 1j * phase)
res = Vobs - V
J=res*wt*conjugate(res)
axes = [S, l, m]
grad = derive_by_array(J, axes)
hess = derive_by_array(grad, axes)

print("J = %s" % J)
print("grad = %s" % grad)
print("hess = %s" % hess)
