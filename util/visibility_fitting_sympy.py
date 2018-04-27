from sympy import *
init_printing()

# Need the following to fold the conjugates. See
# https://stackoverflow.com/questions/48754975/simplification-of-derivative-of-square-using-sympy

from sympy.core.rules import Transform
fold_conjugates = Transform(lambda f: 2*re(f.args[0]),
                            lambda f: isinstance(f, Add) and len(f.args) == 2 and f.args[1] == f.args[0].conjugate())

fold_conjugates_2 = Transform(lambda f: f.args[0] + 2*re(f.args[1]),
                            lambda f: isinstance(f, Add) and len(f.args) == 3 and f.args[2] == f.args[1].conjugate())

S, l, m, n = symbols('S l m n', real=True)
u, v, w, wt = symbols('u v w wt', real=True)
Vobs, Vres0 = symbols('Vobs Vres', complex=True)
Vres = Vobs - S * exp(- I * 2 * pi * (u*l+v*m))
J=Vres*wt*conjugate(Vres)
axes = [S, l, m]
grad = derive_by_array(J, axes)
hess = derive_by_array(grad, axes)

print("J = ", J)

for axis in range(3):
    print("grad[%d] = %s" % (axis, grad[axis].subs(Vres, Vres0).conjugate().subs(Vres, Vres0).conjugate().xreplace(
        fold_conjugates)))

for axis2 in range(3):
    for axis1 in range(3):
        print("hess[%d,%d] = %s " % (axis2, axis1, hess[axis2, axis1].subs(Vres, Vres0).conjugate() \
                                     .subs(Vres, Vres0).xreplace(fold_conjugates).xreplace(fold_conjugates_2) \
                                     .factor()))