#!/usr/bin/python

from numpy import *
from matplotlib.pyplot import *

set_printoptions(linewidth=200)

# -------------------------------------------------------------------------------------------------------------------- #
def gen_model_vis( Vm, g ):

    return array(matrix(g) * matrix(g).getH()) * Vm

# -------------------------------------------------------------------------------------------------------------------- #
# add entries for this visibility to the design matrix
def update_design_matrix( A, Marr, g, i, j ):

    gi = g[i]
    gj = g[j]
    M = Marr[i,j]
 
    # visibility indices (i.e. design matrix rows)
    kM = kx2
 
    # parameter indices (i.e. design matrix columns)
    ip = 2*i
    jp = 2*j
 
    #dfdeiRe
    A[kM,ip]     += +real(+conj(gj)*M)
    A[kM+1,ip]   += +imag(+conj(gj)*M)
 
    #dfdeiIm
    A[kM,ip+1]   += -imag(+conj(gj)*M)
    A[kM+1,ip+1] += +real(+conj(gj)*M)
 
    #dfdejRe
    A[kM,jp]     += +real(+gi*M)
    A[kM+1,jp]   += +imag(+gi*M)
 
    #dfdejIm
    A[kM,jp+1]   += +imag(+gi*M)
    A[kM+1,jp+1] += -real(+gi*M)

# -------------------------------------------------------------------------------------------------------------------- #
# add entries for this visibility to the normal matrix
def update_normal_matrix( AA, MM, g, i, j ):
 
    # parameter indices (i.e. design matrix columns)
    ip = 2*i
    jp = 2*j

    ggMM = g[i]*g[j] * MM[i,j]

    # set diagonal terms for this visibility
    AA[ip,ip]     += abs(g[j,0])**2 * abs(MM[i,j])
    AA[ip+1,ip+1] += abs(g[j,0])**2 * abs(MM[i,j])
    AA[jp,jp]     += abs(g[i,0])**2 * abs(MM[i,j])
    AA[jp+1,jp+1] += abs(g[i,0])**2 * abs(MM[i,j])

    # set off-diagonal terms for this visibility
    AA[ip,jp]     +=  real(ggMM)
    AA[ip,jp+1]   +=  imag(ggMM)
    AA[ip+1,jp]   +=  imag(ggMM)
    AA[ip+1,jp+1] += -real(ggMM)
 
    AA[jp,ip]     +=  real(ggMM)
    AA[jp,ip+1]   +=  imag(ggMM)
    AA[jp+1,ip]   +=  imag(ggMM)
    AA[jp+1,ip+1] += -real(ggMM)

# -------------------------------------------------------------------------------------------------------------------- #
# add entries for this visibility to the data vector
def update_data_vector( Av, MV, MM, g, i, j ):

    gi = g[i]
    gj = g[j]

    # parameter indices (i.e. design matrix columns)
    ip = 2*i
    jp = 2*j

    # R = V - gi*conj(gj)*M
    #Av[ip]   +=  real(conj(gj)*M)*real(R) + imag(conj(gj)*M)*imag(R)
    #Av[ip+1] += -imag(conj(gj)*M)*real(R) + real(conj(gj)*M)*imag(R)
    #Av[jp]   +=  real(gi*M)*real(R) + imag(gi*M)*imag(R)
    #Av[jp+1] +=  imag(gi*M)*real(R) - real(gi*M)*imag(R)

    #Av[ip]   +=  real( gj * conj(M)*R )
    #Av[ip+1] +=  imag( gj * conj(M)*R )
    #Av[jp]   +=  real( conj(gi) * conj(M)*R )
    #Av[jp+1] += -imag( conj(gi) * conj(M)*R )

    # separate the variable gains from the constant terms (M & V) so the latter can be pre-averaged
    Av[ip]   +=   real(      gj  * MV[i,j] ) - real(      gi *abs(gj)**2 * MM[i,j] )
    Av[ip+1] +=   imag(      gj  * MV[i,j] ) - imag(      gi *abs(gj)**2 * MM[i,j] )
    Av[jp]   +=   real( conj(gi) * MV[i,j] ) - real( conj(gj)*abs(gi)**2 * MM[i,j] )
    Av[jp+1] +=  -imag( conj(gi) * MV[i,j] ) + imag( conj(gj)*abs(gi)**2 * MM[i,j] )

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

# MAIN

r2d = 180.0/pi
d2r = pi/180.0
r2h = 12.0/pi
h2r = pi/12.0

# SKA1-LOW coordinates
latitude  = -26.82472208
longitude = 116.7644482

lat = latitude * d2r

#----------------------------------------------------------------------------------------------------------------------#

N = 32              # number of antennas
Nsamp = 10          # number of samples (i.e. time and frequency)
Nbl = ((N-1)*N)//2  # number of baselines
Nvis = Nbl * Nsamp  # number of visibilities
Np = 2*N            # number of free parameters: N x real & imag

Nit = 100           # number of iterations for the solver

# do referencing, etc., using extra NE constraints.
#  - in gain mode, there is only 1: phase referencing
add_constraints = 0
Nextra = add_constraints

# set antenna gain statistics
meanG = 1.0 + 0j
sigmaG = 0.1

# set visibility noise statistics
sigmaV = 0.01

# ideal gains (could make these random if we want)
g = meanG * ones([N,1],"complex")

# generate gain deviations from ideal
gn = sigmaG * (random.randn(N,1) + 1j*random.randn(N,1) )

# true gains
gt = g + gn

# phase referencing:
ref = 0
gt_ref = gt * exp(-1j*angle(gt[ref]))

#----------------------------------------------------------------------------------------------------------------------#

# Ideal visibility model for baseline ij:
#   I_ij(t,f)
# Visibility model (over the range of time and frequency that antenna gains are assumed constant):
#   M_ij(t,f) = g_i*conj(g_j) * I_ij(t,f)
# Measured visibility:
#   V_ij(t,f) = (g_i+e_i)*conj(g_j+e_j) * I_ij(t,f) + N_ij(t,f)
#
# N is the noise and e are additive errors for the antenna gains. The residual visibility is
#   R_ij(t,f) = V_ij(t,f) - M_ij(t,f)
#             = g_i*conj(e_j)*I_ij(t,f) + e_i*conj(g_j)*I_ij(t,f) + e_i*conj(e_j)*I_ij(t,f) + N_ij(t,f)
#             ~ g_i*conj(e_j)*I_ij(t,f) + e_i*conj(g_j)*I_ij(t,f) + N_ij(t,f)
#
# Solve as a least-squares problem with data re(R) & im(R) and free parameters re(e) & im(e). This allows the problem
# to be linearised as A*p = v, where A is the design matrix, p are the parameters, and v is the residual vector. e.g.
#   re(R_ij(t,f)) ~ re( conj(e_j)*g_i*I_ij(t,f) + e_i*conj(g_j)*I_ij(t,f) + re(N_ij(t,f)) )
#                 ~ + re(g_i*I_ij(t,f)) * re(e_j)
#                   + im(g_i*I_ij(t,f)) * im(e_j)
#                   + re(conj(g_j)*I_ij(t,f)) * re(e_i)
#                   - im(conj(g_j)*I_ij(t,f)) * im(e_i)
#                   + re(N_ij(t,f))
#
# So, for example, if the first residual vector element (v[0]) is the real component of baseline 0-1, re(R_01(t,f))
# would be added to it and the four coefficients above would be added to the first row of the design matrix (A[0,:])
# in the columns corresponding to their free parameter (ordered re0,im0,re1,im1,re2,im2,...). And the system is
# solved in the standard way: p = inv(A'*A) * A'*v, where ' represents a Hermitian transpose (here just a transpose
# because the system is real).
#
# Extra time and frequency can be added as extra rows of A and v, however averaging in time and frequency cannot
# take place until after forminng the normal matrix and data vector, otherwise the calibartor signal will decorrelate
# (except for very simple calibrators). Also, A and v need to be regenerated each iteration. To avoid this, note that
# the normal matrix AA and data vector Av can be formed separately for each sample and then averaged, and furthermore
# can be separated into sky terms that are variable with baseline and sample but constant with iteration, and gain
# terms that vary with iteration but are constant over different samples. So in a pre-averaging step the sky part
# of AA and Av can be averaged acroos all samples, and during calibration iterations only the averaged matrices need to
# be considered. This is shown in update_normal_matrix and update_data_vector.
# 
#----------------------------------------------------------------------------------------------------------------------#
#
# In the antsol approach (at least my version which is the basis for the RTS approach)
#
# Measured visibility:
#   V_ij(t,f) = (g_i*e_i)*conj(g_j*e_j) * I_ij(t,f) + N_ij(t,f)
# where e are now multiplicative errors for the antenna gains.
#
# For antenna i assume the j errors are approximately unity:
#   V_ij(t,f) = (g_i*e_i)*conj(g_j) * I_ij(t,f) + N_ij(t,f)
#             ~ e_i * M_ij(t,f) + N_ij(t,f)
#
# Solve via standard linear least squares:
#   e_i ~ sum_jtf{ V_ij(t,f)*conj(M_ij(t,f)) } / sum_jtf{ M_ij(t,f)*conj(M_ij(t,f)) } 
#
# Update: g_i *= e_i and iterate.
#  - not sure why, but the updates must be halved. I think the stefcal paper goes into this.
#
# But again, don't need to redo the sums over t and f each iteration:
#   e_i ~ sum_jtf{ V_ij(t,f)*conj(M_ij(t,f)) } / sum_jtf{ M_ij(t,f)*conj(M_ij(t,f)) } 
#   e_i ~ sum_j{ conj(g_i)*g_j * sum_tf{ V_ij(t,f)*conj(I_ij(t,f)) } } /
#         sum_j{ abs(g_i*g_j)**2 * sum_tf{ I_ij(t,f)*conj(I_ij(t,f)) } } 
#
# During each iteration, to update the VM sums mult by conj(e_i)*e_j and the MM sums by abs(e_i*e_j)**2
#
#----------------------------------------------------------------------------------------------------------------------#

# initialise free parameters for four different approaches:

# normal-equation approach via design matrix (no pre-averaging)
g1 = copy(g)
g1_ref = g1 * exp(-1j*angle(g1[ref]))

# normal-equation approach, forming the normal matrix directly (which allows for pre-averaging)
g2 = copy(g)
g2_ref = g2 * exp(-1j*angle(g2[ref]))

# antsol-style using NEq elements (to show that antsol is equivalent to ignoring off-diag terms of a linear system)
g3 = copy(g)
g3_ref = g3 * exp(-1j*angle(g3[ref]))

# antsol-style approach
g4 = copy(g)
g4_ref = g4 * exp(-1j*angle(g4[ref]))

#----------------------------------------------------------------------------------------------------------------------#

# generate visibilities and do the pre-averaging

MV2 = zeros((N,N),'complex')
MM2 = zeros((N,N),'complex')
MV3 = zeros((N,N),'complex')
MM3 = zeros((N,N),'complex')
VM4 = zeros((N,N),'complex')
MM4 = zeros((N,N),'complex')

V = []
Vm = []

# loop over samples, such as time or frequency
for sample in range(0,Nsamp):

    # visibility model. Make it random to be general.
    # e.g. could include point sources beating together or a large resolved object
    Vm.append( 1.0 * (random.randn(N,N) + 1j*random.randn(N,N)) )
    # give the model conjugate symmetry
    for i in range(0,N-1):
        for j in range(i+1,N):
            Vm[sample][i,j] = conj(Vm[sample][j,i])

    # generate visibility noise
    Vn = sigmaV * (random.randn(N,N) + 1j*random.randn(N,N))
    # give the noise conjugate symmetry
    for i in range(0,N-1):
        for j in range(i+1,N):
            Vn[i,j] = conj(Vn[j,i])

    # generate the true visibilities after passing through the instrument
    V.append( gen_model_vis( Vm[sample], gt ) + Vn )

    # do pre-averaging
    for i in range(0,N-1):
      for j in range(i+1,N):

        MM2[i,j] += conj(Vm[sample][i,j])*Vm[sample][i,j]
        MV2[i,j] += conj(Vm[sample][i,j])*V[sample][i,j]

        MM3[i,j] += conj(Vm[sample][i,j])*Vm[sample][i,j]
        MV3[i,j] += conj(Vm[sample][i,j])*V[sample][i,j]

        VM4[i,j] += V[sample][i,j]  * conj(Vm[sample][i,j])
        MM4[i,j] += Vm[sample][i,j] * conj(Vm[sample][i,j])

# initialise the multiplicative errors (antsol-style approach)
e4 = ones(N,'complex')

ant = 1
print ("%+e exp(%+ei)" % (abs(gt_ref[ant]), angle(gt_ref[ant])*r2d))
print (
    "----------------------------------------------------------------------------------------------------------------")

print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g1_ref[ant]), r2d*angle(g1_ref[ant]),
                                           abs(g1_ref[ant])-abs(gt_ref[ant]), r2d*angle(g1_ref[ant])-r2d*angle(gt_ref[ant])))
print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g2_ref[ant]), r2d*angle(g2_ref[ant]),
                                           abs(g2_ref[ant])-abs(gt_ref[ant]), r2d*angle(g2_ref[ant])-r2d*angle(gt_ref[ant])))
print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g3_ref[ant]), r2d*angle(g3_ref[ant]),
                                           abs(g3_ref[ant])-abs(gt_ref[ant]), r2d*angle(g3_ref[ant])-r2d*angle(gt_ref[ant])))
print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g4_ref[ant]), r2d*angle(g4_ref[ant]),
                                           abs(g4_ref[ant])-abs(gt_ref[ant]), r2d*angle(g4_ref[ant])-r2d*angle(gt_ref[ant])))
print ()

#----------------------------------------------------------------------------------------------------------------------#

err1 = []
err2 = []
err3 = []
err4 = []

err1.append( sqrt(mean(abs(g1_ref-gt_ref)**2)) )
err2.append( sqrt(mean(abs(g2_ref-gt_ref)**2)) )
err3.append( sqrt(mean(abs(g3_ref-gt_ref)**2)) )
err4.append( sqrt(mean(abs(g4_ref-gt_ref)**2)) )

it = 0
for it in range(0,Nit):

    # Normal equation approach
    # generate the design matrix directly for comparison
    AA1 = matrix(zeros([Np,Np],"double"))
    Av1 = matrix(zeros([Np,1],"double"))

    # or generate the normal matrix and data vector directly
    AA2 = matrix(zeros([Np,Np],"double"))
    Av2 = matrix(zeros([Np,1],"double"))

    # A third set of normal equations to test how antsol compares to ignoring off-diagonal terms of the normal matrix
    AA3 = matrix(zeros([Np,Np],"double"))
    Av3 = matrix(zeros([Np,1],"double"))

    # coefficients for a antsol style approach
    VMsum = zeros(N,'complex')
    MMsum = zeros(N,'complex')

    # update residual visibilities and design matrix. Need to loop over all of the samples each time...
    for sample in range(0,Nsamp):
        # could form a long matrix and vector with all Nvis data points, but better to update AA and Av
        A1 = matrix(zeros([2*Nbl+Nextra,Np],"double"))
        dv1 = matrix(zeros([2*Nbl+Nextra,1],"double"))
        # update the visibility models
        V1 = gen_model_vis( Vm[sample], g1 )
        kx2 = 0
        for i in range(0,N-1):
          for j in range(i+1,N):
            # update residual visibilities and design matrix
            dv1[kx2]   = real(V[sample][i,j] - V1[i,j])
            dv1[kx2+1] = imag(V[sample][i,j] - V1[i,j])
            update_design_matrix( A1, Vm[sample], g1, i, j )
            kx2 += 2
        if add_constraints:
            # ref phase = 0
            A1[kx2,ref+1] = 1
        # form the normal matrix and data vector
        AA1 += A1.T * A1
        Av1 += A1.T * dv1

    for i in range(0,N-1):
      for j in range(i+1,N):

        # update normal matrix and data vector
        update_normal_matrix( AA2, MM2, g2, i, j )
        update_data_vector( Av2, MV2, MM2, g2, i, j )

        # update normal matrix and data vector
        update_normal_matrix( AA3, MM3, g3, i, j )
        update_data_vector( Av3, MV3, MM3, g3, i, j )

        # update the visibility products
        VM4[i,j] *= conj(e4[i]) * e4[j]
        MM4[i,j] *= abs(e4[i] * e4[j])**2
        # update the antsol-style coefficients
        VMsum[i] += VM4[i,j]
        MMsum[i] += MM4[i,j]
        VMsum[j] += conj(VM4[i,j])
        MMsum[j] += MM4[i,j]

    # --------- #

    # update the design matrix approach gains
    u1,s1,v1 = linalg.svd(AA1, full_matrices=True)
    s1inv = 1.0/s1
    for k in range(0,Np):
        if (s1[k]/s1[0])<1e-6: s1inv[k]=0
    AAinv = v1.T*diag(s1inv)*u1.T
    # form updates for the gains
    e1 = AAinv * Av1
    # update the gains
    g1 += array(e1[0:2*N-1:2] + 1j*e1[1:2*N:2])
    g1_ref = g1 * exp(-1j*angle(g1[ref]))

    # update the direct normal matrix gains
    u2,s2,v2 = linalg.svd(AA2, full_matrices=True)
    s2inv = 1.0/s2
    for k in range(0,Np):
        if (s2[k]/s2[0])<1e-6: s2inv[k]=0
    AAinv = v2.T*diag(s2inv)*u2.T
    e2 = AAinv * Av2
    g2 += array(e2[0:2*N-1:2] + 1j*e2[1:2*N:2])
    g2_ref = g2 * exp(-1j*angle(g2[ref]))

    # update the antsol-style gains
    for i in range(0,N):
        ix2 = i*2
        g3[i] += 0.5 * ( Av3[ix2,0]/AA3[ix2,ix2] + 1j*Av3[ix2+1,0]/AA3[ix2+1,ix2+1] )
    g3_ref = g3 * exp(-1j*angle(g3[ref]))

    # update the antsol gains
    for i in range(0,N):
        e4[i] = 1.0 + 0.5 * ( VMsum[i] / MMsum[i] - 1.0 )
        g4[i] *= e4[i]
    g4_ref = g4 * exp(-1j*angle(g4[ref]))

    err1.append( sqrt(mean(abs(g1_ref-gt_ref)**2)) )
    err2.append( sqrt(mean(abs(g2_ref-gt_ref)**2)) )
    err3.append( sqrt(mean(abs(g3_ref-gt_ref)**2)) )
    err4.append( sqrt(mean(abs(g4_ref-gt_ref)**2)) )

    if 1:
        print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g1_ref[ant]), r2d*angle(g1_ref[ant]),
                                                   abs(g1_ref[ant])-abs(gt_ref[ant]), r2d*angle(g1_ref[ant])-r2d*angle(gt_ref[ant])))
        print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g2_ref[ant]), r2d*angle(g2_ref[ant]),
                                                   abs(g2_ref[ant])-abs(gt_ref[ant]), r2d*angle(g2_ref[ant])-r2d*angle(gt_ref[ant])))
        print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g3_ref[ant]), r2d*angle(g3_ref[ant]),
                                                   abs(g3_ref[ant])-abs(gt_ref[ant]), r2d*angle(g3_ref[ant])-r2d*angle(gt_ref[ant])))
        print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g4_ref[ant]), r2d*angle(g4_ref[ant]),
                                                   abs(g4_ref[ant])-abs(gt_ref[ant]), r2d*angle(g4_ref[ant])-r2d*angle(gt_ref[ant])))
        print ()

    # test for convergence
    if abs(err1[-1]-err1[-2])/err1[-1]<1e-3 and \
       abs(err2[-1]-err2[-2])/err2[-1]<1e-3 and \
       abs(err3[-1]-err3[-2])/err3[-1]<1e-3 and \
       abs(err4[-1]-err4[-2])/err4[-1]<1e-3: break

print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g1_ref[ant]), r2d*angle(g1_ref[ant]),
                                           abs(g1_ref[ant])-abs(gt_ref[ant]), r2d*angle(g1_ref[ant])-r2d*angle(gt_ref[ant])))
print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g2_ref[ant]), r2d*angle(g2_ref[ant]),
                                           abs(g2_ref[ant])-abs(gt_ref[ant]), r2d*angle(g2_ref[ant])-r2d*angle(gt_ref[ant])))
print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g3_ref[ant]), r2d*angle(g3_ref[ant]),
                                           abs(g3_ref[ant])-abs(gt_ref[ant]), r2d*angle(g3_ref[ant])-r2d*angle(gt_ref[ant])))
print ("%+e exp(%+ei) -> %+e exp(%+ei)" % (abs(g4_ref[ant]), r2d*angle(g4_ref[ant]),
                                           abs(g4_ref[ant])-abs(gt_ref[ant]), r2d*angle(g4_ref[ant])-r2d*angle(gt_ref[ant])))
print ()

print ("converged in %d iterations. 1/SNR = %.1e, 1/SNR/sqrt(N-1)/sqrt(Nsamples) = %.1e" % \
      ( it, sigmaV/abs(meanG), sigmaV/abs(meanG)/sqrt(N-1)/sqrt(Nsamp) ))
print ()

figure(num=0, figsize=(18,9), facecolor='w', edgecolor='k')

subplot(2,3,1)
plot( real(gt_ref), 'b', label="true gains" )
plot( real(g1_ref), 'r--', label="Norm Eq" )
plot( real(g2_ref), 'c:', label="ANTSOL" )
plot( real(g3_ref), 'm--', label="ANTSOL using NEq" )
plot( real(g4_ref), 'g:', label="ANTSOL" )
legend(loc=0, fontsize=11, frameon=False)
xlabel("antenna")
ylabel("real(gain)")

subplot(2,3,2)
plot( imag(gt_ref), 'b' )
plot( imag(g1_ref), 'r--' )
plot( imag(g2_ref), 'c:' )
plot( imag(g3_ref), 'm--' )
plot( imag(g4_ref), 'g:' )
xlabel("antenna")
ylabel("imag(gain)")

subplot(2,3,4)
plot( real(g1_ref-gt_ref), 'r' )
plot( real(g2_ref-gt_ref), 'c--' )
plot( real(g3_ref-gt_ref), 'm' )
plot( real(g4_ref-gt_ref), 'g--' )
xlabel("antenna")
ylabel("real(gain error)")

subplot(2,3,5)
plot( imag(g1_ref-gt_ref), 'r' )
plot( imag(g2_ref-gt_ref), 'c--' )
plot( imag(g3_ref-gt_ref), 'm' )
plot( imag(g4_ref-gt_ref), 'g--' )
xlabel("antenna")
ylabel("imag(gain error)")

ca=subplot(1,3,3)
plot( err1, '.r-', label="Norm Eq (via A)" )
plot( err2, '.c--', label="Norm Eq (via AA)" )
plot( err3, '.m-', label="ANTSOL using NEq" )
plot( err4, '.g--', label="ANTSOL" )
xlabel("iteration")
ylabel("gain error")
ca.set_yscale('log')
legend(loc=1, fontsize=11, frameon=False)

show()

