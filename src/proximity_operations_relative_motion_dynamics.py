import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import spiceypy as spice

spice.furnsh('gm_de440.tpc')
spice.furnsh('pck00010.tpc')

mu=spice.bodvrd('Earth','GM',1)[1][0]
RE=spice.bodvrd('Earth','RADII',3)[1][0]
j2=1.08262668e-3

kep_t_t0=np.array([6723.2576, 0.1, 0., 0., 0., 270.0882])
kep_c_t0=np.array([6723.2576, 0.1, 0., 0., 0., 270.0832])

## FUNCTIONS

def kep_prop(kep0,t0,tf):
    # This function propagates true anomaly given initial keplerian elements
    # INPUTS:
    # kep0 initial keplerian elements
    # kep0 = [a, e, i, OM, om, f0]
    #    a semi major axis [km]
    #    e eccentricity
    #    i inclination [deg]
    #    OM RAAN [deg]
    #    om pericenter anomaly [deg]
    #    f0 initial true anomaly [deg]
    # t0 initial epoch
    # tf final epoch
    # OUTPUTS:
    # kep final keplerian elements
    # kep = [a, e, i, OM, om, f0]
    #     f final true anomaly [deg]


    a=kep0[0]
    e=kep0[1]

    f0=np.deg2rad(kep0[-1])                                                         # initial true anomaly [rad]
    E0=(2*np.atan2(np.sqrt(1-e)*np.sin(f0/2),np.sqrt(1+e)*np.cos(f0/2)))%(2*np.pi)  # initial eccentric anomaly [rad]
    M0=E0-e*np.sin(E0)                                                              # initial mean anomaly [rad]

    n=np.sqrt(mu/a**3)          # mean angular velocity
    M=(M0+n*(tf-t0))%(2*np.pi)  # final mean anomaly [rad]

    fzero=lambda x: x-e*np.sin(x)-M
    fzerop=lambda x: 1-e*np.cos(x)

    sol=sp.optimize.root_scalar(
        f=fzero,
        fprime=fzerop,
        x0=M,
        xtol=1e-12
        )
    
    E=sol.root
    f=(2*np.atan2(np.sqrt(1+e)*np.sin(E/2),np.sqrt(1-e)*np.cos(E/2)))%(2*np.pi)

    kep=kep0.copy()
    kep[-1]=np.rad2deg(f)

    return kep

def kep2car(kep):
    # This function transforms keplerian elements into cartesian state vector
    # INPUT:
    # kep = [a, e, i, OM, om, f]
    #    a semi major axis [km]
    #    e eccentricity
    #    i inclination [deg]
    #    OM RAAN [deg]
    #    om pericenter anomaly [deg]
    #    f true anomaly [deg]
    # OUTPUT:
    # x = [r, v]
    #     r position vector [km]
    #     v velocity vector [km/s]

    a=kep[0]
    e=kep[1]
    i=np.deg2rad(kep[2])    # inclination [rad]
    OM=np.deg2rad(kep[3])   # RAAN [rad]
    om=np.deg2rad(kep[4])   # pericenter anomaly [rad]
    f=np.deg2rad(kep[-1])   # true anomaly [rad]

    r=a*(1-e**2)/(1+e*np.cos(f))
    v=np.sqrt(2*mu/r-mu/a)
    g=np.atan2(e*np.sin(f),1+e*np.cos(f))

    x=np.empty(6)
    x[0]=r*(np.cos(om+f)*np.cos(OM)-np.sin(om+f)*np.cos(i)*np.sin(OM))
    x[1]=r*(np.cos(om+f)*np.sin(OM)+np.sin(om+f)*np.cos(i)*np.cos(OM))
    x[2]=r*(np.sin(om+f)*np.sin(i))
    x[3]=v*(-np.sin(f+om-g)*np.cos(OM)-np.cos(f+om-g)*np.cos(i)*np.sin(OM))
    x[4]=v*(-np.sin(f+om-g)*np.sin(OM)+np.cos(f+om-g)*np.cos(i)*np.cos(OM))
    x[5]=v*(np.cos(f+om-g)*np.sin(i))

    return x

def T_LVLH_I(kep):
    # This function builds the LVLH direction cosine matrix in the ECI coordinate frame
    # INPUT:
    # kep = [a, e, i, OM, om, f]
    #    a semi major axis [km]
    #    e eccentricity
    #    i inclination [deg]
    #    OM RAAN [deg]
    #    om pericenter anomaly [deg]
    #    f true anomaly [deg]
    # OUTPUT:
    # T LVLH DCM in ECI coordinate frame

    i=np.deg2rad(kep[2])
    OM=np.deg2rad(kep[3])
    om=np.deg2rad(kep[4])
    f=np.deg2rad(kep[5])

    c=lambda x: np.cos(x)
    s=lambda x: np.sin(x)

    T=np.array([
        [ c(OM)*c(om+f)-s(OM)*s(om+f)*c(i),  s(OM)*c(om+f)+c(OM)*s(om+f)*c(i),   s(om+f)*s(i)],
        [-c(OM)*s(om+f)-s(OM)*c(om+f)*c(i), -s(OM)*s(om+f)+c(OM)*c(om+f)*c(i),   c(om+f)*s(i)],
        [s(OM)*s(i),                        -c(OM)*s(i),                         c(i)]
    ])

    return T
    
def T_TH(kep,h):
    # This function builds the transformation matrix to chance from time domain to modified true anomaly domain
    # INPUTS:
    # kep = [a, e, i, OM, om, f]
    #    a semi major axis [km]
    #    e eccentricity
    #    i inclination [deg]
    #    OM RAAN [deg]
    #    om pericenter anomaly [deg]
    #    f true anomaly [deg]
    # h angular momentum norm [km2/s]
    # OUTPUT:
    # T Tschauner-Hempel domain transformation matrix

    e=kep[1]
    f=np.deg2rad(kep[5])

    p=h**2/mu

    T=np.zeros((6,6))

    T[:3,:3]=(1+e*np.cos(f))*np.eye(3)
    T[3:,:3]=-e*np.sin(f)*np.eye(3)
    T[3:,3:]=(p**2/(h*(1+e*np.cos(f))))*np.eye(3)

    return T

## RELATIVE MOTION PROPAGATORS

def kep_mot(kep_t_t0,kep_c_t0,t_eval):
    # This function propagates relative motion using keplerian dynamics
    # INPUTS:
    # kep_t_t0 initial target orbital elements [a, e, i, OM, om, f]
    # kep_c_t0 initial chaser orbital elements
    # t_eval time evaluation grid
    # OUTPUTS:
    # rho_xx_lvlh_kep relative state in LVLH frame [r, v]    

    rho_xx_lvlh_kep=np.empty((len(t_eval),6))

    for i in range(len(t_eval)):
            
        kep_t=kep_prop(kep_t_t0,t0,t_eval[i])
        kep_c=kep_prop(kep_c_t0,t0,t_eval[i])

        x_t=kep2car(kep_t)
        x_c=kep2car(kep_c)

        if i==0:
            h0=np.linalg.norm(np.cross(x_t[:3],x_t[3:]))

        rho_x_eci=x_c-x_t

        R=T_LVLH_I(kep_t)

        w=h0/np.dot(x_t[:3],x_t[:3])
        W=np.zeros((3,3))
        W[0,1]=-w
        W[1,0]=w

        T=np.zeros((6,6))
        T[:3,:3]=T[3:,3:]=R
        T[3:,:3]=-W@R

        rho_xx_lvlh_kep[i,:]=T@rho_x_eci

    return rho_xx_lvlh_kep

def tbp_mot(x_t_t0,x_c_t0,t_eval):
    # This function propagates relative motion using two-body-problem dynamics with J2 perturbation
    # INPUTS:
    # x_t_t0 initial target state in ECI frame [r, v]
    # x_c_t0 initial chaser state in ECI frame
    # t_eval time evaluation grid
    # OUTPUTS:
    # rho_xx_lvlh_tbp relative state in LVLH frame [r, v]

    def tbp_pert(t,x):

        R=x[:3]
        V=x[3:]

        r=np.linalg.norm(R)
        [rx,ry,rz]=R

        a2b=-mu*R/r**3
        aj2=-3/2*j2*mu*RE**2/r**4*np.array([
            (1-5*(rz/r)**2)*rx/r,
            (1-5*(rz/r)**2)*ry/r,
            (3-5*(rz/r)**2)*rz/r
        ])

        dx=np.empty(6)
        dx[:3]=V
        dx[3:]=a2b+aj2

        return dx
    
    int=sp.integrate.solve_ivp(
            fun=tbp_pert,
            t_span=[t0,tf],
            t_eval=t_eval,
            y0=x_t_t0,
            atol=1e-15,
            rtol=1e-13,
            method='DOP853'
            )
    
    xx_t=int.y

    int=sp.integrate.solve_ivp(
            fun=tbp_pert,
            t_span=[t0,tf],
            t_eval=t_eval,
            y0=x_c_t0,
            atol=1e-15,
            rtol=1e-13,
            method='DOP853'
            )
    
    xx_c=int.y

    rho_xx_lvlh_tbp=np.empty((len(t_eval),6))

    for i in range(len(t_eval)):
        R=xx_t[:3,i]    # ECI position
        V=xx_t[3:,i]    # ECI velocity

        H=np.cross(R,V)

        ix=R/np.linalg.norm(R)
        iz=H/np.linalg.norm(H)
        iy=np.cross(iz,ix)

        RR=np.array([ix, iy, iz])   # LVLH DCM in ECI frame

        fz=tbp_pert(t_eval[i],xx_t[:,i])[2]

        w=np.array([np.linalg.norm(R)*fz/np.linalg.norm(H), 0, np.linalg.norm(H)/np.dot(R,R)])

        W=np.zeros((3,3))
        W[0,1]=-w[2]
        W[1,0]=w[2]
        W[1,2]=-w[0]
        W[2,1]=w[0]

        T=np.zeros((6,6))
        T[:3,:3]=T[3:,3:]=RR
        T[3:,:3]=-W@RR

        rho_xx_lvlh_tbp[i,:]=T@(xx_c[:,i]-xx_t[:,i])

    return rho_xx_lvlh_tbp

def CW_mot(x_t_t0,x_c_t0,t_eval):
    # This function propagates relative motion using Clohessy-Wiltshire dynamics
    # INPUTS:
    # x_t_t0 initial target state in ECI frame [r, v]
    # x_c_t0 initial chaser state in ECI frame
    # t_eval time evaluation grid
    # OUTPUTS:
    # rho_xx_lvlh_cw relative state in LVLH frame [r, v]

    R=x_t_t0[:3]
    V=x_t_t0[3:]

    H=np.cross(R,V)

    ix=R/np.linalg.norm(R)
    iz=H/np.linalg.norm(H)
    iy=np.cross(iz,ix)

    RR=np.array([ix, iy, iz])   # LVLH DCM in ECI frame

    n=np.sqrt(mu/np.linalg.norm(R)**3)

    W=np.zeros((3,3))
    W[0,1]=-n
    W[1,0]=n

    T=np.zeros((6,6))
    T[:3,:3]=T[3:,3:]=RR
    T[3:,:3]=-W@RR

    rho_lvlh_t0=T@(x_c_t0-x_t_t0)

    A=np.zeros((6,6))
    A[:3,3:]=np.eye(3)
    A[3,0]=3*n**2
    A[5,2]=-n**2
    A[3,4]=2*n
    A[4,3]=-2*n

    rho_xx_lvlh_cw=np.empty((len(t_eval),6))

    for i in range(len(t_eval)):
        rho_xx_lvlh_cw[i,:]=sp.linalg.expm(A*(t_eval[i]))@rho_lvlh_t0

    return rho_xx_lvlh_cw

def TH_mot(kep_t_t0,kep_c_t0,t_eval):
    # This function propagates relative motion using Tschauner-Hempel dynamics
    # INPUTS:
    # kep_t_t0 initial target orbital elements [a, e, i, OM, om, f]
    # kep_c_t0 initial chaser orbital elements
    # t_eval time evaluation grid
    # OUTPUTS:
    # rho_xx_lvlh_th relative state in LVLH frame [r, v]

    kep_t=np.array([kep_prop(kep_t_t0,t_eval[0],ti) for ti in t_eval])

    f_eval=np.array([np.deg2rad(kep[-1]) for kep in kep_t])     # true anomaly evaluation grid
    
    for i in range(len(f_eval)-1):      # monotonous true anomaly evaluation grid
        if f_eval[i+1]<f_eval[i]:
            kc=np.round(np.abs(f_eval[i+1]-f_eval[i])/(2*np.pi))
            f_eval[i+1]+=2*kc*np.pi

    x_t_t0=kep2car(kep_t_t0)
    x_c_t0=kep2car(kep_c_t0)

    e=kep_t_t0[1]

    R=x_t_t0[:3]
    V=x_t_t0[3:]

    H=np.cross(R,V)

    ix=R/np.linalg.norm(R)
    iz=H/np.linalg.norm(H)
    iy=np.cross(iz,ix)

    RR=np.array([ix, iy, iz])

    fdot=np.linalg.norm(H)/np.dot(R,R)
    
    W=np.zeros((3,3))
    W[0,1]=-fdot
    W[1,0]=fdot

    T=np.zeros((6,6))
    T[:3,:3]=T[3:,3:]=RR
    T[3:,:3]=-W@RR

    rho_lvlh_t0=T@(x_c_t0-x_t_t0)   # initial relative state in LVLH frame

    TTH=T_TH(kep_t_t0,np.linalg.norm(H))    # TH domain transformation matrix

    rho_hat_xx_lvlh_t0=TTH@rho_lvlh_t0      # initial modified relative state in LVLH frame

    def TH_dyn(f,rho):

        A=np.zeros((6,6))
        A[:3,3:]=np.eye(3)
        A[3,0]=3/(1+e*np.cos(f))
        A[-1,2]=-1.
        A[3,4]=2.
        A[4,3]=-2

        drho=A@rho

        return drho
    
    int=sp.integrate.solve_ivp(
            fun=TH_dyn,
            t_span=[f_eval[0],f_eval[-1]],
            t_eval=f_eval,
            y0=rho_hat_xx_lvlh_t0,
            atol=1e-15,
            rtol=1e-13,
            method='DOP853'
            )
    
    rho_hat_xx_lvlh_th=int.y
    rho_xx_lvlh_th=np.array([np.linalg.inv(T_TH(kep_t[i],np.linalg.norm(H)))@rho_hat_xx_lvlh_th[:,i] for i in range(len(f_eval))])
    
    return rho_xx_lvlh_th   # relative state in LVLH frame

##############################################################

t0=0.           # initial time
tf=180.*60      # final time
n_plot=250      # evaluaiton points

t_eval=np.linspace(t0,tf,n_plot,True)   # evaluation space

x_t_t0=kep2car(kep_t_t0)    # initial target eci position
x_c_t0=kep2car(kep_c_t0)    # initial chaser eci position

rho_xx_lvlh_kep=kep_mot(kep_t_t0,kep_c_t0,t_eval)   # relative lvlh position (kep dyn) [km, km/s]
rho_xx_lvlh_tbp=tbp_mot(x_t_t0,x_c_t0,t_eval)       # relative lvlh position (pert 2bp dyn) [km, km/s]
rho_xx_lvlh_cw=CW_mot(x_t_t0,x_c_t0,t_eval)         # relative lvlh position (cw dyn) [km, km/s]
rho_xx_lvlh_th=TH_mot(kep_t_t0,kep_c_t0,t_eval)     # relative lvlh position (th dyn) [km, km/s]

rho_dd_kep=np.array([np.linalg.norm(rho[:3]) for rho in rho_xx_lvlh_kep])*1e3   # relative distance (kep dyn) [m]
rho_dd_tbp=np.array([np.linalg.norm(rho[:3]) for rho in rho_xx_lvlh_tbp])*1e3   # relative distance (pert 2bp dyn) [m]
rho_dd_cw=np.array([np.linalg.norm(rho[:3]) for rho in rho_xx_lvlh_cw])*1e3     # relative distance (cw dyn) [m]
rho_dd_th=np.array([np.linalg.norm(rho[:3]) for rho in rho_xx_lvlh_th])*1e3     # relative distance (th dyn) [m]

plt.figure()
plt.suptitle('@LVLH')

ax=plt.subplot(2,2,1)
plt.plot(rho_xx_lvlh_kep[:,0],rho_xx_lvlh_kep[:,1],'b')
plt.plot(rho_xx_lvlh_kep[0,0],rho_xx_lvlh_kep[0,1],'bo')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['KEP'])

plt.subplot(2,2,2, sharex=ax, sharey=ax)
plt.plot(rho_xx_lvlh_tbp[:,0],rho_xx_lvlh_tbp[:,1],'b')
plt.plot(rho_xx_lvlh_tbp[0,0],rho_xx_lvlh_tbp[0,1],'bo')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['2BP+J2'])

plt.subplot(2,2,3, sharex=ax, sharey=ax)
plt.plot(rho_xx_lvlh_cw[:,0],rho_xx_lvlh_cw[:,1],'b')
plt.plot(rho_xx_lvlh_cw[0,0],rho_xx_lvlh_cw[0,1],'bo')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['CW'])

plt.subplot(2,2,4, sharex=ax, sharey=ax)
plt.plot(rho_xx_lvlh_th[:,0],rho_xx_lvlh_th[:,1],'b')
plt.plot(rho_xx_lvlh_th[0,0],rho_xx_lvlh_th[0,1],'bo')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['TH'])

plt.figure()

ax=plt.subplot(2,2,1)
plt.plot(t_eval/60,rho_dd_kep)
plt.xlabel('t [min]')
plt.ylabel('|rho| [m]')
plt.legend(['KEP'])

plt.subplot(2,2,2, sharex=ax, sharey=ax)
plt.plot(t_eval/60,rho_dd_tbp)
plt.xlabel('t [min]')
plt.ylabel('|rho| [m]')
plt.legend(['2BP+J2'])

plt.subplot(2,2,3, sharex=ax, sharey=ax)
plt.plot(t_eval/60,rho_dd_cw)
plt.xlabel('t [min]')
plt.ylabel('|rho| [m]')
plt.legend(['CW'])

plt.subplot(2,2,4, sharex=ax, sharey=ax)
plt.plot(t_eval/60,rho_dd_th)
plt.xlabel('t [min]')
plt.ylabel('|rho| [m]')
plt.legend(['TH'])

plt.show()