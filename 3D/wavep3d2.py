from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
c=200
P0=Point(-2.0,-2.0)
P1=Point(2.0,2.0)
mesh = RectangleMesh(P0,P1,150,150)
V=FunctionSpace(mesh, "Lagrange", 1)

# Time variables
dt = 0.0001; t = 0; T = 1

# Previous and current solution
u1= interpolate(Constant(0.0), V)
u0= interpolate(Constant(0.0), V)

# Variational problem at each time
u = TrialFunction(V)
v = TestFunction(V)

a = u*v*dx + dt*dt*c*c*inner(grad(u), grad(v))*dx
L = 2*u1*v*dx-u0*v*dx


def GammaD(x, on_boundary):
    r2=x[0]**2+x[1]**2
    if np.abs(r2-4)<0.1:
        return True
    else:
        return False 


bc = DirichletBC(V, 0, GammaD)
A, b = assemble_system(a, L, bc)

u=Function(V)
ii=0
while t <= T:
    A, b = assemble_system(a, L, bc)
    delta = PointSource(V, Point(0, 0), sin(c * 10 * t))
    delta.apply(b)
    mesh_points=mesh.coordinates()
    x0=np.linspace(-2,2,150)
    
    x1=np.linspace(-2,2,150)
    Z=[]
    for iii in x0:
        Z1=[]
        for jj in x1:
            Z1.append(u(iii,jj))
        Z.append(np.array(Z1))
    Z=np.array(Z)
    X,Y= np.meshgrid(x0,x1)
    
    #print(Z[0])
    #Z=X**2+Y**2
    #print(Z[0])
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_zlim(-0.8,0.8)
    #curva
    tt=np.linspace(0,2*np.pi,100)
    xx=2*np.cos(tt)
    yy=2*np.sin(tt)
    ax.plot(xx, yy,color="black")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    solve(A, u.vector(), b)
    u0.assign(u1)
    u1.assign(u)
    t += dt
    j1 = 0
    for i1 in u.vector():
        i1 = min(.01, i1)
        i1 = max(-.01, i1)
        u.vector()[j1] = i1;
        j1 += 1
    #plot(u)
    plt.savefig("./images/Data{:}.png".format(ii))
    
    ii+=1
    
