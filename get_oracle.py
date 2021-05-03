import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_oracle(bounds,nx,valid,oracle,optimum_points,nlines=50):
	"""
	Plot the oracle function (if the input is 1D or 2D)
	bounds input space boundaries
	nx discretization Size
	valid function that returns if an input is valid
	oracle oracle function to plot
	optimum_points optimum location
	nlines number of contour lines
	"""
	if len(bounds)==2:
		X1 = np.linspace(bounds[0][0],bounds[0][1],nx)
		X2 =  np.linspace(bounds[1][0],bounds[1][1],nx)
		X1, X2 = np.meshgrid(X1, X2)
		Z=np.zeros((X1.shape[0],X1.shape[1]))
		V=np.zeros((X1.shape[0],X1.shape[1]))
		for i in range(nx):
			for j in range(nx):
				if valid(np.atleast_2d(np.hstack([X1[i,j],X2[i,j]])),oracle)==1:
					V[i,j]=1
				Z[i,j]=oracle(np.atleast_2d(np.hstack([X1[i,j],X2[i,j]])))

		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111)
		for xx in optimum_points:
			plt.scatter(xx[0],xx[1], marker='*',s=120,color='r')
		cp = ax.contour(X1, X2, Z, nlines, cmap=cm.hot_r,vmin=np.min(Z), vmax=np.max(Z) ) #, colors=colours)

	elif len(bounds)==1:
		X1 = np.linspace(bounds[0][0],bounds[0][1],nx)
		Z=np.zeros(len(X1))
		V=np.zeros(len(X1))
		for i in range(nx):
			if valid(X1[i],oracle)==1:
				V[i]=1
			Z[i]=oracle(X1[i])
		fig = plt.figure()#figsize=(10,10))
		ax = fig.add_subplot(111)
        #for xx in optimum_points:
        #    plt.plot(xx,0, marker='*',color='r')
		plt.plot(X1,Z)



def valid(x,f):
	"""
    Function that returns if an input is valid
    """
	return 1.0

def get_oracle(nameBenchmark):
	"""
    Get the oracle function

	nameBenchmark name of the benchmark used
    """
	if nameBenchmark=='Forrester':
		def oracle(x):
			return -(6*x-2)**2*np.sin(12*x-4)
		bounds=[[0,1]]
		optimum_points=[[0.75675]]
		plot_oracle(bounds,50,valid,oracle,optimum_points)
	elif nameBenchmark=='sixhump':
		def oracle(x):
			x=np.atleast_2d(x)
			x1 = x[:,0]
			x2 = x[:,1]
			return -((4 - 2.1*(x1*x1) + (x1*x1*x1*x1)/3.0)*(x1*x1) + x1*x2 + (-4 + 4*(x2*x2))*(x2*x2))
		bounds=[[-2,2],[-1,1]]
		optimum_points=[[0.089,-0.7126],[-0.089,0.7126]]
		plot_oracle(bounds,50,valid,oracle,optimum_points)
	elif nameBenchmark=='goldstein':
		def oracle(x):
			x=np.atleast_2d(x)
			x1 = x[:,0]
			x2 = x[:,1]
			fact1a = (x1 + x2 + 1)**2;
			fact1b = 19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2;
			fact1 = 1 + fact1a*fact1b;

			fact2a = (2*x1 - 3*x2)**2;
			fact2b = 18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2;
			fact2 = 30 + fact2a*fact2b;

			return -fact1*fact2;

		bounds=[[-2,2],[-2,2]]
		optimum_points=[[0,-1]]
		plot_oracle(bounds,100,valid,oracle,optimum_points,50)
	elif nameBenchmark=='levy':
		def oracle(x):
			x=np.atleast_2d(x*10)
			out=[]
			for ll in range(x.shape[0]):
				d = x.shape[1]
				w=np.zeros(d)
				for ii in range(d):
					w[ii] = 1 + (x[ll,ii] - 1)/4;

				term1 = (np.sin(np.pi*w[0]))**2;
				term3 = (w[-1]-1)**2 * (1+(np.sin(2*np.pi*w[-1])**2))

				sumv = 0;
				for ii in range(0,d-1):
					wi = w[ii]
					new = (wi-1)**2 * (1+10*(np.sin(np.pi*wi+1))**2)
					sumv = sumv + new;
				out.append(-(term1 + sumv + term3))
			return np.array(out)
		bounds=[[-1,1],[-1,1]]
		optimum_points=[[1/10,1/10]]
		plot_oracle(bounds,100,valid,oracle,optimum_points,50)
	elif nameBenchmark=='rosenbrock':
		def oracle(x):
			r = np.sum(100 * (x.T[1:] - x.T[:-1] ** 2.0) ** 2 + (1 - x.T[:-1]) ** 2.0, axis=0)
			return -r
		bounds=[[-3,3],[-3,3],[-3,3],[-3,3],[-3,3]]
		optimum_points=[[1,1,1,1,1]]
	elif nameBenchmark=='hartman6':
		def oracle(x):
			x=np.atleast_2d(x)
			A =  np.array([[10, 3, 17, 3.5, 1.7, 8],
			[0.05, 10, 17, 0.1, 8, 14],
			[3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]], dtype=float)

			B = np.array([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
			[0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
			[0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
			[0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]], dtype=float)

			C =  np.array([1, 1.2, 3, 3.2], dtype=float)
			y=[]
			for j in range(x.shape[0]):
				y.append(-np.sum(C[i] * np.exp(-sum(A[i] * (x[j]-B[i])**2)) for i in range(4)))
			return -np.array(y)
		bounds=[[0,1]]*6
		optimum_points=[[0.2017, 0.15, 0.4769, 0.2753, 0.3117, 0.6573]]
	else:
		print("Wrong nameBenchmark")

	return oracle, bounds, optimum_points
