# %load triangleplot.py
def triangleplot(x,y,z,levels=[],scale='lin',title='',xlabel='',ylabel='',zlabel=''):
    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    import numpy as np
    import matplotlib.cm as cm
    import matplotlib.mlab as mlab
    import scipy.interpolate

    import matplotlib.cbook as cbook
    import matplotlib.tri as mtri

    # Data to plot triangular grid
    data=np.loadtxt("grid.dat")

    # Create  & plot triangulation.
    triang = mtri.Triangulation(data[:,0], data[:,1])
    plt.triplot(triang, color='0.5')

    #Load the exclusion limits data in the form: BRa BRb BRc limit. 
    #This is in principle the result of running your dedicated code implementing the analysis.



    #Drawing the medians if necessary

    plt.plot([0.5,0.5], [0, 0.866], 'ko-')
    plt.plot([0.25,1], [0.433, 0], 'ko-')
    plt.plot([0,0.75], [0, 0.433], 'ko-')


    #x=(2*b + c)/2
    #y=sqrt(3)*c/2
    
    #plt.hexbin(x,y,np.abs(df.m_nu_1),norm=matplotlib.colors.LogNorm(),gridsize =200)


    #Building the griddata by interpolating between the grid points 
    #plt.title(r'$m_{MIN}$ (eV)',verticalalignment='bottom')
    # just an arbitrary number for grid point
    ngrid = 200

    # you could use x.min()/x.max() for creating xi and y.min()/y.max() for yi
    xi = np.linspace(x.min(),x.max(),ngrid)
    yi = np.linspace(y.min(),y.max(),ngrid)

    # create the grid data for the contour plot
    zi = mlab.griddata(x,y,z,xi,yi, interp='linear')

    CS3 = plt.contourf(xi,yi,zi,levels=levels, extend='both') #,cmap=plt.cm.winter
    #CS3 = plt.contourf(xi,yi,zi,norm=matplotlib.colors.LogNorm() )
    CS3c=plt.colorbar(pad=0.07)
    if scale=='log':
        CS3c.set_ticklabels([ r'$10^{%d}$' %i for i in levels])
    #Vertex Labels
    #plt.text(-0.1,-0.02, r'$g \tilde{\chi}^0_1$ ', fontsize=20)
    #plt.text(1.02,-0.02,r'$ b \bar{b} \tilde{\chi}^0_1$'min_nulight=min_nulight,\
    #plt.text(0.48,0.88,r'$t \bar{t} \tilde{\chi}^0_1$', fontsize=20)
    plt.text(1.,0.93,title, fontsize=20)
    plt.text(-0.07,-0.02, xlabel, fontsize=20)
    plt.text(1.02,-0.02,ylabel, fontsize=20)
    plt.text(0.48,0.88,zlabel, fontsize=20)

    #grid labels
    plt.text(0.11,0.18, '0.2')
    plt.text(0.21,0.35, '0.4')
    plt.text(0.31, 0.52, '0.6')
    plt.text(0.41,0.69, '0.8')
    plt.text(0.85,0.173, '0.8', rotation=60)
    plt.text(0.75,0.34, '0.6', rotation=60)
    plt.text(0.65, 0.51, '0.4', rotation=60)
    plt.text(0.55,0.69, '0.2', rotation=60)
    plt.text(0.18,0.04, '0.8',rotation=-60)
    plt.text(0.38,0.04, '0.6', rotation=-60)
    plt.text(0.58, 0.04, '0.4', rotation=-60)
    plt.text(0.78,0.04, '0.2', rotation=-60)
    plt.ylim(-0.05,0.9)
    plt.axis('off')
    