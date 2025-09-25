import matplotlib.pyplot as plt; plt.ion()
import numpy as np
from PrototypingRead import AstrocalibRead
import pandas as pd
from scipy import interpolate

class CreateDist():
    def __init__(self,*,nsamp=31, dist_samp=15,dxdy=0.5,num_shifts=2,rand_scale=10e-5,std=5,N_poly=6,centroid_noise_std=10e-6,mode='random',plot='no',int_test='no'):
        self._nsamp = nsamp # Number of points in each dimension for evaluating/plotting distortion
        self._dist_samp = dist_samp # Number of points in each dimension for generating distortion
        self._dxdy = dxdy # The radial distance of the shifts
        self._num_shifts = num_shifts # The number of shifts to perform
        self._rand_scale = rand_scale # The scale of the random distortion
        self._std = std # The spatial correlation scale of the random distortion
        self._N_poly = N_poly # The order of the polynomial to fit
        self._centroid_noise_std = centroid_noise_std # The centroid noise to add to the simulated measurements
        self._mode = mode   # The mode of distortion to use (random or u_hat)
        self._plot = plot   # Whether to plot the distortion or not
        self._int_test = int_test   # Whether to perform the interpolation test or not
       
        # Create a grid of data points for generating/holding distortions
        self._x_rand = np.linspace(-17, 17, dist_samp)  # X-coordinates
        self._y_rand = np.linspace(-17, 17, dist_samp)  # Y-coordinates
        x_rand_mesh, y_rand_mesh = np.meshgrid(self._x_rand,self._y_rand)
        self._x_rand_flattened = x_rand_mesh.flatten()
        self._y_rand_flattened = y_rand_mesh.flatten()

        # Grids to evaluate/plot distortions
        self._x_linspace = np.linspace(-15,15,self._nsamp)
        self._y_linspace = np.linspace(-15,15,self._nsamp)
        x_mesh,y_mesh = np.meshgrid(self._x_linspace,self._y_linspace)
        self._x_flattened = x_mesh.flatten()
        self._y_flattened = y_mesh.flatten()

        #Loading the AstrocalibRead package
        data = pd.DataFrame({'home':[np.nan],'shiftx':[np.nan],'shifty':[np.nan]})
        corners = pd.DataFrame({'data':[np.nan],'dimension':np.nan,'dx':np.nan,'dy':np.nan})
        self._sim = AstrocalibRead(data,corners,'C')
        


        #Random distortion with spatial correlations
    def _gen_dist(self,plot='no'):
        #n_points:  number of points in each grid dimension
        #std:  spatial covariance modelled as gaussian with std: (higher std = more spatially correlated)

        # create a grid and centre it at (0,0)
        xx,yy = np.meshgrid(np.linspace(-17, 17, self._dist_samp),np.linspace(-17, 17, self._dist_samp),indexing="xy")
        xx = xx - xx.mean()
        xx = xx.flatten()
        yy = yy - yy.mean()
        yy = yy.flatten()
        # build distance matrix from all points to all points:
        rr = ((xx[:,None]-xx[None,:])**2 + (yy[:,None]-yy[None,:])**2)**0.5
        # evaluate covariance matrix at distance matrix points:
        cov = 1/(self._std*(2*np.pi)**0.5)*np.exp(-0.5*rr**2/self._std**2)
        # factorise covariance matrix into "square root" form, so we can generate random realisations
        w,v = np.linalg.eigh(cov)
        filt = w>1e-7
        w = w[filt]
        v = v[:,filt]
        L = v @ np.diag(w**0.5)
        # generate normally distributed random input (one each for x and y):
        v = np.random.randn(L.shape[1],2)
        # map it to random variable with desired covariance:
        d = L @ v
        # plot
        if plot=='yes':
            plt.figure(figsize=[8,8])
            arrow_sf = self._rand_scale/np.std(d)
            for i in range(xx.shape[0]):
                plt.arrow(xx[i],yy[i],arrow_sf*d[i,0],arrow_sf*d[i,1],width=0.05,color='b',head_width=0.35,length_includes_head=True)
            plt.axis("equal")
            plt.legend(["random distortion"])
            plt.title(f"Random Spatially Correlated Distortion")
            plt.xlabel("x-position in field [arcsec]")
            plt.ylabel("y-position in field [arcsec]")
            plt.show()
        return d*self._rand_scale/np.std(d)

    def _do_shifts(self):
    #Generate a unit circle to determine the shifting amount
        angles = np.linspace(0, 2 * np.pi, self._num_shifts, endpoint=False)

        # Calculate the x and y coordinates of the points using trigonometry
        x_ang = np.cos(angles)
        y_ang = np.sin(angles)

        # Create a list of (x, y) tuples representing the points
        if self._num_shifts == 1:
            print('Provide a num_shifts number greater than 1')
        elif self._num_shifts == 2:
            self._circ_points = np.array([[self._dxdy,0],[0,self._dxdy]])
        else:
            self._circ_points = np.array([(x_ang[i], y_ang[i]) for i in range(self._num_shifts)]) * self._dxdy
        return 
        ###############
        ###############
        ###############
        ###############

    #Perform DAC on the randomly generated distortion
    def _random_dist(self,x,y):
        if x is None or y is None:
            x=self._x_flattened
            y=self._y_flattened
        random_dist = self._gen_dist()
        random_dist_x = random_dist[:,0].reshape(self._dist_samp,self._dist_samp)
        random_dist_y = random_dist[:,1].reshape(self._dist_samp,self._dist_samp)

        # Define the interpolation function using RectBivariateSpline
        dist_func_x = interpolate.RectBivariateSpline(self._x_rand, self._y_rand, random_dist_x,kx=1,ky=1)
        dist_func_y = interpolate.RectBivariateSpline(self._x_rand, self._y_rand, random_dist_y,kx=1,ky=1)

        #Home Distortions
        spatial_dist_h = np.c_[(dist_func_x(y,x,grid=False)), 
                               (dist_func_y(y,x,grid=False))]

        #Shifted Distortions
        shifted_dist = np.zeros([len(self._circ_points),np.shape(spatial_dist_h)[0],np.shape(spatial_dist_h)[1]])
  
        for i in range(len(self._circ_points)):
            x_points = x + self._circ_points[i][0]
            y_points = y + self._circ_points[i][1]
            shifted_dist[i] = np.c_[dist_func_x(y_points,x_points,grid=False),dist_func_x(y_points,x_points,grid=False)]
        
          
        home_meas = np.c_[x,y] + spatial_dist_h
        shiftxy_meas = shifted_dist + np.c_[x,y] + self._circ_points[:,None]

        #Perform DAC
        home1_distx, home1_disty = self._sim.recovered_dist_manual(x,y,home_meas,shiftxy_meas,
                                                             dxy_meas=self._circ_points,n_poly=self._N_poly,r=30)
        self._input_dist = spatial_dist_h
        self._home1_dist = np.c_[home1_distx, home1_disty]
        self._u_hat = self._sim._u_hat
        print('Input spatial Distortion RMS:', np.std(spatial_dist_h))
        print('Fitted Distortion RMS:',np.std(self._home1_dist))
        print('Residual Distortionh RMS:',np.std(spatial_dist_h - self._home1_dist))
        return 
    

    #Distortion function based on the DAC coefficients
    def _u_hat_dist_create(self,xx,yy):
        def _recovered_distortions_ana(N):
            n_tot_poly = ((N+1)*(N+2))//2-1
            return lambda x,y: np.c_[self._sim._hbvpoly(np.c_[x,y],self._u_hat[:n_tot_poly]),
                                     self._sim._hbvpoly(np.c_[x,y],self._u_hat[n_tot_poly:])]
        out_x = xx*0
        out_y = xx*0
        for i in range(xx.shape[0]):
            func_handle = _recovered_distortions_ana(self._N_poly)
            out_x[i],out_y[i] = func_handle(xx[i],yy[i])[0]
        return np.c_[out_x,out_y]
    
    def do_uhat_dist(self,x=None,y=None,plot='no'):
        if x is None or y is None:
            x=self._x_flattened
            y=self._y_flattened

        self._do_shifts()
        #Obtain the u_coeffs after running DAC once.
        self._random_dist(x,y)
        print('...')
        print('Reperforming DAC using the calculated polynomial coefficients as the distortion:')
        self._u_hat_dist_h = self._u_hat_dist_create(x,y)
        u_hat_dist_shifted = np.zeros([len(self._circ_points),self._u_hat_dist_h.shape[0],self._u_hat_dist_h.shape[1]])
        for j in range(len(self._circ_points)):
            xx_points = x + self._circ_points[j][0]
            yy_points = y + self._circ_points[j][1]
            u_hat_dist_shifted[j] = self._u_hat_dist_create(xx_points,yy_points)
        
        self._uhat_home_meas = np.c_[x,y] + self._u_hat_dist_h + np.random.randn(*self._u_hat_dist_h.shape)*self._centroid_noise_std
        self._uhat_shiftxy_meas = u_hat_dist_shifted + np.c_[x,y] + self._circ_points[:,None] + np.random.randn(*self._u_hat_dist_h.shape)*self._centroid_noise_std

        #Perform DAC
        uhat_home_distx, uhat_home_disty = self._sim.recovered_dist_manual(x,y,self._uhat_home_meas,self._uhat_shiftxy_meas,self._circ_points,
                                                                           n_poly=self._N_poly,r=30)
        self._uhat_home_dist = np.c_[uhat_home_distx, uhat_home_disty]

        print('Input U_hat Distortion RMS:', np.std(self._u_hat_dist_h))
        print('Fitted Distortion RMS:',np.std(self._uhat_home_dist))
        print('Residual Distortionh RMS:',np.std(self._uhat_home_dist - self._u_hat_dist_h))
        
        pos_res = self._uhat_home_meas - self._uhat_home_dist - np.c_[x,y]
        print('Position recovery for Home Position Residual RMS:',np.std(pos_res))

    
        if plot=='yes':
            print('Plotting 1...')
            plt.figure(figsize=[8,8])
            arrow_sf = 300
            Distortion_scale = np.sqrt(np.mean(self._uhat_home_dist[:,0]**2+self._uhat_home_dist[:,1]**2))*1000
            for i in range(len(x)):
                plt.arrow(x[i],y[i],arrow_sf*self._input_dist[:,0][i],arrow_sf*self._input_dist[:,1][i],
                          color="b",head_width=0.15,width=0.01,length_includes_head=True)
                plt.arrow(x[i],y[i],arrow_sf*self._home1_dist[:,0][i],arrow_sf*self._home1_dist[:,1][i],
                          color="r",head_width=0.15,width=0.01,length_includes_head=True)
                plt.axis("square")
            plt.legend(["random distortion","fitted distortion"])
            plt.title(f"Field Distortion with Random Distortion\n(arrows scaled {arrow_sf:0.0f}x)\n(Distortion RMS {Distortion_scale*1000:0.02f}uas)")
            plt.xlabel("x-position in field [arcsec]")
            plt.ylabel("y-position in field [arcsec]")
            plt.show()

            print('Plotting 2...')
            plt.figure(figsize=[8,8])
            arrow_sf = 300
            Distortion_scale = np.sqrt(np.mean(self._uhat_home_dist[:,0]**2+self._uhat_home_dist[:,1]**2))*1000
            for i in range(len(x)):
                plt.arrow(x[i],y[i],arrow_sf*uhat_home_distx[i],arrow_sf*uhat_home_disty[i],
                        color="g",head_width=0.35,width=0.01,length_includes_head=True)
                plt.arrow(x[i],y[i],arrow_sf*self._home1_dist[:,0][i],arrow_sf*self._home1_dist[:,1][i],
                          color="r",head_width=0.15,width=0.01,length_includes_head=True)
                plt.axis("square")
            plt.legend(["polynomial distortion","fitted distortion"])
            plt.title(f"Field Distortion with Polynomial Distortion\n(arrows scaled {arrow_sf:0.0f}x)\n(Distortion RMS {Distortion_scale*1000:0.02f}uas)\n(Residual RMS {np.std(self._uhat_home_dist - self._u_hat_dist_h)}uas)")
            plt.xlabel("x-position in field [arcsec]")
            plt.ylabel("y-position in field [arcsec]")
            plt.show()
        return 
    
    def do_uhat_interp(self,x=None,y=None,plot='no'):
        if x is None or y is None:
            x=self._x_flattened
            y=self._y_flattened
        self.do_uhat_dist(x,y,plot)

        print('...')
        print('Reperforming DAC using the calculated polynomial coefficients as the distortion but with the interpolation method:')

        u_hat_interp_dist_gen = self._u_hat_dist_create(self._x_rand_flattened,self._y_rand_flattened)
        u_hat_dist_gen_xreshape = u_hat_interp_dist_gen[:,0].reshape(self._dist_samp,self._dist_samp)
        u_hat_dist_gen_yreshape = u_hat_interp_dist_gen[:,1].reshape(self._dist_samp,self._dist_samp)


        # Define the interpolation function using RectBivariateSpline
        uhat_dist_func_x = interpolate.RectBivariateSpline(self._x_rand, self._y_rand, u_hat_dist_gen_xreshape,kx=1,ky=1)
        uhat_dist_func_y = interpolate.RectBivariateSpline(self._x_rand, self._y_rand, u_hat_dist_gen_yreshape,kx=1,ky=1)

        #Home Distortion
        #uhat_interp_dist_h = np.c_[(uhat_dist_func_x(x_linspace,y_linspace)).flatten(), 
        #                        (uhat_dist_func_y(x_linspace,y_linspace)).flatten()]
        uhat_interp_dist_h = np.c_[uhat_dist_func_x(y,x,grid=False), 
                                   uhat_dist_func_y(y,x,grid=False)]
        #Shifted Distortions
        uhat_interp_shifted_dist = np.zeros([len(self._circ_points),np.shape(uhat_interp_dist_h)[0],np.shape(uhat_interp_dist_h)[1]])

        
        for k in range(len(self._circ_points)):
            #x_points = x_linspace + circ_points[k][0]
            #y_points = y_linspace + circ_points[k][1]
            x_points = x + self._circ_points[k][0]
            y_points = y + self._circ_points[k][1]
            uhat_interp_shifted_dist[k] = np.c_[uhat_dist_func_x(y_points,x_points,grid=False),
                                                uhat_dist_func_y(y_points,x_points,grid=False)]
        
        
        uhat_interp_home_meas = np.c_[x,y] + uhat_interp_dist_h
        uhat_interp_shiftxy_meas = uhat_interp_shifted_dist + np.c_[x,y] + self._circ_points[:,None]
        print('Input U_hat Interpolated Distortion RMS:', np.mean(np.abs(uhat_interp_dist_h)))

        #Perform DAC
        uhat_interp_home_distx, uhat_interp_home_disty = self._sim.recovered_dist_manual(x,y,
                                                                                         uhat_interp_home_meas,
                                                                                         uhat_interp_shiftxy_meas,
                                                                                         dxy_meas=self._circ_points,n_poly=self._N_poly,r=30)
        
        
        print('Fitted Distortion RMS:',np.std(np.c_[uhat_interp_home_distx,uhat_interp_home_disty]))
        print('Residual Distortionh RMS:',np.std(np.c_[uhat_interp_home_distx,uhat_interp_home_disty]-uhat_interp_dist_h))

        pos_res2 = uhat_interp_home_meas - np.c_[uhat_interp_home_distx, uhat_interp_home_disty] - np.c_[x,y]
        print('Position recovery for Home Position Residual RMS:',np.std(pos_res2))
        return 