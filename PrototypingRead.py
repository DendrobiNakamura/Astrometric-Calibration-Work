import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit


class AstrocalibRead():
    def __init__(self,dataframe,corners_home,mode,peak='yes',bright_wind_size=50,calc_wind_size_C=20,calc_wind_size_G=10):
        ''' dataframe: image data in the following format, contains 3 datasets and an approximated dx,dy
                       pd.dataframe({'home':
                                     'shiftx':
                                     'shifty':
                                     'dx':
                                     'dy':})
                                     
            corners_home: contains 'data' and 'dimension'
                        data: locations of the corners of the 'home' dataset in the format
                                list:[[..,..],[..,..],[..,..],[..,..]] in the order
                                top-left, top-right, bottom-left, bottom-right where the origin is at bottom-left
                        dimension: the dimension of the pinholes enclosed by the corners.[no. rows, no. columns]

            mode: either as "C" or "G" for calculating pinhole positions using Centroids or Gaussian fitting.

            peak: whether or not to use the brightest pixel as the nominal position when windowing.

            bright_wind_size: window radius to determine the brightest pixel from the approximate position
            calc_wind_size: window radius to calculate the exact pinhole position using either Gaussian (G) or centroid (C)
        '''
        self._corners_home=corners_home['data'][0]
        self._row_num=corners_home['dimension'][0]
        self._col_num=corners_home['dimension'][0]
        self._dataframe=dataframe
        self._home=dataframe['home']
        self._shiftx=dataframe['shiftx']
        self._shifty=dataframe['shifty']
        self._dx_approx=corners_home['dx'][0]
        self._dy_approx=corners_home['dy'][0]
        self._bright_wind_size=bright_wind_size
        self._calc_wind_size_C=calc_wind_size_C
        self._calc_wind_size_G=calc_wind_size_G
        self._mode=mode
        self._peak_arg=peak

        #Corners
        x_corners = np.array([[1,0],[1,0],[1,0],[1,0]])
        y_corners = np.array([[0,1],[0,1],[0,1],[0,1]])

        # Manually add the shift
        # Want to know the locations of the corners in shiftx/shiftx where the pinhole grid is spatially shifted
        self._corners_shift_x = self._corners_home+self._dx_approx*x_corners
        self._corners_shift_y = self._corners_home+self._dy_approx*y_corners


    #This function generates a grid of pinhole positions based on the locations of the corners
    # The generated pinhole positions takes into account of any net rotation of the grid
    def _Generate_Grid(self,corners):
        """Generate a meshgrid and rotate it by RotRad radians."""
        x_lower_av = (corners[0][0]+corners[2][0])/2
        x_upper_av = (corners[1][0]+corners[3][0])/2
        y_lower_av = (corners[2][1]+corners[3][1])/2
        y_upper_av = (corners[0][1]+corners[1][1])/2

        centre = [(x_lower_av+x_upper_av)/2,(y_lower_av+y_upper_av)/2] #need to shift the centre to 0 so that the rotation centres about the centre 

        x,y = np.meshgrid(np.linspace(x_lower_av-centre[0],x_upper_av-centre[0],self._col_num)
                          ,np.linspace(y_lower_av-centre[1],y_upper_av-centre[1],self._row_num))

        centre = [(x_lower_av+x_upper_av)/2,(y_lower_av+y_upper_av)/2]

        #calculate required rotation angle
    
        angle=[]
        angle.append(np.arctan((corners[0][0]-corners[2][0])/2/np.abs((corners[0][1]-corners[2][1])/2)))
        RotRad=np.mean(angle)
        
        # Clockwise, 2D rotation matrix
        RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                            [-np.sin(RotRad), np.cos(RotRad)]])
        #return x,y
        grid = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x, y]))
        rot_x = grid[0]+centre[0] #returns to original position
        rot_y = grid[1]+centre[1]
        return rot_x.flatten(),rot_y.flatten() 
    

    def _make_hex_grid(self,p1,p2,lim):
        """
        First create a hexagonal grid and then rotate it to match the input points.
        p1:     [..,..]
        p2:     [..,..]        The adjacent shortest point to p1
        lim:    [0,4096]
        """

        point_spacing = ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5
        mid_point = [(p1[0]+p2[0])/2 , (p1[1]+p2[1])/2]


        y_small = mid_point[1]-point_spacing/2     #first point of y in smaller direction
        ny = math.floor(np.sqrt(2)*np.max(lim)/point_spacing)     #amount of points needed to reach the limit.
        y_min_point = y_small-point_spacing*(ny)    #smallest y-valued point
        num_points = ny*2+2                # Number of points in y


        x_spacing = np.sqrt(point_spacing**2-(point_spacing/2)**2)

        x_mid = mid_point[0]
        nx = math.floor(np.sqrt(2)*np.max(lim)/x_spacing) 
        x_min_point = x_mid-x_spacing*(nx+(nx % 2)) #Add an extra row if the input points exist in a 'middle' row


        # Create arrays to store x and y coordinates of points
        x_coords = []
        y_coords = []


        for row in range(num_points):
            for col in range(math.ceil(num_points*point_spacing/x_spacing)): #so that the output looks like a square grid
                x = x_min_point + col * x_spacing
                y = y_min_point + row * point_spacing + (col % 2) * point_spacing / 2
                x_coords.append(x)
                y_coords.append(y)


        x_reshaped = np.array(x_coords).reshape([num_points,math.ceil(num_points*point_spacing/x_spacing)]) - mid_point[0]
        y_reshaped = np.array(y_coords).reshape([num_points,math.ceil(num_points*point_spacing/x_spacing)]) - mid_point[1]

        sign=1
        if p1[1]-p2[1]<0:
            sign= -1
        RotRad = np.arctan((p1[0]-p2[0])/2/np.abs((p1[1]-p2[1])/2))*sign


        # Clockwise, 2D rotation matrix
        RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                            [-np.sin(RotRad), np.cos(RotRad)]])
        #return x,y
        grid = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([x_reshaped, y_reshaped]))
        rot_x = (grid[0]+mid_point[0]).flatten() #returns to original position
        rot_y = (grid[1]+mid_point[1]).flatten()

        mask = (0 < rot_x) & (rot_x < lim[0]) & (0 < rot_y) & (rot_y < lim[1])
        
        return rot_x[mask], rot_y[mask]
    
    # Find the location of the brightest pixel of each pinhole
    # The output value provides a centre for each pinhole window
    def _peak(self,data):
        argmax = np.argmax(data)
        row_ind = int(argmax/(data.shape[0]))
        column_ind = argmax-data.shape[0]*row_ind
        return row_ind,column_ind

    def _peak_locations(self,im,location):
        position=[]
        for loc_x,loc_y in location:
            x_min = loc_x-self._bright_wind_size
            x_max = loc_x+self._bright_wind_size
            y_min = loc_y-self._bright_wind_size
            y_max = loc_y+self._bright_wind_size

            row,column = self._peak(im[y_min:y_max+1,x_min:x_max+1])
            position.append([x_min+column,y_min+row])
        return position
    

    # This centroids function computes the centre of mass of the pinhole window.
    def _centroids(self,im,pos):
        """Centroids function"""
        # iterate over images:
        position=[]
        for images in im:
            cog_meas = []
            # iterate over pinholes:
            for pin_x,pin_y in pos:
                # integer-valued indices for desired pinhole-window
                win_idx = np.mgrid[ int((pin_x-self._calc_wind_size_C)):
                                    int((pin_x+self._calc_wind_size_C))+1,
                                    int((pin_y-self._calc_wind_size_C)):
                                    int((pin_y+self._calc_wind_size_C))+1]
                win_idx = np.clip(win_idx,0,images.shape[0]-1)
                #print('win_idx is',win_idx)
                # corresponding pixel coordinates within that window
                win_as = [w.flatten() for w in win_idx]
                #print('win_as is', win_as)
                # pixel intensities of window, flattened for centroiding
                window = images[win_idx[1],win_idx[0]].flatten()
                #print('window is',window)
                # centroid calculation
                cog_meas.append([(window @ p) / window.sum() for p in win_as])
                #print(cog_meas[-1])
                #plt.plot(images[y_min:y_max+1,x_min:x_max+1])
            position.append(np.r_[cog_meas])
        return position
    
    # These next two functions tries to fit a 2D Gaussian to the pinhole, and hence obtain the position parameter of the fitted value.
    def _twoD_Gaussian(self,xy, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        """Defines a 2D Gaussian used for optimization"""
        x, y = xy
        xo = float(xo)
        yo = float(yo)    
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                + c*((y-yo)**2)))
        return g.ravel()

    def _Do_Gaussian_Fit(self,im,location,plot='yes',amp=1000,offset=400,sigx=1,sigy=1):
        #im is the list of images
        #location is a list of locations
        #window_size is the 2xwindow_size x window_size window around the nominated location (win_rad)
        #plots the last fit in the list
        position=[]
        error=[]
        opt=[]
        #i=0
        for images in im:
            ind_position=[]
            ind_err=[]
            optopt=[]
            for loc_x,loc_y in location:
                x_min = loc_x-self._calc_wind_size_G
                x_max = loc_x+self._calc_wind_size_G
                y_min = loc_y-self._calc_wind_size_G
                y_max = loc_y+self._calc_wind_size_G
                #make grid
                # Create x and y indices
                x = np.linspace(x_min, x_max, x_max-x_min+1)
                y = np.linspace(y_min, y_max, y_max-y_min+1)
                x, y = np.meshgrid(x, y)

                data = images[y_min:y_max+1,x_min:x_max+1].ravel()

                initial_guess = (amp,loc_x,loc_y,sigx,sigy,0,offset)
                #print(i)
                #i+=1
                popt, pcov = curve_fit(self._twoD_Gaussian, (x, y), data, p0=initial_guess,maxfev=5000)
                ind_position.append([popt[1],popt[2]])
                xerr = np.sqrt(np.diag(pcov[1]))
                yerr = np.sqrt(np.diag(pcov[1]))
                ind_err.append([xerr,yerr])
                optopt.append(popt)
            position.append(ind_position)
            opt.append(optopt)
            error.append(ind_err) #this is the error from the fitting

        if plot == 'yes':
            data_fitted = self._twoD_Gaussian((x, y), *popt)
            fig, ax = plt.subplots(1, 1)
            ax.imshow(data.reshape(x_max-x_min+1, y_max-y_min+1), cmap=plt.cm.jet, origin='lower',
                extent=(x.min(), x.max(), y.min(), y.max()))
            ax.contour(x, y, data_fitted.reshape(x_max-x_min+1, y_max-y_min+1), 8, colors='w')
            plt.axis('off')
            #plt.savefig('../../../Dist_good.pdf')
            plt.show()

        #print('Opt: amp,x,y,sigx,sigy,ignore,offset') 
        return position,error,opt

    # Function to measure the pinhole positions using the above functions
    def _measure_positions(self,data,corners):
        """location of corners of the grid: example [[1460,2558],[2565,2563],[1468,1458],[2570,1461]]
            #order being top left, top right, bottom left, bottom right"""
        #e.g., 10x12 grid: row_num=12, column_num=10
        x,y=self._Generate_Grid(corners) #creates grid
        location = np.int0(np.c_[x,y])
        self._location = location.copy()
        loc_peak = self._peak_locations(data[0],location)   #find the location of the peak (max) at each window
        self._loc_peak=loc_peak.copy()
        if self._mode=='G':
            if self._peak_arg=='no':
                return self._Do_Gaussian_Fit(data,location,amp=1000,offset=400,sigx=1,sigy=1)
            else:
                return self._Do_Gaussian_Fit(data,loc_peak,amp=1000,offset=400,sigx=1,sigy=1)   #perform Gaussian
        elif self._mode == 'C':
            if self._peak_arg=='no':
                return self._centroids(data,location)
            else:
                return self._centroids(data,loc_peak)
        else:
            print('Please enter mode="Gaussian" or mode="Centroid".')


    #Data usually comes in a set of 10-repeated measurements. Want to find the mean and std of the 10 measurements.
    def _av_std(self,positions):
        """finds the average and standard deviation of the positions of a set of repeated exposures"""
        av_positions = np.sum(positions,axis=0)/len(positions)
        positions_diff = np.array(positions)
        for i in range(len(positions)):
            positions_diff[i] = positions[i]-av_positions
        std_positions = np.sqrt(np.sum(positions_diff**2,axis=0)/(len(positions)))
        return av_positions, std_positions
    
    # Find the average translation and zoom of each image with respect to the average and then remove the translation/zoom for each image
    def _data_transform(self,data):
        translation = []
        for i in range(len(data)):
            translation.append(data[i]-self._av_std(data)[0])
        mean_trans = np.mean(translation,axis=1)

        trans_data = data.copy()
        for i in range(len(trans_data)):
            trans_data[i] = trans_data[i] - mean_trans[i]

        zoom = []
        for i in range(len(trans_data)):
            zoom.append(trans_data[i]/self._av_std(trans_data)[0])
        mean_zoom = np.mean(zoom,axis=1)

        zoomed_trans_data = trans_data.copy()
        for i in range(len(zoomed_trans_data)):
            zoomed_trans_data[i] = trans_data[i]/mean_zoom[i]

        return zoomed_trans_data


    #Data is in pixels x:[0:4096],y:[0:4096]. Want data to be centered at 0,0 in 30x30".
    def _scale_it(self,home,shifted_x, shifted_y, scale=30/4096,mode='camera'):
        if mode=='camera':
            x_centre = 2048
            y_centre = 2048
        elif mode=='image':
            x_centre, y_centre = np.mean(home,axis=0)        #find the centre of the grid
        else:
            print('mode = camera or image')    
        home_scaled=scale*(home - np.r_[x_centre,y_centre])
        shiftx_scaled=scale*(shifted_x - np.r_[x_centre,y_centre])
        shifty_scaled=scale*(shifted_y - np.r_[x_centre,y_centre])
        return home_scaled,shiftx_scaled,shifty_scaled
    

 #Outputs the measured positions after scaling and averaging.
    def position_output(self):
        positions_home = self._measure_positions(self._home,self._corners_home)
        self._nom_pos_h = self._location.copy()
        positions_shiftx = self._measure_positions(self._shiftx,self._corners_shift_x)
        self._nom_pos_x = self._location.copy()
        positions_shifty = self._measure_positions(self._shifty,self._corners_shift_y)
        self._nom_pos_y = self._location.copy()

        #subtract/divide off the average random transition/zooom
        self._positions_home = self._data_transform(positions_home)
        self._positions_shiftx = self._data_transform(positions_shiftx)
        self._positions_shifty = self._data_transform(positions_shifty)
        #scale them
        H,X,Y = self._scale_it(self._positions_home,self._positions_shiftx,self._positions_shifty)

        #Average them
        self._home_pos,self._home_pos_std = self._av_std(H)
        self._shiftx_pos,self._shiftx_pos_std = self._av_std(X)
        self._shifty_pos,self._shifty_pos_std = self._av_std(Y)
        return

###############################################################################
###############################################################################
############################################################################### 
# This section is for the purpose of plotting when investigating different parts of the code.

    def _reshape(self,data):
        """array([[x,y],[x,y],]x,y],.....]) --> [matrix(x),matrix(y)]"""
        return data.ravel(order='F').reshape(2,self._col_num,self._row_num)
    
    def _deshape(self,data):
        """Opposite of _reshape"""
        x=data[0]
        y=data[1]
        return np.c_[x.flatten(),y.flatten()]
    
    def plot_dxdy(self):
        residual_x = (self._shiftx_pos-self._home_pos - np.r_[self._dx_meas_x,0])
        residual_y = (self._shifty_pos-self._home_pos - np.r_[0,self._dy_meas_y])
        res_X_dx = self._reshape(residual_x)[0]
        res_Y_dx = self._reshape(residual_x)[1]
        res_X_dy = self._reshape(residual_y)[0]
        res_Y_dy = self._reshape(residual_y)[1]



        fig = plt.figure(figsize=[10, 10])
        gs = fig.add_gridspec(2, 2, hspace=0.1, wspace=0.1)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax4 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax3)

        p1 = ax1.pcolor(np.linspace(-14, 14, self._col_num), np.linspace(-14, 14, self._col_num), res_X_dx * 1000)
        ax1.set_title(f'x residual from dx shift\n(Std {np.std(res_X_dx*1000):0.2f} mas)')
        fig.colorbar(p1, ax=ax1, label='mas')

        p2 = ax2.pcolor(np.linspace(-14, 14, self._col_num), np.linspace(-14, 14, self._col_num), res_Y_dx * 1000)
        ax2.set_title(f'y residual from dx shift\n(Std {np.std(res_Y_dx*1000):0.2f} mas)')
        fig.colorbar(p2, ax=ax2, label='mas')

        p3 = ax3.pcolor(np.linspace(-14, 14, self._col_num), np.linspace(-14, 14, self._col_num), res_X_dy * 1000)
        ax3.set_title(f'x residual from dy shift\n(Std {np.std(res_X_dy*1000):0.2f} mas)')
        fig.colorbar(p3, ax=ax3, label='mas')

        p4 = ax4.pcolor(np.linspace(-14, 14, self._col_num), np.linspace(-14, 14, self._col_num), res_Y_dy * 1000)
        ax4.set_title(f'y residual from dy shift\n(Std {np.std(res_Y_dy*1000):0.2f} mas)')
        fig.colorbar(p4, ax=ax4, label='mas')

        for ax in fig.get_axes():
            ax.set_aspect('equal', adjustable='box')  # Set aspect ratio to make subplots square
            ax.label_outer()

        plt.show()

        return

    def plot_pos(self,xmin,xmax,ymin,ymax,image='home'):
        """
        Want to plot the measured positions with the raw image given a small window
        xmin,xmax,ymin,ymax are the indicies of the positions intending to be plotted.
        """

        # Create a mask and then apply the mask to the datasets 

        mask = np.zeros([self._col_num,self._row_num])
        mask[:]=np.nan
        mask[xmin:xmax,ymin:ymax] = 1
        #print(mask)
        filt = np.c_[mask.flatten(),mask.flatten()]
        #print(filt)
        initial_location_h = self._nom_pos_h * filt
        initial_location_x = self._nom_pos_x * filt
        initial_location_y = self._nom_pos_y * filt
        
        home = self._av_std(self._positions_home)[0] * filt
        shiftx = self._av_std(self._positions_shiftx)[0] * filt
        shifty = self._av_std(self._positions_shifty)[0] * filt

        xmin_im = int(np.nanmin([home[:,0],shiftx[:,0],shifty[:,0]]))-50
        xmax_im = int(np.nanmax([home[:,0],shiftx[:,0],shifty[:,0]]))+50
        ymin_im = int(np.nanmin([home[:,1],shiftx[:,1],shifty[:,1]]))-50
        ymax_im = int(np.nanmax([home[:,1],shiftx[:,1],shifty[:,1]]))+50

        
        
        plt.figure(figsize=[12,12])
        plt.plot(home[:,0],home[:,1],"r*")
        plt.plot(shiftx[:,0],shiftx[:,1],"b*")
        plt.plot(shifty[:,0],shifty[:,1],'k*')
        
        #plt.title('No Distortion (Corner)')
        if image=='home':
            plt.imshow(self._home[0],vmax=2000)
            plt.plot(initial_location_h[:,0],initial_location_h[:,1],'g*')
        if image=='shiftx':
            plt.imshow(self._shiftx[0],vmax=2000)
            plt.plot(initial_location_x[:,0],initial_location_x[:,1],'g*')
        if image=='shifty':
            plt.imshow(self._shifty[0],vmax=2000)
            plt.plot(initial_location_y[:,0],initial_location_y[:,1],'g*')
        plt.legend(['Home','Shift X','Shift Y','Nom Pos'])
        plt.xlim([xmin_im,xmax_im])
        plt.ylim([ymin_im,ymax_im])
        plt.show()
        return 


    def plot_std(self):
        """
        A colormap plotting the values of the std after averaging over the 10 repeated measurements
        """
       
        C_H=np.sqrt(self._reshape(self._home_pos_std)[0]**2+self._reshape(self._home_pos_std)[1]**2)
        C_X=np.sqrt(self._reshape(self._shiftx_pos_std)[0]**2+self._reshape(self._shiftx_pos_std)[1]**2)
        C_Y=np.sqrt(self._reshape(self._shifty_pos_std)[0]**2+self._reshape(self._shifty_pos_std)[1]**2)


        av_H = np.mean(np.mean(C_H))*1000
        av_X = np.mean(C_X)*1000
        av_Y = np.mean(C_Y)*1000
        if self._mode=='C':
            label='Centroid'

        if self._mode=='G':
            label='Gaussian'
        
        plt.figure()
        plt.pcolor(np.linspace(-14,14,self._col_num),np.linspace(-14,14,self._row_num),C_H*1000)
        plt.colorbar(label='mas')
        plt.title(f'Home\n(Average {av_H:0.2f} mas)')
        plt.legend([label])
        plt.xlabel('FoV (arcsec)')
        plt.ylabel('FoV (arcsec)')
        plt.axis('square')
        plt.show()
        plt.figure()
        plt.pcolor(np.linspace(-13,13,37),np.linspace(-13,13,37),C_X*1000)
        plt.colorbar(label='mas')
        plt.title(f'Shift X\n(Average {av_X:0.2f} mas)')
        plt.legend([label])
        plt.xlabel('FoV (arcsec)')
        plt.ylabel('FoV (arcsec)')
        plt.axis('square')
        plt.show()
        plt.figure()
        plt.pcolor(np.linspace(-13,13,37),np.linspace(-13,13,37),C_Y*1000)
        plt.colorbar(label='mas')
        plt.title(f'Shift Y\n(Average {av_Y:0.2f} mas)')
        plt.legend([label])
        plt.xlabel('FoV (arcsec)')
        plt.ylabel('FoV (arcsec)')
        plt.axis('square')
        plt.show()
        return
    
    def plot_scatter(catalogue,data,sort='no',style='.'):
        if sort=='yes':
            positions = data
        else:
            positions = catalogue._positions_home
        std=[]
        n=[]
        for i in range(len(positions)):
            stnd = catalogue._av_std(positions[:i+1])[1]
            std.append(np.sqrt(np.sum(stnd[:,0]**2+stnd[:,1]**2)/len(stnd)))
            n.append(i+1)
        plt.figure()
        plt.loglog(n,std,style)                 #standard deviation of the positions over 10 repeated exposures
        plt.xlabel('number of samples')
        plt.ylabel('Standard Deviation of positions')
        plt.show()
        return

###############################################################################
###############################################################################
############################################################################### 
# This section performs astrometric calibration calculation


    def _hbvpoly(self,p,a):
        """ Evaluate the homogenous bi-variate polynomial defined by 
        coefficients in a at position p.
        
        Arguments:
            p: np.ndarray : position to evaluate polynomial at, (M,2)
            a: np.ndarray : coefficients defining polynomial, (((N+1)(N+2))//2-1,)
            N: int: maximum homogenous polynomial order to go to.
        
        Returns:
            out: np.ndarray : evaluated polynomial, scalar or (M,1)
        """
        if len(p.shape)!=2:
            raise ValueError("p must be 2D, i.e., p.shape=(M,2)")
        out = np.zeros_like(p[:,0])
        counter = 0
        #print(len(a))
        for n in range(1,self._n_poly+1): # skip tip-tilt
            for j in range(n+1):
                out[:] += a[counter]*p[:,0]**j*p[:,1]**(n-j)/np.math.factorial(n)
                counter += 1
        return out



    def _hbvpoly_grad(self,p):
        """ Evaluate the gradient of the homogenous bi-variate polynomial 
        defined by coefficients in a at position p.
        
        Arguments:
            p: np.ndarray : position to evaluate polynomial gradient at, (2,) or (M,2)
            n_poly: int: maximum homogenous polynomial order to go to.

        Returns:
            out: np.ndarray : evaluated polynomial gradient,
        
        """
        dx = np.zeros([p.shape[0],((self._n_poly+1)*(self._n_poly+2))//2-1])
        dy = np.zeros([p.shape[0],((self._n_poly+1)*(self._n_poly+2))//2-1])
        counter = -1
        for n in range(1,self._n_poly+1): # skip tip-tilt
            for j in range(n+1):
                counter += 1
                if j==0:
                    continue
                dx[:,counter] += j*p[:,0]**(j-1)*p[:,1]**(n-j)/np.math.factorial(n)

        counter = -1
        for n in range(1,self._n_poly+1): # skip tip-tilt
            for j in range(n+1):
                counter += 1
                if j==n:
                    continue
                dy[:,counter] += (n-j)*p[:,0]**j*p[:,1]**(n-j-1)/np.math.factorial(n)
        return dx,dy
    

    def _fit_poly(self,p0_meas,ppx_meas,ppy_meas):
        """
            - p0_nom is the home position
            - ppx_meas is the moved position after moving by x
            - ppy_meas is the moved position after moving by y
            - n_poly is the degree of polynomial accuracy
            - dx_meas is the amount of dx movement
            - dy_meas is the amount of dy movement
        Returns:
            _type_: _description_
        """
        
        n_tot_poly = ((self._n_poly+1)*(self._n_poly+2))//2-1

        n_pos = len(p0_meas)

        d_mat = np.zeros([4*n_pos,2*n_tot_poly])
        grad_tmp = self._hbvpoly_grad(p0_meas)

        d_mat[0::4,:n_tot_poly]   = grad_tmp[0]
        d_mat[1::4,n_tot_poly:]   = grad_tmp[0]
        d_mat[2::4,:n_tot_poly]   = grad_tmp[1]
        d_mat[3::4,n_tot_poly:]   = grad_tmp[1]

        d_inv = np.linalg.solve(d_mat.T@d_mat,d_mat.T)

        dx_meas = self._dx_meas_x
        dy_meas = self._dy_meas_y
        dx_meas = 1.7182130584192439
        dy_meas = 1.7182130584192439
        # component-wise gradients:
        #dpdx = (ppx_meas-p0_meas) - np.r_[dx_meas_x,0]
        #dpdx /= dx_meas_x
        #dpdy = (ppy_meas-p0_meas) - np.r_[0,dy_meas_y]
        #dpdy /= dy_meas_y
        
        dpdx = (ppx_meas-p0_meas) - np.r_[dx_meas,0]
        #dpdx /= np.sqrt(dx_meas_x**2+dx_meas_y**2)
        dpdx /= dx_meas
        dpdy = (ppy_meas-p0_meas) - np.r_[0,dy_meas]
        #dpdy /= np.sqrt(dy_meas_x**2+dy_meas_y**2)
        dpdy /= dy_meas

        # estimated gradients:
        z_hat = np.c_[dpdx,dpdy].flatten()
        
        self._z_hat=z_hat.copy()

        # estimated polynomial coefficients:
        return d_inv @ z_hat  #matrix multiplication
    

    def _fit_poly2(self,p0_meas,ppxy_meas):
        """
            - p0_nom is the home position
            - ppx_meas is the moved position after moving by x
            - ppy_meas is the moved position after moving by y
            - n_poly is the degree of polynomial accuracy
            - dx_meas is the amount of dx movement
            - dy_meas is the amount of dy movement
        Returns:
            _type_: _description_
        """
        
        n_tot_poly = ((self._n_poly+1)*(self._n_poly+2))//2-1

        n_pos = len(p0_meas)

        d_mat = np.zeros([4*n_pos,2*n_tot_poly])
        grad_tmp = self._hbvpoly_grad(p0_meas)

        d_mat[0::4,:n_tot_poly]   = grad_tmp[0]
        d_mat[1::4,n_tot_poly:]   = grad_tmp[0]
        d_mat[2::4,:n_tot_poly]   = grad_tmp[1]
        d_mat[3::4,n_tot_poly:]   = grad_tmp[1]

        d_inv = np.linalg.solve(d_mat.T@d_mat,d_mat.T)

        # component-wise gradients:
        #dpdx = (ppx_meas-p0_meas) - np.r_[dx_meas_x,0]
        #dpdx /= dx_meas_x
        #dpdy = (ppy_meas-p0_meas) - np.r_[0,dy_meas_y]
        #dpdy /= dy_meas_y


        #shift1 = ppx_meas-p0_meas-np.array([self._dx_meas_x,self._dx_meas_y])[None,:]
        #shift2 = ppy_meas-p0_meas-np.array([self._dy_meas_x,self._dy_meas_y])[None,:]
        #D_mat = np.array([shift1.flatten(),shift2.flatten()]).T


        shifts = ppxy_meas - p0_meas - self._dxy_meas[:,None]
        D_ele = []
        for i in range(len(shifts)):
            D_ele.append(shifts[i].flatten())
        D_mat = np.array(D_ele).T

        #xses = np.column_stack((shift1[:,0], shift2[:,0]))
        #yses = np.column_stack((shift1[:,1], shift2[:,1]))

        #D_mat = np.vstack((xses,yses))
        self._D_mat = D_mat.copy()
        #n_shift = 2
        #S_mat = np.zeros([2,n_shift])
        #S_mat[0,0]=self._dx_meas_x
        #S_mat[1,0]=self._dx_meas_y
        #S_mat[0,1]=self._dy_meas_x
        #S_mat[1,1]=self._dy_meas_y

        n_shift = len(self._dxy_meas)
        S_mat = np.zeros([2,n_shift])
        S_mat[0,:]=self._dxy_meas[:,0]
        S_mat[1,:]=self._dxy_meas[:,1]

        self._S_mat = S_mat.copy()

        # solve this system:
        # D = J @ S
        # D @ S.T = J    @ (S @ S.T)
        # S @ D.T =    
        J = np.linalg.solve((S_mat@S_mat.T),(S_mat@D_mat.T)).T

        #J = D_mat@np.linalg.pinv(S_mat)
        dx_x = J[0::2,0]   #dx_meas_x
        dx_y = J[1::2,0]   #dx_meas_y
        dy_x = J[0::2,1]   #dy_meas_x
        dy_y = J[1::2,1]    #dy_meas_y

        ###TODO Need to verify the above for dx_y and dy_x direction 

        self._J = J.copy()

        # estimated gradients:
        z_hat = np.c_[dx_x,dx_y,dy_x,dy_y].flatten()
        
        self._z_hat=z_hat.copy()

        # estimated polynomial coefficients:
        return d_inv @ z_hat  #matrix multiplication
    
    
    def _recovered_distortions_ana(self,p0_meas,ppxy_meas):
        n_tot_poly = ((self._n_poly+1)*(self._n_poly+2))//2-1
        u_hat = self._fit_poly2(p0_meas,ppxy_meas)
        #u_hat,dpdx,dpdy = self._fit_poly(p0_meas,ppx_meas,ppy_meas,n_poly,dx_meas_x,dy_meas_y)
        #print(u_hat)
        self._u_hat = u_hat.copy()
        return lambda x,y: np.c_[self._hbvpoly(np.c_[x,y],u_hat[:n_tot_poly]),
                                self._hbvpoly(np.c_[x,y],u_hat[n_tot_poly:])]


    def recovered_dist(self,x,y,*,n_poly=6,r=15):
        """Evaluate recovered/estimated input distortions at arbitrary coordinates.
        This is the estimated distortion via the differential calibration method
        based on the input static distortion.

        `x` and `y` (in arcsec) can be anywhere in the science field, but must
        be array-like and the same size.

        Args:
            x : array-like float : field x-coordinates (arcsec)
            y : array-like float : field y-coordinates (arcsec)
        
        Returns
            out_x : array-like float : x-component of distortion at each coord
            out_y : array-like float : y-component of distortion at each coord
        """
        self._n_poly = n_poly

        self.position_output()
        x = np.array(x).copy()
        y = np.array(y).copy()
        xx = x.flatten()
        yy = y.flatten()
        out_xx = xx*0
        out_yy = xx*0

        rrr = (self._home_pos[:,0]**2 + self._home_pos[:,1]**2)**0.5

        self._dx_meas_x = np.mean(self._shiftx_pos[rrr<=r]-self._home_pos[rrr<=r],axis=0)[0]
        self._dx_meas_y = np.mean(self._shiftx_pos[rrr<=r]-self._home_pos[rrr<=r],axis=0)[1]
        self._dy_meas_x = np.mean(self._shifty_pos[rrr<=r]-self._home_pos[rrr<=r],axis=0)[0]
        self._dy_meas_y = np.mean(self._shifty_pos[rrr<=r]-self._home_pos[rrr<=r],axis=0)[1]
    

        for i in range(xx.shape[0]):
            #func_handle = self._recovered_distortions_ana(self._home_pos,self._shiftx_pos,self._shifty_pos,n_poly,self._dx_meas_x,self._dy_meas_y)
            func_handle = self._recovered_distortions_ana(self._home_pos[rrr<=r],self._shiftx_pos[rrr<=r],self._shifty_pos[rrr<=r])
            out_xx[i],out_yy[i] = func_handle(xx[i],yy[i])[0]


        out_xx = out_xx.reshape(x.shape)
        out_yy = out_yy.reshape(x.shape)
        return out_xx,out_yy

    def recovered_dist_manual(self,x,y,home,shiftxy,dxy_meas,*,n_poly=6,r=30):
        """Evaluate recovered/estimated input distortions at arbitrary coordinates.
        This is the estimated distortion via the differential calibration method
        based on the input static distortion.

        `x` and `y` (in arcsec) can be anywhere in the science field, but must
        be array-like and the same size.

        Args:
            x : array-like float : field x-coordinates (arcsec)
            y : array-like float : field y-coordinates (arcsec)
        
        Returns
            out_x : array-like float : x-component of distortion at each coord
            out_y : array-like float : y-component of distortion at each coord
        """
        self._n_poly = n_poly

        x = np.array(x).copy()
        y = np.array(y).copy()
        xx = x.flatten()
        yy = y.flatten()
        out_xx = xx*0
        out_yy = xx*0

        rrr = (home[:,0]**2 + home[:,1]**2)**0.5

        #self._dx_meas_x = np.mean(shiftx[rrr<=r]-home[rrr<=r],axis=0)[0]
        #self._dx_meas_y = np.mean(shiftx[rrr<=r]-home[rrr<=r],axis=0)[1]
        #self._dy_meas_x = np.mean(shifty[rrr<=r]-home[rrr<=r],axis=0)[0]
        #self._dy_meas_y = np.mean(shifty[rrr<=r]-home[rrr<=r],axis=0)[1]

        self._dxy_meas = dxy_meas

        for i in range(xx.shape[0]):
            #func_handle = self._recovered_distortions_ana(self._home_pos,self._shiftx_pos,self._shifty_pos,n_poly,self._dx_meas_x,self._dy_meas_y)
            func_handle = self._recovered_distortions_ana(home[rrr<=r],shiftxy[:,rrr<=r])
            out_xx[i],out_yy[i] = func_handle(xx[i],yy[i])[0]


        out_xx = out_xx.reshape(x.shape)
        out_yy = out_yy.reshape(x.shape)
        return out_xx,out_yy




#######################################################
#######################################################
#######################################################
#Ignore for now.


    # Rotating the distortion field and coordinates

    def Rotate_Grid(self,x_coor,y_coor,RotRad):
        #positions: av_home_scaled format
        centre = np.array([np.mean(x_coor),np.mean(y_coor)]) #need to shift the centre to 0 so that the rotation centres about the centre 
        xx = x_coor-centre[0]
        yy = y_coor-centre[1]
        rr = (xx**2+yy**2)**0.5
        # use slightly smaller field because distortions arent defined beyond 30"x30" box
        #xx = xx[rr<=13.0]
        #yy = yy[rr<=13.0]

        # Clockwise, 2D rotation matrix
        RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                            [-np.sin(RotRad), np.cos(RotRad)]])
        #return x,y
        grid = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([xx, yy]))
        rot_x = (grid[0]+centre[0]).flatten() #returns to original position
        rot_y = (grid[1]+centre[1]).flatten()
        return rot_x,rot_y
 
    def Rotate_Distortion(self,x_coor,y_coor,dist_x,dist_y,RotRad):
        #positions: av_home_scaled format
        centre = np.array([np.mean(x_coor),np.mean(y_coor)]) #need to shift the centre to 0 so that the rotation centres about the centre 
        xx = x_coor-centre[0]
        yy = y_coor-centre[1]
        rr = (xx**2+yy**2)**0.5
        # use slightly smaller field because distortions arent defined beyond 30"x30" box
        #xx = xx[rr<=13.0]
        #yy = yy[rr<=13.0]

        # Clockwise, 2D rotation matrix
        RotMatrix = np.array([[np.cos(RotRad),  np.sin(RotRad)],
                            [-np.sin(RotRad), np.cos(RotRad)]])
        #return x,y
        grid = np.einsum('ji, mni -> jmn', RotMatrix, np.dstack([xx, yy]))
        rot_x = (grid[0]+centre[0]).flatten() #returns to original position
        rot_y = (grid[1]+centre[1]).flatten()
        

        #Define vectors:
        Dist_x_new = []
        Dist_y_new = []
        for i in range(len(dist_x)):
            vec = np.array([dist_x[i],dist_y[i]])
            new_vec = np.dot(RotMatrix,vec)
            Dist_x_new.append(new_vec[0])
            Dist_y_new.append(new_vec[1])

        #if RotRad==np.pi/2:
            #rot_x_arranged = np.flip(rot_x.reshape(37,37),1).transpose().flatten() #37 is the array shape
            #rot_y_arranged = np.flip(rot_y.reshape(37,37),1).transpose().flatten()
            #Dist_x_arranged = np.flip(np.array(Dist_x_new).reshape(37,37),1).transpose().flatten()
            #Dist_y_arranged = np.flip(np.array(Dist_y_new).reshape(37,37),1).transpose().flatten()
            #return rot_x_arranged,rot_y_arranged,Dist_x_arranged,Dist_y_arranged

        #elif RotRad==-np.pi/2:
            #rot_x_arranged = np.flip(rot_x.reshape(37,37),0).transpose().flatten() 
            #rot_y_arranged = np.flip(rot_y.reshape(37,37),0).transpose().flatten()
            #Dist_x_arranged = np.flip(np.array(Dist_x_new).reshape(37,37),0).transpose().flatten()
            #Dist_y_arranged = np.flip(np.array(Dist_y_new).reshape(37,37),0).transpose().flatten()
            #return rot_x_arranged,rot_y_arranged,Dist_x_arranged,Dist_y_arranged
        
        #else:
            #return rot_x,rot_y,np.array(Dist_x_new),np.array(Dist_y_new)
        return rot_x,rot_y,np.array(Dist_x_new),np.array(Dist_y_new)

    