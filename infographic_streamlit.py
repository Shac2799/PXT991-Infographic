import streamlit as st
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

st.set_page_config(page_title="Gravity Simulation", page_icon=None, 
                   layout="wide", initial_sidebar_state="auto", menu_items=None) 

st.title("Simulation of gravity between bodies")

class Body:
      G = 6.67428e-11 #gravitational constant
      solar_mass = 1.98892e30 # mass of sun in kg
      danger_bound = 29920000000.0 # 1/5 of AU to sun as boundary 
      
      def __init__(self,radius,mass,centre_x,centre_y):
         
          self.radius = radius
          self.mass = mass*self.solar_mass # so user can change weight in solar mass
          self.centre_x = centre_x   # coords of centre of star/body
          self.centre_y = centre_y
          
      def density(self):
          volume = (4/3)*np.pi*self.radius**3
          return self.mass/volume     
        
      def gravity(self):          
          return (self.G*self.mass)/(self.radius**2)  # g = GM/r^2
      
        
class Object:
    AU = 149.6e9 # Astronomical units in metres
    time_interval = 86400 # number of seconds in a year
    No_danger = True # The object starts off outside of the body, if inside then this is false
    
    def __init__(self,x,y,velocity,mass,num_days):
        
        self.x = x*self.AU # converting from AU to m
        self.y = y*self.AU
        self.velocity = velocity*1000 # initial velocity in m/s
        self.mass = mass # mass in kg
        self.num_days = num_days # defining duration of graphic
        self.distance = 0
        self.path_x = [self.x/self.AU] # storing coordinates of object in number of AU
        self.path_y = [self.y/self.AU]
        self.x_vel = 0 #initialise x vel 
        self.y_vel = self.velocity # initialise y vel
        self.L = []
        self.KE,self.PE = [], []
        
    def danger_zone(self,body):
          if abs(self.x) <=  body.danger_bound and abs(self.y) <= body.danger_bound:
              self.No_danger = False
          else:
              self.No_danger = True
              
        
    def force_of_attract(self,body):  # calculates gravitational force of attraction between bodies
        pos_x = body.centre_x - self.x # coords of object relative to star/body
        pos_y = body.centre_y - self.y             
        if pos_x == 0:
            theta = np.pi/2
        else:
            theta = math.atan2(pos_y,pos_x) # atan2 goes from -pi to pi (atan only pi/2)
        distance_metres = math.sqrt(pos_x**2 + pos_y**2) # distance between body and object
        force = (body.G*self.mass*body.mass)/(distance_metres**2) # total grav force
        force_y = force*math.sin(theta)  # force in x and y directions
        force_x = force*math.cos(theta)
        self.distance = distance_metres
        self.L.append(self.mass*self.y_vel*distance_metres)
        self.KE.append(0.5*self.mass*(self.y_vel**2)) # 1/2mv^2
        self.PE.append(-force*distance_metres) # -GMm/r
        return force_x,force_y
        
    def update_path(self,body): # F = ma -> a = (v-u)/t -> v = Ft/m + u
        f_x , f_y = self.force_of_attract(body) 
        self.x_vel += (f_x/self.mass)*self.time_interval # increment of velocities due to changes in force
        self.y_vel += (f_y/self.mass)*self.time_interval
        self.x += self.x_vel*self.time_interval # increment of coords due to changes in velocities
        self.y += self.y_vel*self.time_interval
        self.path_x.append(self.x/self.AU) # storing position in array
        self.path_y.append(self.y/self.AU)
            
    def rescale_grid(self,image,x_limit,y_limit):    # 480 x 853 for stars.jpg
        height,width,_ = image.shape # dimensions of image
        image_xlim, image_ylim = [0,width], [0,height] # setting limits for grid as image dimensions
        new_centrex = width/2 # coords for centre of image
        new_centrey = height/2    
        x_scale = width/sum(abs(np.array(x_limit))) # number of pixels within the limits
        y_scale = height/sum(abs(np.array(y_limit)))
        img_posx = [(posx*x_scale + new_centrex) for posx in self.path_x] # calculating the re-scaled positions relative to image 
        img_posy = [(posy*y_scale + new_centrey) for posy in self.path_y]
        return img_posx, img_posy, image_xlim, image_ylim
      

      
    
def main():
    # choose steps/values for sliders
    mass_body = st.slider("Mass of body [Solar mass]", min_value = 1.0, max_value = 10.0, step = 0.5, value = 1.0) 
    init_vel1 = st.slider("Asteroid orbital velocity [km/s]", min_value = -30.0, max_value = 30.0, step = 5.0, value = -10.0)
    init_vel2 = st.slider("Earth orbital velocity [km/s]", min_value = -30.0, max_value = 30.0, step = 5.0, value = 30.0)
    Days = st.slider("Duration [Days]", min_value = 0.0, max_value = 5000.0, step = 5.0,value = 0.0)
    
    #initiate instances of each object/body
    Earth = Object(-1,0,init_vel2,5.97e24,Days) 
    asteroid = Object(2,0,init_vel1,2.2e14,Days)
    sun = Body(6.96e8,mass_body,0,0)
    
    #import all images for plot
    stars = mpimg.imread("stars.jpg") 
    Sun_img = mpimg.imread("Sun.png")
    Earth_img = mpimg.imread("Earth2.png")
    Asteroid_img = mpimg.imread("meteor2.png")
    height,width,_ = stars.shape # dimensions of background image
    
    #re-scaling central body's position
    sun_scaledx, sun_scaledy = width/2,height/2 # setting sun's initial position at centre of image
    x_lim = y_lim = [-3,3] # -3 to 3 AU limits
    
    #iterating through each day to update paths
    for day in range(int(Days)): # get complete array of positions of objects over x days 
        asteroid.danger_zone(sun) # checking if both objects are outside danger zone of sun 
        if asteroid.No_danger == True:
          asteroid.update_path(sun)
        else:
          break
          
          
    for day in range(int(Days)):
        Earth.danger_zone(sun)
        if Earth.No_danger == True:
          Earth.update_path(sun)
        else:
          break

    #re-scaling to image dimensions
    earth_x,earth_y,xlim,ylim = Earth.rescale_grid(stars, x_lim, y_lim) 
    asteroid_x,asteroid_y,_,_ = asteroid.rescale_grid(stars, x_lim, y_lim) 
    
    #defining parameters to change axes limits to AU
    oldx = np.linspace(0,width,7)
    oldy = np.linspace(0,height,7)
    AU = [-3,-2,-1,0,1,2,3]          
    
    #creating figure for plot
    fig,ax = plt.subplots()
    plt.imshow(stars) # plot background image
    
    #plotting earth
    ax.scatter(earth_x,earth_y, color = 'b', s = 0.1) # plot Earth
    imagebox_earth = OffsetImage(Earth_img, zoom = 0.02)
    ab_earthimg = AnnotationBbox(imagebox_earth, [earth_x[-1],earth_y[-1]], xycoords = 'data', frameon = False)
    ax.add_artist(ab_earthimg)
    
    
    #plotting asteroid
    ax.scatter(asteroid_x,asteroid_y, color = 'r', s = 1) # plot asteroid
    imagebox_asteroid = OffsetImage(Asteroid_img, zoom = 0.02)
    ab_asteroidimg = AnnotationBbox(imagebox_asteroid, [asteroid_x[-1],asteroid_y[-1]], xycoords = 'data', frameon = False)
    ax.add_artist(ab_asteroidimg)
    
        
    #plotting sun 
    ax.scatter(sun_scaledx,sun_scaledy, color = 'tab:orange' , s = 1) # plot sun position
    imagebox_sun = OffsetImage(Sun_img, zoom = 0.07)
    ab_sunimg = AnnotationBbox(imagebox_sun, [sun_scaledx, sun_scaledy], xycoords = 'data', frameon = False)
    ax.add_artist(ab_sunimg)

    # changing limits/labels for axes
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel("Distance [AU]")
    plt.ylabel("Distance [AU]")
    plt.xticks(oldx,AU) # changing the axes so that they display the distance in AU
    plt.yticks(oldy,AU)
    #plt.rcParams['axes.facecolor'] = 'black'
    plt.show()
#     energy_data = pd.DataFrame({'Ek':Earth.KE,'PE': Earth.PE})
#    st.line_chart(energy_data,x = 'Days', y = 'Energy [joules]')
    #st.line_chart(Earth.L,x = 'Days', y = 'Dngular momentum')
    
main()

st.pyplot(fig=None, clear_figure=None)
st.set_option('deprecation.showPyplotGlobalUse', False)

