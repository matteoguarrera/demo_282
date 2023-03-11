# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image

def piece_of_paper(amplitude=2, magnitude_noise=4):
  x = np.linspace(0, np.pi, 100)
  func = lambda x: np.sin(x)

  sign = np.random.choice([-1,1])
  ax_lim = 2*magnitude_noise + amplitude

  # create the figure and axes objects
  fig, (ax1, ax2) = plt.subplots(1,2, figsize = (10,5))  

  # function that draws each frame of the animation
  def animate(i):
      noise = np.random.normal(0, magnitude_noise, size=x.size) # noise magnitude
      y = sign*amplitude*(i/200)*func(x) 
      ax1.clear()
      ax2.clear()
      ax1.plot(x, y + noise)
      ax2.plot(x, y)
      ax1.set_ylabel('y')
      ax2.set_xlabel('x')
      ax2.set_ylabel('y')
      ax1.set_title(' y = A*sin(x) + noise')
      ax2.set_title(' y = sin(x)')
      ax1.set_xlim(0, np.pi)
      ax1.set_ylim(-ax_lim,ax_lim)
      ax2.set_xlim(0, np.pi)
      ax2.set_ylim(-ax_lim,ax_lim)
     
  # run the animation
  ani = FuncAnimation(fig, animate, frames=200, interval=500 , repeat=False);
  ax1.clear()
  ax2.clear()
  
  writergif = PillowWriter(fps=30) 
  ani.save(f'animation.gif' , writer=writergif)
