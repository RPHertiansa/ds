import matplotlib.pyplot as plt
import numpy as np

#declare data
x = np.linspace(0, 10, 1000)
y = np.sin(x)

plt.plot(x,y, label='sin(x)')
plt.plot(x, np.cos(x), 'r', label='cos(x)') #multiple lines
plt.xlabel('x')
plt.ylabel('y')
plt.legend() #add legend
plt.title('function sin(x)')
plt.show()


#coloring and design https://matplotlib.org/2.0.2/api/colors_api.html