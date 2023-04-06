import numpy as np
cos, sin, tan, radians, degrees, arccos, arcsin, arctan = np.cos, np.sin, np.tan, np.radians, np.degrees, np.arccos, np.arcsin, np.arctan2
PI = 3.14159

class Vector2:
    '''x and y components OR the length and angle from X-Axis Counter Clockwise in Degrees'''
    def __init__(self,x=0, y=0, r=0, dtheta=0):
      if x!=0 or y!=0:
        self.x = x
        self.y = y
        self.r = ((self.x**2 + self.y**2)**0.5)
        self.dtheta = degrees(arctan(self.y,self.x))
      else:
        self.r = r
        self.dtheta = dtheta
        self.x = self.r * cos(radians(dtheta))
        self.y = self.r * sin(radians(dtheta))
        
    def plus(a, b) -> 'Vector2':
      return Vector2(x=a.x+b.x, y=a.y+b.y)
    def minus(a, b) -> 'Vector2':
          return Vector2(x=a.x-b.x, y=a.y-b.y)  
    def dot(self, b):
      return (self.x*b.x) + (self.y*b.y)
    def unit(self) -> 'Vector2':
      return Vector2(x=self.x/self.r, y=self.y/self.r) 
    def scale(self, scalar: float) -> 'Vector2':
      return Vector2(x=self.x*scalar, y=self.y*scalar)   
    def angle_from_dot(a, b):
      return degrees(arccos((a.dot(b)) / (a.r * b.r) ))
    def __str__(self):
      return "i:{}, j:{}, r:{}, theta:{}".format(self.x, self.y, self.r, self.dtheta)
    def __repr__(self):
      return "i:{}, j:{}, r:{}, theta:{}".format(self.x, self.y, self.r, self.dtheta)