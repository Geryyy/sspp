import numpy as np
import casadi as ca
from sspp import _sspp as sp

## BSpline basis function scipy help, can be implemented in casadi
# evaluated at theta
# k..spline order
# i..index
# t..knot points
##
def B(theta, k, i, t):
  if k == 0:
    #return 1.0 if t[i] <= theta < t[i+1] else 0.0
    if t[i] <= theta < t[i+1]:
      return 1.0
    #if i == len(t) - k - 2 and theta == t[i+1]:
    #  return 1.0
    return 0.0
  if t[i+k] == t[i]:
    c1 = 0.0
  else:
    c1 = (theta - t[i])/(t[i+k] - t[i]) * B(theta, k-1, i, t)
  if t[i+k+1] == t[i+1]:
    c2 = 0.0
  else:
    c2 = (t[i+k+1] - theta)/(t[i+k+1] - t[i+1]) * B(theta, k-1, i+1, t)
  return c1 + c2

def dB(theta, k, i, t):
  if k == 0:
    return 0.0
  if t[i+k] == t[i]:
    c1 = 0.0
  else:
    c1 = k / (t[i+k] - t[i]) * B(theta, k-1, i, t)
  if t[i+k+1] == t[i+1]:
    c2 = 0.0
  else:
    c2 = -k / (t[i+k+1] - t[i+1]) * B(theta, k-1, i+1, t)
  return c1 + c2

def bspline(theta, t, c, k):
  n = len(t) - k - 1
  #assert (n >= k+1) and (len(c) >= n)
  if theta < 0:
    return c[0] * B(0, k, 0, t)
  if theta >= 1:
    return c[n - 1]# * B(1, k, n - 1, t)
  return sum(c[i] * B(theta, k, i, t) for i in range(n))

def bspline_derivative(theta, t, c, k):
  n = len(t) - k - 1
  return sum(c[i] * dB(theta, k, i, t) for i in range(n))


def knot_vector(n_control_points, k):
  n_knots = n_control_points + k + 1  # Total number of knots for B-spline
  t = np.linspace(0, 1, n_knots - 2 * k)  # Internal knots only
  t = np.concatenate(([0] * k, t, [1] * k))  # Add k repeated knots at the ends
  return t

##
# Computes the SLERP interpolation starting from R0 at theta
# S...skew symmetric matrix, phi rotation angle
##
def evalRotationInterpolation(R0, theta, S, phi):
  return np.dot(R0, np.eye(3) + np.sin(theta * phi) * S + (1 - np.cos(theta * phi)) * np.dot(S, S))

## SLERP derivative
def evalRotationInterpolationDiff(R0, theta, S, phi):
  return np.dot(R0, np.cos(theta * phi) * S + np.sin(theta * phi) * np.dot(S, S))

##
# Computes the SLERP interpolation for a list of rotation matrices R at theta
##
def evalRotationInterpolationFull(R, phi, S, theta, theta_vec):
  for i in range(len(R) - 1):
    if theta < theta_vec[i + 1] and theta >= theta_vec[i]:
      return evalRotationInterpolation(R[i], (theta - theta_vec[i]) / (theta_vec[i + 1] - theta_vec[i]), S[i], phi[i]) # theta scaling is important

def evalRotationInterpolationDiffFull(R, phi, S, theta, theta_vec):
  for i in range(len(R) - 1):
    if theta < theta_vec[i + 1] and theta >= theta_vec[i]:
      return evalRotationInterpolationDiff(R[i], (theta - theta_vec[i]) / (theta_vec[i + 1] - theta_vec[i]), S[i], phi[i]) # theta scaling is important

################################################################################################
# casadi versions
################################################################################################

def casadiBspline(theta, t, c, k):
  n = t.shape[0] - k - 1
  #assert (n >= k+1) and (len(c) >= n)
  
  result = 0
  for i in range(n):
    result += c[i] * casadiB(theta, k, i, t)
  
  return result

def casadiBsplineVec(theta, t, c, k):
  n = t.shape[0] - k - 1
  #assert (n >= k+1) and (len(c) >= n)
  
  result = 0
  for i in range(n):
    result += c[i, :] * casadiB(theta, k, i, t)

  return ca.if_else(theta < 0, c[0, :] * casadiB(0, k, 0, t), ca.if_else(theta >= 1, c[n - 1, :], result))

def casadiB(theta, k, i, t):
  if k == 0: ## case doesn't exist
    return ca.if_else(ca.logic_or(ca.logic_and(t[i] <= theta, theta < t[i+1]), theta == t[i + 1]), 1.0, 0.0) ## TODO theta <= t[i + 1]
  
  c1 = ca.if_else(t[i+k] == t[i], 0.0, (theta - t[i]) / (t[i+k] - t[i]) * casadiB(theta, k-1, i, t))
  c2 = ca.if_else(t[i+k+1] == t[i+1], 0.0, (t[i+k+1] - theta) / (t[i+k+1] - t[i+1]) * casadiB(theta, k-1, i+1, t))
  
  return c1 + c2





def test_bspline():
  import matplotlib.pyplot as plt

  # Step 1: Initialize control points
  q0 = np.zeros(7)
  q1 = np.ones(7)
  c = np.linspace(q0, q1, 7)  # 7 control points from np.zeros(7) to np.ones(7)
  # Step 2: Generate the knot vector
  n_control_points = len(c)  # Number of control points
  k = 3  # Spline order
  
  

  theta = np.linspace(0, 1, 10) # evaluation points
  y = np.array([bspline(x, t, c, k) for x in theta])
  plt.plot(theta, y)
  plt.grid()
  plt.show()



import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def test_casadi_bspline():
    # Step 1: Initialize control points
    q0 = np.zeros(7)
    q1 = np.ones(7)
    c = np.linspace(q0, q1, 7)  # 7 control points from np.zeros(7) to np.ones(7)
    
    # Step 2: Generate the knot vector
    n_control_points = len(c)  # Number of control points
    k = 3  # Spline order
    t = knot_vector(n_control_points, k)

    # Step 3: Evaluate the B-spline using CasADi
    theta = np.linspace(0, 1, 100)  # Evaluation points
    y = np.array([casadiBspline(x, t, c, k) for x in theta])

    # Step 4: Plot the B-spline
    plt.plot(theta, y.squeeze())
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # test_bspline()
    test_casadi_bspline()