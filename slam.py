#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math

t_f = 100.0  # s
t_step = 0.5  # s

range = 20.0  # distance landmarks can be sensed

sigma2_v = 0.1**2  # variance for velocity and angular velocity
sigma2_w = math.radians(10.0)**2  #will these be big enough??
sigma2_r = .2**2
sigma2_phi = math.radians(2.0)**2

def graph_slam(ud, z, x):

    #do linearization
    #do reduction
    #do solving
    #repeat until converges

    return x

def observation(x, ud, lm, x_hat):
    u = get_inputs()

    x = motion_model(x, u)  # augment the current x vector

    z = np.matrix(np.zeros((0, 2))) #place for r and phi

    for [lm_x, lm_y] in lm:
        dx = lm_x - x[0, -1]
        dy = lm_y - x[1, -1]

        r = math.sqrt(dx**2 + dy**2)
        phi = ang_correct(math.atan2(dy, dx) - x[2, -1])
        if r < range:
            r = r + np.random.randn() * sigma2_r
            phi = phi + np.random.randn() * sigma2_phi
            temp = np.matrix([r, phi])
            z = np.vstack((z, temp))

    # calculate the measured velocity
    u[0, 0] = u[0, 0] + np.random.randn() * sigma2_v
    u[1, 0] = u[1, 0] + np.random.randn() * sigma2_w

    ud = np.hstack((ud, u))  # add the new measured velocity

    x_hat = motion_model(x_hat, ud)

    return x, ud, z, x_hat

def get_inputs():
    v = 1.0 # m/s
    w = .1 # rad/s

    u = np.matrix(np.array([[v], [w]]))
    return u

def motion_model(x, u):
    v = u[0, -1]
    w = u[1, -1]
    x_prev = x[:, -1]
    theta = x_prev[2, 0]

    r = v/w
    dx = np.matrix([[-r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                    [r * math.cos(theta) - r * math.cos(theta + w * t_step)],
                    [w * t_step]])
    temp = x_prev + dx
    x = np.hstack((x, temp))

    return x

def ang_correct(ang):
    while ang >= 2 * math.pi:
        ang = ang - 2 * math.pi

    while ang <= -2 * math.pi:
        ang = ang + 2 * math.pi

    return ang

def main():
    print 'Starting GraphSLAM'

    lm = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0],
                     [-5.0, 5.0]])  #x and y positions for each land mark

    x = np.matrix(np.zeros((3, 1)))  #the state variables are x, y, and yaw (heading angle) Actual position
    x_est = np.matrix(np.zeros_like(x)) # this will have the estimated state
    ud = np.matrix(np.zeros((2, 1)))  # first index is v the second is w

    t = 0.0
    while t < t_f:
        x, ud, z, x_est = observation(x, ud, lm, x_est)
        x_hat = graph_slam(ud, z, x_est)

        t += t_step

    print 'Finished'


if __name__=="__main__":
    main()