#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math

t_f = 100.0  # s
t_step = 0.5  # s

MAX_DIST = 20.0  # distance landmarks can be sensed

sigma2_v = 0.1**2  # variance for velocity and angular velocity
sigma2_w = math.radians(10.0)**2  #will these be big enough??

sigma2_r = .2**2
sigma2_phi = math.radians(2.0)**2
Qt = np.diag([sigma2_r, sigma2_phi])
Rt = np.diag([0.1, 0.1, math.radians(1.0)]) ** 2


def graph_slam(u, z, x):
    #repeat the following until convergence. How do I know when it has converged
    omega, xi = linearize(u, z, x)  # Do the linearization
    #do reduction
    #do solving
    #repeat until converges

    return x, z


def linearize(u, z, x):
    omega = np.matrix(np.zeros((3,3))) #infinity in the first 3 positions?
    x0 = np.diag([np.infty, np.infty, np.infty])  # Not sure how to put this into omega?
    xi = np.array([0.0])

    c = u.shape[1]

    for i in range(c):
        v = u[0, i]
        w = u[1, i]
        r = v/w
        theta = x[2, i]

        dx = np.matrix([[-r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                        [r * math.cos(theta) - r * math.cos(theta + w * t_step)],
                        [w * t_step]])

        xhat_t = x[:, i] + dx

        Gt = np.matrix([[1, 0, r * math.cos(theta) - r * math.cos(theta + w * t_step)],
                       [0, 1, r * math.sin(theta) - r * math.sin(theta + w * t_step)],
                       [0, 0, 1]])

    # for [r_lm, phi] in z:
    # need a history of detected landmarks at each step

    return omega, xi


def observation(u, ud, lm):

    x = motion_model(u)  # augment the current x vector

    z = np.matrix(np.zeros((0, 2))) #place for r and phi

    for [lm_x, lm_y] in lm:
        dx = lm_x - x[0, -1]
        dy = lm_y - x[1, -1]

        r = math.sqrt(dx**2 + dy**2)
        phi = ang_correct(math.atan2(dy, dx) - x[2, -1])
        if r < MAX_DIST:
            # Add uncertainty to the values
            r = r + np.random.randn() * sigma2_r
            phi = phi + np.random.randn() * sigma2_phi
            temp = np.matrix([r, phi])
            z = np.vstack((z, temp))

    # calculate the measured velocity
    temp = np.zeros((2, 1))
    temp[0, 0] = u[0, -1] + np.random.randn() * sigma2_v
    temp[1, 0] = u[1, -1] + np.random.randn() * sigma2_w

    ud = np.hstack((ud, temp))  # add the new measured velocity

    x_hat = motion_model(ud)

    return x, ud, lm, x_hat


def get_inputs(u):
    v = 1.0 # m/s
    w = .1 # rad/s

    temp = np.matrix(np.array([[v], [w]]))
    u_new = np.hstack((u, temp))
    return u_new


def motion_model(u):
    x = np.zeros((3, 1))
    c = u.shape[1]

    for i in range(c):
        v = u[0, i]
        w = u[1, i]
        r = v/w
        theta = x[2, -1]
        dx = np.matrix([[-r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                        [r * math.cos(theta) - r * math.cos(theta + w * t_step)],
                        [w * t_step]])
        temp = x[:, -1] + dx
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

    u = np.matrix(np.zeros((2, 0)))  # first index is v the second is w
    ud = np.matrix(np.zeros((2, 0)))
    z1_t = np.matrix(np.zeros((0, 2)))

    t = 0.0
    while t < t_f:
        u = get_inputs(u)
        x, ud, z, x_est = observation(u, ud, lm)  # Get the landmarks and add uncertainty to the measurements
        x_hat, z_hat = graph_slam(ud, z, x_est)

        # Plot the landmarks and estimated landmark positions
        plt.cla()
        plt.plot(lm[:, 0], lm[:, 1], 'kx')

        # Plot the true position and estimated position
        plt.plot(np.array(x[0, :]).flatten(), np.array(x[1, :]).flatten(), 'k')
        plt.plot(np.array(x_est[0, :]).flatten(), np.array(x_est[1, :]).flatten(), 'r')

        plt.pause(.001)

        t += t_step

    print 'Finished'
    plt.waitforbuttonpress()
    plt.close()


if __name__ == "__main__":
    main()