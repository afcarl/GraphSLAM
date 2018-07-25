#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
import copy

t_f = 100.0  # s
t_step = 2.0  # s

MAX_DIST = 20.0  # distance landmarks can be sensed
MAX_ITR = 3  # What should this be
NUM_LM = 5

sigma2_v = 0.1**2  # variance for velocity and angular velocity
sigma2_w = math.radians(10.0)**2

sigma2_r = .2**2
sigma2_phi = math.radians(2.0)**2
Qt = np.diag([sigma2_r, sigma2_phi])
Qt_inv = np.linalg.inv(Qt)
Rt = np.diag([0.1, 0.1, math.radians(1.0)]) ** 2
Rt_inv = np.linalg.inv(Rt)

def get_inputs(u):
    v = 1.0  # m/s
    w = .1  # rad/s

    temp = np.matrix(np.array([[v], [w]]))
    u_new = np.hstack((u, temp))
    return u_new


def motion_model(u):
    x = np.array(np.zeros((3, 1)))
    c = u.shape[1]

    for i in range(c):
        v = u[0, i]
        w = u[1, i]
        r = v/w
        theta = x[2, -1]
        dx = np.array([[-r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                       [r * math.cos(theta) - r * math.cos(theta + w * t_step)],
                       [w * t_step]])
        temp = x[:, -1].reshape(3, 1) + dx
        temp[2, 0] = ang_correct(temp[2, 0])
        x = np.hstack((x, temp))

    return x


def ang_correct(ang):
    while ang >= math.pi:
        ang = ang - 2 * math.pi

    while ang <= -math.pi:
        ang = ang + 2 * math.pi

    return ang


def main():
    print 'Starting GraphSLAM'

    lm = np.array([[10.0, -2.0],
                   [15.0, 10.0],
                   [3.0, 15.0],
                   [-5.0, 20.0],
                   [-5.0, 5.0]])  # x and y positions for each land mark

    u = np.matrix(np.zeros((2, 0)))  # first index is v the second is w
    ud = np.matrix(np.zeros((2, 0)))
    z1_t = []
    z_hat = np.zeros((2, NUM_LM))

    t = 0.0
    while t < t_f:
        u = get_inputs(u)
        # x, ud, z, x_est = observation(u, ud, lm)  # Get the landmarks and add uncertainty to the measurements
        # z1_t.append(z)
        # x_hat, z_hat = graph_slam(ud, z1_t, x_est, z_hat)

        # Plot the landmarks and estimated landmark positions
        plt.cla()
        plt.plot(lm[:, 0], lm[:, 1], 'kx')

        # Plot the true position and estimated position
        # plt.plot(np.array(x[0, :]).flatten(), np.array(x[1, :]).flatten(), 'k')
        # plt.plot(np.array(x_est[0, :]).flatten(), np.array(x_est[1, :]).flatten(), 'r')
        # plt.plot(np.array(x_hat[0, :]).flatten(), np.array(x_hat[1, :]).flatten(), 'b')

        plt.pause(.001)

        t += t_step

    print 'Finished'
    plt.waitforbuttonpress()
    plt.close()


if __name__ == "__main__":
    main()