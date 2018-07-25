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


def graph_slam(u, z1_t, x_est, z_hat):

    for i in range(MAX_ITR):
        omega_xx, omega_mm, omega_mx, omega_xm, xi_x, xi_m = linearize(u, z1_t, x_est, z_hat)

    return x_est  # This will change


def linearize(u, z1_t, x_est, z_hat):
    # Set up different parts of omega
    omega_xx = np.zeros((3*x_est.shape[1], 3*x_est.shape[1]))
    o_0 = np.diag((np.infty, np.infty, np.infty))
    omega_xx[0:3, 0:3] = o_0
    omega_mm = np.zeros((2*NUM_LM, 2*NUM_LM))
    omega_xm = np.zeros((3*x_est.shape[1], 2*NUM_LM))
    omega_mx = np.zeros((2*NUM_LM, 3*x_est.shape[1]))

    # Set up both parts of xi
    xi_x = np.zeros((3, x_est.shape[1]))
    xi_m = np.zeros((2, NUM_LM))

    for i in range(u.shape[1]):
        v = u[0, i]
        w = u[1, i]
        r = v/w
        theta = x_est[2, i]

        dx = np.array([[-r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                       [r * math.cos(theta) - r * math.cos(theta + w * t_step)],
                       [w * t_step]])
        x_hat = x_est[:, i].reshape((3, 1)) + dx

        jacob_G = np.array([[1, 0, -r * math.cos(theta) + r * math.cos(theta + w * t_step)],
                            [0, 1, -r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                            [0, 0, 1]])

        o_temp = np.matmul(np.matmul(-jacob_G.T, Rt_inv), -jacob_G)
        omega_xx[3 * i:3 * (i + 1), 3 * (i + 1):3 * (i + 2)] = o_temp
        omega_xx[3 * (i + 1):3 * (i + 2), 3 * i:3 * (i + 1)] = o_temp

        xi_temp = np.matmul(np.matmul(-jacob_G.T, Rt_inv), x_hat - np.matmul(jacob_G, x_est[:, i].reshape(3, 1)))
        xi_x[:, i+1] = (xi_x[:, i+1].reshape((3, 1)) + xi_temp).reshape((3,))
        xi_x[:, i] = (xi_x[:, i].reshape((3, 1)) + xi_temp).reshape((3, ))

    for i in range(len(z1_t)):
        for z_t in z1_t[i]:
            index = int(z_t.item(2))

            if z_hat[0, index] == 0:
                dx = z_t.item(0) * math.cos(z_t.item(1))
                dy = z_t.item(0) * math.sin(z_t.item(1))
                q = z_t.item(0)
                phi = z_t.item(1)
                x_m = x_est[0, i+1] + dx
                y_m = x_est[1, i+1] + dy
            else:
                dx = z_hat[0, index] - x_est[0, i+1]
                dy = z_hat[1, index] - x_est[1, i+1]
                delta = np.array([dx, dy])
                q = np.matmul(delta, delta.T)[0, 0]
                phi = ang_correct(math.atan2(dy, dx) - z_t.item(1))
                x_m = z_hat[0, index]
                y_m = z_hat[1, index]

            z_est = np.array([math.sqrt(q), phi]).T

            jacob_H = 1.0/q * np.array([[-math.sqrt(q)*dx, -math.sqrt(q)*dy, 0, math.sqrt(q)*dx, math.sqrt(q)*dy],
                                        [dy, -dx, -q, -dy, dx]])

            state = np.array([x_est[0, i+1], x_est[1, i+1], x_est[2, i+1], x_m, y_m]).T
            zt = np.array([z_t.item(0), z_t.item(1)]).T
            o_temp = np.matmul(np.matmul(jacob_H.T, Qt_inv), jacob_H)
            omega_xx[3*(i+1):3*(i+2), 3*(i+1):3*(i+2)] += o_temp[0:3, 0:3]
            omega_mm[2*index:2*(index+1), 2*index:2*(index+1)] += o_temp[3:5, 3:5]
            omega_xm[3*(i+1):3*(i+2), 2*index:2*(index+1)] += o_temp[0:3, 3:5]
            omega_mx[2*index:2*(index+1), 3*(i+1):3*(i+2)] += o_temp[3:5, 0:3]

            xi_temp = np.matmul(np.matmul(jacob_H.T, Qt_inv), zt.reshape((2, 1)) - z_est.reshape((2, 1)) + np.matmul(jacob_H, state).reshape((2, 1)))
            xi_x[:, i+1] = (xi_x[:, i+1].reshape((3, 1)) + xi_temp[0:3]).reshape((3, ))
            xi_m[:, index] = (xi_m[:, index].reshape((2, 1)) + xi_temp[3:5]).reshape((2, ))

    return omega_xx, omega_mm, omega_mx, omega_xm, xi_x, xi_m


def observation(u, ud, lm):

    x = motion_model(u)  # augment the current x vector

    z = np.matrix(np.zeros((0, 3)))  # place for r and phi

    i = 0

    for [lm_x, lm_y] in lm:
        dx = lm_x - x[0, -1]
        dy = lm_y - x[1, -1]

        r = math.sqrt(dx**2 + dy**2)
        phi = ang_correct(math.atan2(dy, dx) - x[2, -1])
        if r <= MAX_DIST:
            # Add uncertainty to the values
            r = r + np.random.randn() * sigma2_r
            phi = phi + np.random.randn() * sigma2_phi
            temp = np.matrix([r, phi, i])
            z = np.vstack((z, temp))

        i += 1

    # calculate the measured velocity
    temp = np.zeros((2, 1))
    temp[0, 0] = u[0, -1] + np.random.randn() * sigma2_v
    temp[1, 0] = u[1, -1] + np.random.randn() * sigma2_w

    ud = np.hstack((ud, temp))  # add the new measured velocity

    x_hat = motion_model(ud)

    return x, ud, z, x_hat


def get_inputs(u):
    v = 1.0  # m/s
    w = .1  # rad/s

    temp = np.array([[v], [w]])
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

    u = np.zeros((2, 0))  # first index is v the second is w
    ud = np.zeros((2, 0))
    z1_t = []
    z_hat = np.zeros((2, NUM_LM))

    t = 0.0
    while t < t_f:
        u = get_inputs(u)
        x, ud, z, x_est = observation(u, ud, lm)  # Get the landmarks and add uncertainty to the measurements
        z1_t.append(z)
        x_hat = graph_slam(ud, z1_t, x_est, z_hat)

        # Plot the landmarks and estimated landmark positions
        plt.cla()
        plt.plot(lm[:, 0], lm[:, 1], 'kx')

        # Plot the true position and estimated position
        plt.plot(np.array(x[0, :]).flatten(), np.array(x[1, :]).flatten(), 'k')
        plt.plot(np.array(x_est[0, :]).flatten(), np.array(x_est[1, :]).flatten(), 'r')
        # plt.plot(np.array(x_hat[0, :]).flatten(), np.array(x_hat[1, :]).flatten(), 'b')

        plt.pause(.001)

        t += t_step

    print 'Finished'
    plt.waitforbuttonpress()
    plt.close()


if __name__ == "__main__":
    main()