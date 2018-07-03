#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math

t_f = 100.0  # s
t_step = 2.0  # s

MAX_DIST = 20.0  # distance landmarks can be sensed
MAX_ITR = 20  #What should this be
NUM_LM = 5

sigma2_v = 0.1**2  # variance for velocity and angular velocity
sigma2_w = math.radians(10.0)**2

sigma2_r = .2**2
sigma2_phi = math.radians(2.0)**2
Qt = np.diag([sigma2_r, sigma2_phi])
Qt_inv = np.linalg.inv(Qt)
Rt = np.diag([0.1, 0.1, math.radians(1.0)]) ** 2
Rt_inv = np.linalg.inv(Rt)


def graph_slam(u, z1_t, x, z_hat):
    # repeat the following until convergence. How do I know when it has converged
    j = 0
    for i in range(MAX_ITR):
        omega_xx, xi_xx, omega_xm, omega_mx, xi_xm, omega_mm= linearize(u, z1_t, x, z_hat)  # Do the linearization
        ox_til, xi_x_til= reduction(omega_xx, xi_xx, omega_xm, omega_mx, xi_xm, omega_mm)  # Reduces the Information matrix
        x, x_hat_lm = solve(ox_til, xi_x_til, omega_xx, omega_xm, xi_xm, omega_mm)

    # omega_xx, xi_xx, omega_xm, omega_mx, xi_xm, omega_mm = linearize(u, z1_t, x)  # Do the linearization
    # ox_til, xi_x_til = reduction(omega_xx, xi_xx, omega_xm, omega_mx, xi_xm, omega_mm)  # Reduces the Information matrix
    # x, x_hat_lm = solve(ox_til, xi_x_til, omega_xx, xi_xx, omega_xm, omega_mx, xi_xm, omega_mm)

    return x, x_hat_lm


def solve(ox_til, xi_x_til, omega_xx, omega_xm, xi_xm, omega_mm):
    #Calculate the mean
    P0_t = np.linalg.inv(ox_til)
    xi_x_t = np.zeros((3 * len(xi_x_til), 1))
    for i in range(len(xi_x_til)):
        temp = xi_x_til[i]
        xi_x_t[3*i, 0] = temp.item(0)
        xi_x_t[3*i+1, 0] = temp.item(1)
        xi_x_t[3*i+2, 0] = temp.item(2)
    x_hat = np.matmul(P0_t, xi_x_t)

    #Calculate the covariance
    x_hat_lm = np.zeros((2*5, 1))
    for j in range(len(xi_xm)):
        mu = np.zeros((0, 1))
        o_xm = np.zeros((2, 0))
        if not np.linalg.det(omega_mm[2 * j:2 * j + 2, 2 * j:2 * j + 2]) == 0:
            o_jj = np.linalg.inv(omega_mm[2 * j:2 * j + 2, 2 * j:2 * j + 2])
            xi_j = xi_xm[j]
        for i in range(omega_xx.shape[1]/3 - 1):
            if np.count_nonzero(omega_xm[2 * i:2 * i + 2, 3 * j:3 * j + 3]):
                temp_m = x_hat[3*i:3*i+3]
                mu = np.vstack((mu, temp_m))

                temp_o = omega_xm[2 * i:2 * i + 2, 3 * j:3 * j + 3]
                o_xm = np.hstack((o_xm, temp_o))

                # # o_xm = omega_xm[2*j:2*j+2, 3*i:3*i+3]
                # o_xm = omega_xm[2 * i:2 * i + 2, 3 * j:3 * j + 3]
                # xi_j = xi_xm[j]
                # mu = x_hat[3*i:3*i+3]
                #
                # temp1 = xi_j + np.matmul(o_xm, mu)
                # est = np.matmul(o_jj, temp1)
                #
                # x_hat_lm[2*j, 0] = est[0, 0]  # PLUS, MINUS OR JUST EQUAL TO?
                # x_hat_lm[2*j+1, 0] = est[1, 0]
        est = np.matmul(o_jj, xi_j + np.matmul(o_xm, mu))
        x_hat_lm[2*j, 0] = est[0, 0]
        x_hat_lm[2*j+1, 0] = est[1, 0]

    # Put x_hat into a 3 x Number of poses array
    x_hat_new = np.zeros((3, x_hat.shape[0]/3))
    for i in range(x_hat_new.shape[1]):
        x_hat_new[:, i] = x_hat[3*i:3*i+3].T
        x_hat_new[2, i] = ang_correct(x_hat_new[2, i])

    x_lm = np.zeros((2, x_hat_lm.shape[0]/2))
    for i in range(x_lm.shape[1]):
        x_lm[:, i] = x_hat_lm[2*i:2*i+2].T

    return x_hat_new, x_lm


def reduction(omega_xx, xi_xx, omega_xm, omega_mx, xi_xm, omega_mm):
    ox_til = omega_xx
    xi_x_til = xi_xx

    '''
    Not sure I have done the reduction entirely right.
    1. From table in book (349): Is every Omega_Tao(j),j*OmegaInv_j,j*xi_j subtracted from every pose in Tao or just the one it corresponds to
    '''

    # For each feature on the map
    for j in range(len(xi_xm)):
        o_temp = np.zeros((3, 3))
        xi_temp = np.zeros((3, 1))
        for i in range(omega_xx.shape[1]/3 - 1):
            if np.count_nonzero(omega_xm[2*i:2*i+2, 3*j:3*j+3]):
                m3 = omega_xm[2*i:2*i+2, 3*j:3*j+3]
                m2 = np.linalg.inv(omega_mm[2*j:2*j+2, 2*j:2*j+2])
                m1 = omega_mx[3*j:3*j+3, 2*i:2*i+2]
                b = xi_xm[j]

                xi_temp += np.matmul(np.matmul(m1, m2), np.array(b))  # This is used with the second for loop right below
                o_temp += np.matmul(np.matmul(m1, m2), m3)

                # xi_temp = np.matmul(np.matmul(m1, m2), np.array(b))
                # xi_x_til[i] -= xi_temp

                # o_temp = np.matmul(np.matmul(m1, m2), m3)
                # ox_til[3*i:3*i+3, 3*i:3*i+3] -= o_temp

        for i in range(omega_xx.shape[1]/3 - 1):
            if np.count_nonzero(omega_xm[2*i:2*i+2, 3*j:3*j+3]):
                xi_x_til[i] -= xi_temp
                ox_til[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= o_temp

    return ox_til, xi_x_til


def linearize(u, z1_t, x, z_hat):
    l_xc = x.shape[1]
    omega_xx = np.zeros((3 * l_xc, 3 * l_xc))
    x0 = np.array(np.diag([np.infty, np.infty, np.infty]))
    omega_xx[0:3, 0:3] = x0
    xi_xx = [np.zeros((3, 1))]

    omega_xm = np.zeros((2*l_xc, 3*NUM_LM))
    omega_mx = np.zeros((3*NUM_LM, 2*l_xc))
    xi_xm = []
    for i in range(NUM_LM):
        xi_xm.append(np.zeros((2, 1)))
    omega_mm = np.zeros((2*NUM_LM, 2*NUM_LM))

    c = u.shape[1] + 1

    for i in range(1, c):
        v = u[0, i - 1]
        w = u[1, i - 1]
        r = v/w
        theta = x[2, i - 1]

        dx = np.array([[-r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                        [r * math.cos(theta) - r * math.cos(theta + w * t_step)],
                        [w * t_step]])

        xhat_t = x[:, i - 1].reshape(3, 1) + dx

        Gt = np.array([[1, 0, -r * math.cos(theta) + r * math.cos(theta + w * t_step)],
                       [0, 1, -r * math.sin(theta) + r * math.sin(theta + w * t_step)],
                       [0, 0, 1]])
        o_temp = np.matmul(np.matmul(-Gt.T, Rt_inv), -Gt)
        omega_xx[3*i:(3*i)+3, 3*(i - 1):3*(i-1)+3] = o_temp
        omega_xx[3*(i - 1):3*(i-1)+3, 3*i:3*i+3] = o_temp.T
        omega_xx[3*i:3*i+3, 3*i:3*i+3] = o_temp

        xi_temp = np.matmul(np.matmul(-Gt.T, Rt_inv), (xhat_t - np.matmul(Gt, x[:, i-1]).reshape(3, 1)))
        xi_xx.append(xi_temp)
        xi_xx[i-1] += xi_temp

    for j in range(len(z1_t)):
        zt_i = z1_t[j]
        lz = zt_i.shape[0]
        for k in range(lz):
            r = zt_i[k, 0]
            phi = zt_i[k, 1]
            index = int(zt_i[k, 2])

            if z_hat[0, index] == 0 and z_hat[1, index] == 0:
                dx = r * math.cos(phi)  # Not sure dx and dy are correct. Where do I get mu_j from?
                dy = r * math.sin(phi)
            else:
                dx = z_hat[0, index] - x[0, j]
                dy = z_hat[1, index] - x[1, j]

            delta = np.array([[dx],
                              [dy]])

            q = np.matmul(delta.T, delta)[0, 0]
            phi_hat = math.atan2(dy, dx) - x[2, j]
            zt_hat = np.array([[math.sqrt(q)], [ang_correct(phi_hat)]])

            Ht = 1/q * np.array([[-math.sqrt(q) * dx, -math.sqrt(q) * dy, 0, math.sqrt(q) * dx, math.sqrt(q) * dy],
                                  [dy, -dx, -q, -dy, dx]])

            o_temp = np.matmul(np.matmul(Ht.T, Qt_inv), Ht)  # Also need to break this up and add to different parts
            omega_xx[3*(j + 1):3*(j+1)+3, 3*(j + 1):3*(j+1)+3] += o_temp[0:3, 0:3]
            omega_xm[2*(j+1):2*(j+1)+2, 3*index:3*index+3] += o_temp[3:5, 0:3]
            omega_mx[3*index:3*index+3, 2*(j+1):2*(j+1)+2] += o_temp[0:3, 3:5]
            omega_mm[2*index:2*index+2, 2*index:2*index+2] += o_temp[3:5, 3:5]

            zt = np.array([[r], [phi]])
            state = np.array([[x[0, j]], [x[1, j]], [x[2, j]], [x[0, j] + dx], [x[1, j] + dy]])
            xi_temp = np.matmul(np.matmul(Ht.T, Qt_inv), (zt - zt_hat + np.matmul(Ht, state)))
            xi_xx[j] += xi_temp[0:3]
            xi_xm[k] += xi_temp[3:5]

    return omega_xx, xi_xx, omega_xm, omega_mx, xi_xm, omega_mm


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
                   [-5.0, 5.0]])  #x and y positions for each land mark

    u = np.matrix(np.zeros((2, 0)))  # first index is v the second is w
    ud = np.matrix(np.zeros((2, 0)))
    z1_t = []
    z_hat = np.zeros((2, NUM_LM))

    t = 0.0
    while t < t_f:
        u = get_inputs(u)
        x, ud, z, x_est = observation(u, ud, lm)  # Get the landmarks and add uncertainty to the measurements
        z1_t.append(z)
        x_hat, z_hat = graph_slam(ud, z1_t, x_est, z_hat)

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
