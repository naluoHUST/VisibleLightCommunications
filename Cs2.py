import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
from scipy import special
from scipy.stats import norm
from scipy.integrate import quad
import math

class PAM_4LED:
    def __init__(self, optical_power, size):
        super(PAM_4LED, self).__init__()
        self.M = size
        self.size = self.M
        self.apmnoise = 10**-7
        self.optical_power_W = 10**(optical_power/10 - 3)
        self.modultaion_index = 1
        self.A_max = (self.optical_power_W/0.44)*self.modultaion_index/self.apmnoise
        #print(self.A_max)
        self.x = (2*np.arange(1, self.M + 1) - self.M-1)/(self.M-1)
        #print(self.x)
        self.d = self.x * self.A_max
        #print(self.d)
        self.N_0 = 1
        self.sigma1 = 1
        self.sigma2 = 5
        self.lamda = 0
        self.h_u = 0
        self.w = 0
        self.eta = 0.44
        self.LEDnumbers = 4
        self.X_amb = 10.93
        self.i_amb = 5*10**(-12)
        self.gamma = 0.73
        self.A_r = 1 *10 **(-4)
        self.Psi = 80*np.pi/180
        self.B_mod = 20 * 10**6
        self.e= 1.602176634 * 10**(-19)
        self.power = 1
        self.noise_power= 10**(-12.882)
        self.LED1 = np.array([-np.sqrt(2), -np.sqrt(2), 3])
        self.LED2 = np.array([ np.sqrt(2), -np.sqrt(2), 3])
        self.LED3 = np.array([-np.sqrt(2),  np.sqrt(2), 3])
        self.LED4 = np.array([ np.sqrt(2),  np.sqrt(2), 3])
        self.Trans_pos = np.array([self.LED1,self.LED2,self.LED3,self.LED4])
        self.Bobstr = [0, 0, 0.5]
        self.Evestr = [-3, 3, 0.5]
        self.Bob_pos = np.array(self.Bobstr)
        self.Eve_pos = np.array(self.Evestr)
        self.saveCap1 = 0
        self.goc = 0
        self.goc_2 = 0
        self.SER_ub = 0
        self.secCap = 0
        self.utility = 0
        #self.prob = np.array([0.25, 0.25, 0.25, 0.25])
        #Channel gain
        self.SemiAngle       = 60 *np.pi/180
        self.lambertian_angle= 120*np.pi/180
        self.PD_FOV          = 60 *np.pi/180
        self.PD_RefracIndex  = 1.5
        self.PD_OptFilGain   = 1
        self.PD_Area         = 10**(-4)
        self.LEDConver       = 0.44
        self.PD_Responsivity  = 0.54
        self.Gaus1 = 1/np.sqrt(2 * np.pi * self.sigma1**2)
        self.Gaus2 = 1/np.sqrt(2 * np.pi * self.sigma2**2)


    def distance(self):
        self.dis_b = np.empty(self.LEDnumbers)
        self.dis_e = np.empty(self.LEDnumbers)
        for i in range(self.LEDnumbers):
            self.dis_b[i] = np.linalg.norm(self.Trans_pos[i] - self.Bob_pos)
        for i in range(self.LEDnumbers):
            self.dis_e[i] = np.linalg.norm(self.Trans_pos[i] - self.Eve_pos)
        #Bob: [3.20156212 3.20156212 3.20156212 3.20156212]
        #Eve: [4.23164912 3.5        3.5        2.56771216]
        return self.dis_b, self.dis_e


    def angle(self):
        self.phita_b = np.empty(self.LEDnumbers)
        self.phita_e = np.empty(self.LEDnumbers)

        #goi ham khoang cach
        self.distance()

        for i in range(self.LEDnumbers):
            self.phita_b[i] = np.arccos(2.5/self.dis_b[i])
            self.phita_e[i] = np.arccos(2.5/self.dis_e[i])
        #Phita_bob: [0.67474094 0.67474094 0.67474094 0.67474094] checked, true!
        #Phita_eve: [0.93876337 0.77519337 0.77519337 0.23016231]
        return self.phita_b, self.phita_e

    def L_receiver(self):
        self.L_Bob = np.empty(self.LEDnumbers)
        self.L_Eve = np.empty(self.LEDnumbers)

        self.angle()

        for i in range(self.LEDnumbers):
            self.L_Bob[i] = (1+1)*np.cos(self.phita_b[i])/(2*np.pi)
            self.L_Eve[i] = (1+1)*np.cos(self.phita_e[i])/(2*np.pi)
        return self.L_Bob, self.L_Eve

    def goc_receiver(self):
        self.goc_Bob = np.empty(self.LEDnumbers)
        self.goc_Eve = np.empty(self.LEDnumbers)
        for i in range(self.LEDnumbers):
            self.goc_Bob[i] = self.PD_RefracIndex**2 / np.sin(self.PD_FOV)**2
            self.goc_Eve[i] = self.PD_RefracIndex**2 / np.sin(self.PD_FOV)**2
        #goc of Bob: [3. 3. 3. 3.] checked, true!
        #goc of Eve: [3. 3. 3. 3.]
        return self.goc_Bob, self.goc_Eve

    def channelgain(self):
        self.H_Bob = np.empty(self.LEDnumbers)
        self.H_Eve = np.empty(self.LEDnumbers)


        self.distance()
        self.goc_receiver()
        self.L_receiver()

        for i in range(self.LEDnumbers):
            self.H_Bob[i] = self.PD_Area*(1/(self.dis_b[i]**2))*self.goc_Bob[i]*np.cos(self.phita_b[i])*self.L_Bob[i]*self.LEDConver*self.PD_Responsivity
        for i in range(self.LEDnumbers):
            self.H_Eve[i] = self.PD_Area*(1/(self.dis_e[i]**2))*self.goc_Eve[i]*np.cos(self.phita_e[i])*self.L_Eve[i]*self.LEDConver*self.PD_Responsivity



        return self.H_Bob, self.H_Eve

Cam = PAM_4LED(10, 2)
H, H_Eve = Cam.channelgain()
sigma = 10**-5
W = np.zeros(4)
epsilon = 10**-3
L = 10
Delta = 1
tol = 10**-3


def f(alpha, H, W, sigma):
  return alpha*np.log((2*np.dot(H,W))/(np.sqrt(2*np.pi*np.e*sigma**2))) - np.log((alpha**alpha)*(1-alpha)**(1.5*(1-alpha)))

def df(alpha, H, W, sigma):
  return np.log((2*np.dot(H, W))/np.sqrt(2*np.pi*np.e*sigma**2)) - np.log(alpha) + 1.5*np.log(1-alpha) + 0.5

def bisection(df, a, b, tol, H, W, sigma):
    if df(a, H, W, sigma)*df(b, H, W, sigma) >= 0:
        return None
    alpha = a
    while (b-a)/2 > tol:
        alpha = (a+b)/2
        if df(alpha, H, W, sigma) == 0:
            return alpha
        elif df(alpha, H, W, sigma)*df(a, H, W, sigma) < 0:
            b = alpha
        else:
            a = alpha
    return alpha
for i in range(L):
    a = 0.01
    b = 0.99
    
    while (b-a)/2 > tol:
        alpha = (a+b)/2
        if df(alpha, H, W, sigma)*df(a, H, W, sigma) < 0:
            b = alpha
        else:
            a = alpha

    W_var = cp.Variable(4)
    term1 = alpha * cp.log(2 * cp.sum(cp.matmul(H, W_var)) / (np.sqrt(2 * np.pi * np.e * sigma)))
    term2 = cp.log(alpha) * alpha + cp.log(1 - alpha) * (1.5 * (1 - alpha))
    obj = cp.Maximize(term1 - term2)
    constraints = [cp.norm(W_var, 'inf') <= Delta, cp.matmul(H_Eve, W_var) == 0]
    problem = cp.Problem(obj, constraints)
    problem.solve()

    W_new = W_var.value

    # Kiểm tra điều kiện hội tụ
    if np.linalg.norm(W_new - W) < epsilon:
        break

    W = W_new

# Final values
alpha_final = alpha
W_final = W
print(W_final)
