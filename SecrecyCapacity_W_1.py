import matplotlib.pyplot as plt
import numpy as np
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
        self.Evestr = [3, 3, 0.5]
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


        #Bob: [2.21357353e-06 2.21357353e-06 2.21357353e-06 2.21357353e-06]
        #Eve: [1.26706391e-06 1.85217377e-06 1.85217377e-06 3.44132066e-06]

        return self.H_Bob, self.H_Eve

    def cal_Bob_func(self, y):
        self.p_Bob = 0
        self.channelgain()
        self.T1 = 0
        self.chain_gain_Bob = np.sum(self.H_Bob)
        #print(self.chain_gain_Bob)
        for i in range(self.M):
            self.T1 = self.chain_gain_Bob * self.d[i]
            self.p_Bob += (1/self.M) * self.Gaus1 * np.exp(-(y - self.T1)**2/(2*self.sigma1**2))

        #print(self.p_Bob)
        if self.p_Bob < 10**-300:
            self.p_Bob = 10**-300
        self.func_Bob = - self.p_Bob * np.log2(self.p_Bob)
        return self.func_Bob
    def cal_Eve_func(self, y):
        self.p_Eve = 0
        self.channelgain()
        self.T2 = 0
        self.chain_gain_Eve = np.sum(self.H_Eve)
        #print(self.chain_gain_Bob)
        for i in range(self.M):
            self.T2 = self.chain_gain_Eve * self.d[i]
            self.p_Eve += (1/self.M) * self.Gaus2 * np.exp(-(y - self.T2)**2/(2*self.sigma2**2))

        #print(self.p_Bob)
        if self.p_Eve < 10**-300:
            self.p_Eve = 10**-300
        self.func_Eve = - self.p_Eve * np.log2(self.p_Eve)
        return self.func_Eve

    def Capacity_Bob_2PAM(self):
        self.CB2 = 0
        self.CB2 = quad(self.cal_Bob_func, -60, 60, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma1**2))
        return self.CB2
    def Capacity_Bob_4PAM(self):
        self.CB4 = 0
        self.CB4 = quad(self.cal_Bob_func, -90, 90, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma1**2))
        return self.CB4
    def Capacity_Bob_8PAM(self):
        self.CB8 = 0
        self.CB8 = quad(self.cal_Bob_func, -90, 90, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma1**2))
        return self.CB8
    def Capacity_Bob_16PAM(self):
        self.CB16 = 0
        self.CB16 = quad(self.cal_Bob_func, -90, 90, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma1**2))
        return self.CB16
    def Capacity_Bob_32PAM(self):
        self.CB32 = 0
        self.CB32 = quad(self.cal_Bob_func, -180, 180, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma1**2))
        return self.CB32
    def Capacity_Bob_64PAM(self):
        self.CB64 = 0
        self.CB64 = quad(self.cal_Bob_func, -180, 180, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma1**2))
        return self.CB64


 #Eve



    def Capacity_Eve_2PAM(self):
        self.CE2 = 0
        self.CE2 = quad(self.cal_Eve_func, -60, 60, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma2**2))
        return self.CE2
    def Capacity_Eve_4PAM(self):
        self.CE4 = 0
        self.CE4 = quad(self.cal_Eve_func, -90, 90, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma2**2))
        return self.CE4
    def Capacity_Eve_8PAM(self):
        self.CE8 = 0
        self.CE8 = quad(self.cal_Eve_func, -90, 90, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma2**2))
        return self.CE8
    def Capacity_Eve_16PAM(self):
        self.CE16 = 0
        self.CE16 = quad(self.cal_Eve_func, -90, 90, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma2**2))
        return self.CE16
    def Capacity_Eve_32PAM(self):
        self.CE32 = 0
        self.CE32 = quad(self.cal_Eve_func, -180, 180, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma2**2))
        return self.CE32
    def Capacity_Eve_64PAM(self):
        self.CE64 = 0
        self.CE64 = quad(self.cal_Eve_func, -180, 180, limit = 5000)[0] - np.log2(np.sqrt(2*np.pi* math.e * self.sigma2**2))
        return self.CE64


snr_set = np.arange(-5, 26, 0.5)# dB
optical_power = np.empty_like(snr_set)
for i in range(len(snr_set)):
   optical_power[i] = 10**(snr_set[i] / 10)
#Bob
Cap2 = np.zeros(len(optical_power))
Cap4 = np.zeros(len(optical_power))
Cap8 = np.zeros(len(optical_power))
Cap16 = np.zeros(len(optical_power))
Cap32 = np.zeros(len(optical_power))
Cap64 = np.zeros(len(optical_power))

for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 2)
   Cap2[i] = Cam.Capacity_Bob_2PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 4)
   Cap4[i] = Cam.Capacity_Bob_4PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 8)
   Cap8[i] = Cam.Capacity_Bob_8PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 16)
   Cap16[i] = Cam.Capacity_Bob_16PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 32)
   Cap32[i] = Cam.Capacity_Bob_32PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 64)
   Cap64[i] = Cam.Capacity_Bob_64PAM()
   #print(a[i])

for i in range(len(optical_power)-26):
    if Cap2[i+26] < Cap2[i + 25]:
        Cap2[i + 26] = 1
for i in range(len(optical_power)-26):
    if Cap4[i+26] < Cap4[i + 25]:
        Cap4[i + 26] = 2
for i in range(len(optical_power)-26):
    if Cap8[i+26] < Cap8[i + 25]:
        Cap8[i + 26] = 3
for i in range(len(optical_power)-26):
    if Cap16[i+26] < Cap16[i + 25]:
        Cap16[i + 26] = 4
for i in range(len(optical_power)-26):
    if Cap32[i+26] < Cap32[i + 25]:
        Cap32[i + 26] = 5
for i in range(len(optical_power)-28):
    if Cap64[i+28] < Cap64[i + 27]:
        Cap64[i + 28] = 6

#Eve
Cap2E = np.zeros(len(optical_power))
Cap4E = np.zeros(len(optical_power))
Cap8E = np.zeros(len(optical_power))
Cap16E = np.zeros(len(optical_power))
Cap32E = np.zeros(len(optical_power))
Cap64E = np.zeros(len(optical_power))

for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 2)
   Cap2E[i] = Cam.Capacity_Eve_2PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 4)
   Cap4E[i] = Cam.Capacity_Eve_4PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 8)
   Cap8E[i] = Cam.Capacity_Eve_8PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 16)
   Cap16E[i] = Cam.Capacity_Eve_16PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 32)
   Cap32E[i] = Cam.Capacity_Eve_32PAM()
for i in range(len(optical_power)):
   Cam = PAM_4LED(optical_power[i], 64)
   Cap64E[i] = Cam.Capacity_Eve_64PAM()


for i in range(len(optical_power)-39):
    if Cap2E[i+39] < Cap2E[i + 38]:
        Cap2E[i + 39] = 1
for i in range(len(optical_power)-39):
    if Cap4E[i+39] < Cap4E[i + 38]:
        Cap4E[i + 39] = 2
for i in range(len(optical_power)-39):
    if Cap8E[i+39] < Cap8E[i + 38]:
        Cap8E[i + 39] = 3
for i in range(len(optical_power)-40):
    if Cap16E[i+40] < Cap16E[i + 39]:
        Cap16E[i + 40] = 4
for i in range(len(optical_power)-41):
    if Cap32E[i+41] < Cap32E[i + 40]:
        Cap32E[i + 41] = 5
for i in range(len(optical_power)-41):
    if Cap64E[i+41] < Cap64E[i + 40]:
        Cap64E[i + 41] = 6

# plt.plot(snr_set, Cap2 - Cap2E, '-oc')
# plt.plot(snr_set, Cap4 - Cap4E, '-oy')
# plt.plot(snr_set, Cap8 - Cap8E, '-ok')
# plt.plot(snr_set, Cap16 - Cap16E, '-om')
# plt.plot(snr_set, Cap32 - Cap32E, '-og')
# plt.plot(snr_set, Cap64 - Cap64E, '-ob')
# plt.subplots_adjust(left=0.1, right=2)

plt.plot(snr_set, Cap2, '-oc')
plt.plot(snr_set, Cap4, '-oy')
plt.plot(snr_set, Cap8, '-ok')
plt.plot(snr_set, Cap16, '-om')
plt.plot(snr_set, Cap32, '-og')
plt.plot(snr_set, Cap64, '-ob')
plt.subplots_adjust(left=0.1, right=2)

# plt.plot(snr_set, Cap2E, '-oc')
# plt.plot(snr_set, Cap4E, '-oy')
# plt.plot(snr_set, Cap8E, '-ok')
# plt.plot(snr_set, Cap16E, '-om')
# plt.plot(snr_set, Cap32E, '-og')
# plt.plot(snr_set, Cap64E, '-ob')
# plt.subplots_adjust(left=0.1, right=2)

plt.xlabel('SNR(dB)',size=16)
plt.ylabel('Secrecy Capacity (Bit/s)',size=18)
plt.legend(['2PAM','4PAM','8PAM','16PAM','32PAM','64PAM'],fontsize=15)
plt.grid(True)
plt.show()

