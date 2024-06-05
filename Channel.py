from cmath import pi, sqrt
import math
import numpy as np
import matplotlib.pyplot as plt
import random

#setup for LED
numberLEDs = 4
#size of room 5x5x3, gốc ở chính giữa phòng, dưới đất
LED1 = np.array([-sqrt(2), -sqrt(2), 3]) 
LED2 = np.array([sqrt(2), sqrt(2), 3])
LED3 = np.array([-sqrt(2), sqrt(2), 3])
LED4 = np.array([sqrt(2), sqrt(2), 3])

transmitter_pos = np.array([LED1, LED2, LED3, LED4]) #sửa từ transpose sang
NT = transmitter_pos.shape[0]

LED_semiangle = pi/3 #góc nửa công suất
LED_conversion_factor = 0.44 #hệ số chuyển đổi A/W
LED_beam_angle = (2*pi)/3
#setup for PDs
numberPDs = 1
PD1 = np.array([0, 0, 0.5]) #position of PD(x, y, z)
receiver_pos = np.transpose(PD1)
PD_Ar = 1e-4 #diện tích bề mặt PD
PD_FOV = pi/3 #Field of view
PD_responsivity = 0.54 #hệ số phản hồi
PD_refractiveIndex = 1.5
PD_Ts = 1
SNR = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])

distance = [np.linalg.norm(LED1 - PD1), np.linalg.norm(LED2 - PD1), np.linalg.norm(LED3 - PD1), np.linalg.norm(LED4 - PD1)]
#distance = 3.201
AoI = math.acos(float(3/np.linalg.norm(LED1 - PD1)))#do khoảng cách giữa PD và các LED là bằng nhau nên AoI của PD so với các Led bằng nhau nên chỉ tính 1 lần
#radian
# AoI = 0.356

#Tính gain of the optical concentrator
if AoI < PD_FOV and AoI >= 0:
    goc = float(pow(PD_refractiveIndex/math.sin(PD_FOV), 2))#gain of the optical concentrator
elif AoI > PD_FOV:
    goc = 0
#goc = 3
#Tính L
#Bậc Lambertian = 1
L = float((((1+1)/(2*pi))*math.cos(AoI)))
#L = 0.298
#Tính h
if AoI < pi/3 and AoI >= 0:
    h = float((PD_Ar/pow(np.linalg.norm(LED1 - PD1), 2))*L*1*goc*math.cos(AoI))
elif AoI > pi/3:
    h = 0 



datas = np.array([[1, 1, 1, 1],[-1, -1, -1, -1]])
H = np.transpose(np.ones((1,4))*h)

data = random.choice(datas)
#năng lượng nhận được
Y = pow(np.dot(data, H), 2)
BER = []
numberErrors = 0
numberTransmissions = 10**5
for i in SNR:
    numberErrors = 0
    for j in range(numberTransmissions):
        y = 0
        white_noise = np.random.randn(1)*math.sqrt((NT*h)**2/(10**(i/10)))
        y = np.dot(data, H) + white_noise
        if y > 0:
            d_detected = 1
        elif y < 0:
            d_detected = -1
        if d_detected != data[1]:
            numberErrors += 1
    
    ber = float(numberErrors / (numberTransmissions))
    BER.append(ber)
   # print(ber)    

#print(data)
#plt.semilogy(SNR, BER, marker='o')
#plt.xlabel('SNR')
#plt.ylabel('BER')
#plt.title('BER vs SNR')
#plt.grid(True)
#plt.show()

