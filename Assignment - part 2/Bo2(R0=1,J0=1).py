import numpy as np
# import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as f_mng

# kieu chu, kich thuoc
plt.figure(figsize=(7, 7))

font = f_mng.FontProperties(family='serif',
                                   weight='normal',
                                   style='normal', size=13)

font1 = {'family': 'serif',
         'color':  'darkred',
         'weight': 'normal',
         'size': 13,
         }

########################*********########################
########################*********########################
######################## ++ x ++ ########################
########################_________########################
# _________######################## #truong hop nay luon co 2 nghiem phan biet
a = 2
b = 5
c = 4
d = 1


R0 = 1
J0 = 1

delta = (a+d)**2 - 4*(a*d-b*c)
L = (a+d)/2

L1 = L + np.sqrt(delta)/2
L2 = L - np.sqrt(delta)/2

# bien t
t = np.linspace(0, 2, 100)
##########################

c1 = (R0 * (L2-a) - J0*b) / (b*(L2-L1))
c2 = (J0*b - R0 * (L1-a)) / (b*(L2-L1))


R = b * (c1*np.e**(L1*t) + c2*np.e**(L2*t))
J = (L1 - a) * c1*np.e**(L1*t) + (L2 - a) * c2*np.e**(L2*t)
plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN EAGER BEAVERS", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN EAGER BEAVERS", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## ++ x +- ########################
########################_________########################
# @@@@@@@@@@@@@@@@@@@@@@ _________######################## #luon co 2 n0 pbiet

a = 1
b = 2
c = 3
d = -4


R0 = 1
J0 = 1

delta = (a+d)**2 - 4*(a*d-b*c)
L = (a+d)/2

L1 = L + np.sqrt(delta)/2
L2 = L - np.sqrt(delta)/2

# bien t
t = np.linspace(0, 2, 100)
##########################

c1 = (R0 * (L2-a) - J0*b) / (b*(L2-L1))
c2 = (J0*b - R0 * (L1-a)) / (b*(L2-L1))


R = b * (c1*np.e**(L1*t) + c2*np.e**(L2*t))
J = (L1 - a) * c1*np.e**(L1*t) + (L2 - a) * c2*np.e**(L2*t)
plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN AN EAGER BEAVER \nAND A NARCISSISTIC NERD", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN AN EAGER BEAVER \nAND A NARCISSISTIC NERD", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## ++ x -+ ########################
########################_________########################
# _________######################## #co th n0 kep

a = 2
b = 1
c = -1
d = 4


R0 = 1
J0 = 1

L = (a+d)/2

# bien t
t = np.linspace(0, 2, 100)
##########################

c1 = J0*b - R0*L + a*R0
c2 = R0


R = (c1*t + c2) * np.e**(L*t)
J = ((c*c1) / (L-d)) * t * np.e ** (L*t) + (c1 + L*c2 - a*c2)/b

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN AN EAGER BEAVER \nAND A CAUTIOUS LOVER", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN AN EAGER BEAVER \nAND A CAUTIOUS LOVER", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## ++ x -- ########################
########################_________########################
# _________######################## #co n0 kep

a = 2
b = 1/2
c = -9/2
d = -1


R0 = 1
J0 = 1

L = (a+d)/2

# bien t
t = np.linspace(0, 2, 100)
##########################

c1 = J0*b - R0*L + a*R0
c2 = R0


R = (c1*t + c2) * np.e**(L*t)
J = ((c*c1) / (L-d)) * t * np.e ** (L*t) + (c1 + L*c2 - a*c2)/b

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN AN EAGER BEAVER AND A HERMIT", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN AN EAGER BEAVER AND A HERMIT", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## +- x +- ########################
########################_________########################
# _________######################## # co n0 kep

a = 5
b = -3
c = 3
d = -1


R0 = 1
J0 = 1

L = (a+d)/2

# bien t
t = np.linspace(0, 2, 100)
##########################

c1 = J0*b - R0*L + a*R0
c2 = R0


R = (c1*t + c2) * np.e**(L*t)
J = ((c*c1) / (L-d)) * t * np.e ** (L*t) + (c1 + L*c2 - a*c2)/b

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN NARCISSISTIC NERDS", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN NARCISSISTIC NERDS", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## +- x -+ ########################
########################_________########################
# _________######################## #luon 2 n0 pbiet

a = 1
b = -2
c = -3
d = 4


R0 = 1
J0 = 1

delta = (a+d)**2 - 4*(a*d-b*c)
L = (a+d)/2

L1 = L + np.sqrt(delta)/2
L2 = L - np.sqrt(delta)/2

# bien t
t = np.linspace(0, 2, 100)
##########################

c1 = (R0 * (L2-a) - J0*b) / (b*(L2-L1))
c2 = (J0*b - R0 * (L1-a)) / (b*(L2-L1))


R = b * (c1*np.e**(L1*t) + c2*np.e**(L2*t))
J = (L1 - a) * c1*np.e**(L1*t) + (L2 - a) * c2*np.e**(L2*t)

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN A NARCISSISTIC NERD \nAND A CAUTIOUS LOVER", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN A NARCISSISTIC NERD \nAND A CAUTIOUS LOVER", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## +- x -- ########################
########################_________########################
# _________######################## #luon co n0 pbiet

a = 1
b = -2
c = -3
d = -4


R0 = 1
J0 = 1

delta = (a+d)**2 - 4*(a*d-b*c)
L = (a+d)/2

L1 = L + np.sqrt(delta)/2
L2 = L - np.sqrt(delta)/2

# bien t
t = np.linspace(0, 2, 100)
##########################

c1 = (R0 * (L2-a) - J0*b) / (b*(L2-L1))
c2 = (J0*b - R0 * (L1-a)) / (b*(L2-L1))


R = b * (c1*np.e**(L1*t) + c2*np.e**(L2*t))
J = (L1 - a) * c1*np.e**(L1*t) + (L2 - a) * c2*np.e**(L2*t)

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN A NARCISSISTIC NERD \nAND A HERMIT", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN A NARCISSISTIC NERD \nAND A HERMIT", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## -+ x -+ ########################
########################_________########################
# _________######################## #co n0 kep

a = -4
b = 1
c = -9
d = 2


R0 = 1
J0 = 1

L = (a+d)/2

# bien t
t = np.linspace(0, 10, 100)
##########################

c1 = J0*b - R0*L + a*R0
c2 = R0


R = (c1*t + c2) * np.e**(L*t)
J = ((c*c1) / (L-d)) * t * np.e ** (L*t) + (c1 + L*c2 - a*c2)/b

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN CAUTIOUS LOVERS", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN CAUTIOUS LOVERS", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## -+ x -- ########################
########################_________########################
# _________######################## #co n0 kep

a = -3
b = 2
c = -1/2
d = -1


R0 = 1
J0 = 1

L = (a+d)/2

# bien t
t = np.linspace(0, 10, 100)
##########################

c1 = J0*b - R0*L + a*R0
c2 = R0


R = (c1*t + c2) * np.e**(L*t)
J = ((c*c1) / (L-d)) * t * np.e ** (L*t) + (c1 + L*c2 - a*c2)/b

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN A CAUTIOUS LOVER \nAND A HERMIT", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN A CAUTIOUS LOVER \nAND A HERMIT", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()

########################*********########################
########################*********########################
######################## -- x -- ########################
########################_________########################
# _________######################## #ko the co n0 kep

a = -4
b = -3
c = -2
d = -1


R0 = 1
J0 = 1

delta = (a+d)**2 - 4*(a*d-b*c)
L = (a+d)/2

L1 = L + np.sqrt(delta)/2
L2 = L - np.sqrt(delta)/2

# bien t
t = np.linspace(0, 5, 100)
##########################

c1 = (R0 * (L2-a) - J0*b) / (b*(L2-L1))
c2 = (J0*b - R0 * (L1-a)) / (b*(L2-L1))


R = b * (c1*np.e**(L1*t) + c2*np.e**(L2*t))
J = (L1 - a) * c1*np.e**(L1*t) + (L2 - a) * c2*np.e**(L2*t)

plt.plot(t, R)
plt.plot(t, J)
# title
plt.title("LOVE BETWEEN HERMITS", fontdict=font1)
# label
plt.xlabel("Time", fontdict=font1)
plt.ylabel("Love for the other", fontdict=font1)
plt.legend(["Romeo's", "Juliet's"], loc="best")  # , prop=font)
plt.show()

# Trajectory x va y
plt.plot(R, J, color="red")
plt.xlim([-5, 5])
plt.ylim([-5, 5])

R = np.linspace(-5, 5, 11)
# nullcline f', g' = 0
plt.plot(R, R*(-a)/b, linestyle="--", color="blue")
plt.plot(R, R*(-c)/d, linestyle="--", color="purple")

R, J = np.meshgrid(np.linspace(-4.5, 4.5, 11), np.linspace(-4.5, 4.5, 11))
# f', g' plotting
RR = a * R + b * J
JJ = c * R + d * J

Rsqrt = np.hypot(RR, JJ)
Rsqrt[Rsqrt == 0] = 1
RR /= Rsqrt
JJ /= Rsqrt

# title
plt.title("LOVE BETWEEN HERMITS", fontdict=font1)
# label
plt.xlabel("Romeo's love for Juliet", fontdict=font1)
plt.ylabel("Juliet's love for Romeo", fontdict=font1)
# truong vector
plt.quiver(R, J, RR, JJ, color='g', width=0.004)
# do thi R', J'
plt.streamplot(R, J, RR, JJ, density=0.2, linewidth=1, color="gray",
               broken_streamlines=False)
plt.legend(["Trajectory", "Nullcline 1", "Nullcline 2", "Vector field",
            "Trajectory"], loc="lower left")
plt.show()
