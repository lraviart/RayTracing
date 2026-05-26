import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "font.family": "serif",
    "mathtext.fontset": "cm",  # This gives the LaTeX look to math text
    "axes.edgecolor": "black", # Black bounding box
    "axes.linewidth": 0.8
})


def reflection_coeff(epsilon_r, c, d, f, z, angle):

    epsilon_0 = 8.854187817e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f

    sigma = c * (f/1e9)**d
    eta = epsilon_r - 1j * sigma / (omega * epsilon_0)

    r = (np.cos(angle) - np.sqrt(eta - np.sin(angle)**2)) / (np.cos(angle) + np.sqrt(eta - np.sin(angle)**2))
    q = (2 * np.pi * f / 3e8) * z * np.sqrt(eta - np.sin(angle)**2)

    return r * (1 - np.exp(-1j * 2 * q)) / (1 - r**2 * np.exp(-1j * 2 * q))


def transmission_coeff(epsilon_r, c, d, f, z, angle):

    epsilon_0 = 8.854187817e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f

    sigma = c * (f/1e9)**d
    eta = epsilon_r - 1j * sigma / (omega * epsilon_0)

    r = (np.cos(angle) - np.sqrt(eta - np.sin(angle)**2)) / (np.cos(angle) + np.sqrt(eta - np.sin(angle)**2))
    q = (2 * np.pi * f / 3e8) * z * np.sqrt(eta - np.sin(angle)**2)

    return (1 - r**2) * np.exp(-1j * q) / (1 - r**2 * np.exp(-1j * 2 * q))



# Concrete
e_r_concrete = 5.24
c_concrete = 0.0462
d_concrete = 0.7822
frequencies = np.linspace(1e9, 30e9, 100)
z = 0.2

if True:
    plt.figure(figsize=(8, 6))
    plt.plot(frequencies/1e9, 20*np.log10(np.abs(transmission_coeff(e_r_concrete, c_concrete, d_concrete, frequencies, z, 0))), color='#0060DE')
    # plt.title('Attenuation through Concrete Wall')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel('Transmission [dB]')
    plt.xlim(-2, 32)
    plt.ylim(-125, 5)
    plt.xticks(np.arange(0, 31, 5))
    plt.yticks(np.arange(-120, 1, 20))
    plt.grid()
    plt.savefig("concrete_attenuation.pdf", bbox_inches='tight')
    plt.show()

f = 3.5e9
z = np.linspace(0.1, 0.3, 1000)
if True:
    plt.figure(figsize=(8, 6))
    plt.plot(z, 20*np.log10(np.abs(transmission_coeff(e_r_concrete, c_concrete, d_concrete, f, z, 0))), color='#0060DE')
    # plt.title('Transmission through Concrete Wall vs Thickness')
    plt.xlabel('Thickness [m]')
    plt.ylabel('Transmission [dB]')
    plt.xlim(0.09, 0.31)
    plt.ylim(-31, -9)
    plt.xticks(np.arange(0.1, 0.31, 0.05))
    plt.yticks(np.arange(-30, -9, 5))
    plt.grid()
    plt.savefig("concrete_transmission.pdf", bbox_inches='tight')
    plt.show()

angle = np.arange(0, np.pi/2, 0.01)
if True:
    plt.figure(figsize=(8, 6))
    plt.plot(angle*180/np.pi, 20*np.log10(np.abs(transmission_coeff(e_r_concrete, c_concrete, d_concrete, f, 0.2, angle))), color='#0060DE')
    # plt.title('Transmission through Concrete Wall vs Angle of Incidence')
    plt.xlabel('Angle of Incidence [°]')
    plt.ylabel('Transmission [dB]')
    plt.xlim(-5, 95)
    plt.ylim(-75, -15)
    plt.grid()
    plt.savefig("concrete_transmission_vs_angle.pdf", bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(angle*180/np.pi, 20*np.log10(np.abs(reflection_coeff(e_r_concrete, c_concrete, d_concrete, f, 0.2, angle))), color='#0060DE')
    # plt.title('Reflection through Concrete Wall vs Angle of Incidence')
    plt.xlabel('Angle of Incidence [°]')
    plt.ylabel('Reflection [dB]')
    plt.xlim(-5, 95)
    plt.ylim(-11, 3)
    plt.grid()
    plt.savefig("concrete_reflection_vs_angle.pdf", bbox_inches='tight')
    plt.show()

angle = 19.29 * np.pi / 180
print(20*np.log10(np.abs(reflection_coeff(e_r_concrete, c_concrete, d_concrete, f, 0.2, angle))))
print(20*np.log10(np.abs(transmission_coeff(e_r_concrete, c_concrete, d_concrete, f, 0.2, angle))))

# Glass
e_r_glass = 6.31
c_glass = 0.0036
d_glass = 1.3394
z = np.linspace(0.04, 0.18, 1000)

if True:
    plt.figure(figsize=(8, 6))
    plt.plot(z*1000, 20*np.log10(np.abs(transmission_coeff(e_r_glass, c_glass, d_glass, f, z, 0))), color='#0060DE')
    # plt.title('Transmission through Glass vs Thickness')
    plt.xlabel('Thickness [mm]')
    plt.ylabel('Transmission [dB]')
    plt.xlim(38, 182)
    plt.ylim(-5.25, 0.5)
    plt.xticks(np.arange(40, 181, 20))
    plt.yticks(np.arange(-5, 1, 1))
    plt.grid()
    plt.savefig("glass_transmission.pdf", bbox_inches='tight')
    plt.show()