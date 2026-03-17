import numpy as np
import matplotlib.pyplot as plt


def loss(epsilon_r, c, d, f, z):

    epsilon_0 = 8.854187817e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f

    sigma = c * (f/1e9)**d
    eta = epsilon_r - 1j * sigma / (omega * epsilon_0)
    gamma = 1j * omega * np.sqrt(mu_0 * epsilon_0 * eta)

    return np.exp(-np.real(gamma) * z)


def reflection_coeff(epsilon_r, c, d, f, z, angle):

    epsilon_0 = 8.854187817e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f

    sigma = c * (f/1e9)**d
    eta = epsilon_r - 1j * sigma / (omega * epsilon_0)

    r = (eta * np.cos(angle) - np.sqrt(eta - np.sin(angle)**2)) / (eta * np.cos(angle) + np.sqrt(eta - np.sin(angle)**2))
    q = (2 * np.pi * f / 3e8) * z * np.sqrt(eta - np.sin(angle)**2)

    return r * (1 - np.exp(-1j * 2 * q)) / (1 - r**2 * np.exp(-1j * 2 * q))


def transmission_coeff(epsilon_r, c, d, f, z, angle):

    epsilon_0 = 8.854187817e-12
    mu_0 = 4 * np.pi * 1e-7
    omega = 2 * np.pi * f

    sigma = c * (f/1e9)**d
    eta = epsilon_r - 1j * sigma / (omega * epsilon_0)

    r = (eta * np.cos(angle) - np.sqrt(eta - np.sin(angle)**2)) / (eta * np.cos(angle) + np.sqrt(eta - np.sin(angle)**2))
    q = (2 * np.pi * f / 3e8) * z * np.sqrt(eta - np.sin(angle)**2)

    return (1 - r**2) * np.exp(-1j * q) / (1 - r**2 * np.exp(-1j * 2 * q))


print(20*np.log10(loss(5.24, 0.0462, 0.7822, 3.5e9, 0.1)))
print(20*np.log10(loss(5.24, 0.0462, 0.7822, 2.4e9, 0.1)))
print(20*np.log10(loss(3.91, 0.0238, 0.16, 3.5e9, 0.1)))


# Concrete
e_r_concrete = 5.24
c_concrete = 0.0462
d_concrete = 0.7822
frequencies = np.linspace(1e9, 100e9, 100)
z = 0.1

plt.figure(figsize=(8, 6))
plt.plot(frequencies/1e9, 20*np.log10(loss(e_r_concrete, c_concrete, d_concrete, frequencies, z)))
plt.title('Loss through Concrete Wall')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Loss (dB)')
plt.grid()
plt.show()

f = 3.5e9
z = np.linspace(0.1, 0.3, 100)
plt.figure(figsize=(8, 6))
plt.plot(z, 20*np.log10(loss(e_r_concrete, c_concrete, d_concrete, f, z)))
plt.title('Loss through Concrete Wall vs Thickness')
plt.xlabel('Thickness (m)')
plt.ylabel('Loss (dB)')
plt.grid()
plt.show()

print(20*np.log10(np.abs(reflection_coeff(e_r_concrete, c_concrete, d_concrete, f, z, 0))))
print(20*np.log10(np.abs(transmission_coeff(e_r_concrete, c_concrete, d_concrete, f, z, 0))))

# Brick
e_r_brick = 3.91
c_brick = 0.0238
d_brick = 0.16
frequencies = np.linspace(1e9, 10e9, 100)
z = 0.1

plt.figure(figsize=(8, 6))
plt.plot(frequencies/1e9, 20*np.log10(loss(e_r_brick, c_brick, d_brick, frequencies, z)))
plt.title('Loss through Brick Wall')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Loss (dB)')
plt.grid()
plt.show()

f = 3.5e9
z = np.linspace(0.1, 0.5, 100)
plt.figure(figsize=(8, 6))
plt.plot(z, 20*np.log10(loss(e_r_brick, c_brick, d_brick, f, z)))
plt.title('Loss through Brick Wall vs Thickness')
plt.xlabel('Thickness (m)')
plt.ylabel('Loss (dB)')
plt.grid()
plt.show()
