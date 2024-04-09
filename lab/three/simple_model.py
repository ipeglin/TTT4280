import numpy as np

base_path = '/Users/ipeglin/Documents/Programming Local/TTT4280/lab/three'
muabo = np.genfromtxt(f"{base_path}/muabo.txt", delimiter=",")
muabd = np.genfromtxt(f"{base_path}/muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 515 # Replace with wavelength in nanometres
blue_wavelength = 460 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth
C = np.sqrt(3 * mua * (musr + mua))

penetration_depth = np.sqrt(1) / C
print(f'Penertrasjonsdybde: {penetration_depth} m')

# material_diameter = 0.00003 # 300 µm
material_diameter = 0.0003 # 300 µm
transmittance = np.exp(- C * material_diameter)
print(f'Transmittanse: {transmittance}')

reflectance = np.sqrt(3 * (musr / mua + 1))
print(f'Reflektanse: {reflectance}')

# BVF = 100%
mua = mua_blood * 1 + mua_other
C = np.sqrt(3 * mua * (musr + mua))

print(f'Penertrasjonsdybde (bvf=1): {penetration_depth} m')

transmittance_hbp = np.exp(- C * material_diameter) # high blood pressure
print(f'Transmittanse (bvf=1): {transmittance_hbp}')

reflectance_hbp = np.sqrt(3 * (musr / mua + 1))
print(f'Reflektanse (bvf=1): {reflectance_hbp}')

contrast = np.abs(transmittance_hbp - transmittance) / transmittance
print('Kontrast:', contrast)