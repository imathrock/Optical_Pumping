#!/usr/bin/env python
# coding: utf-8

# I met a traveller from an antique land,
# Who saidâ€”â€œTwo vast and trunkless legs of stone
# Stand in the desert. . . . Near them, on the sand,
# Half sunk a shattered visage lies, whose frown,
# And wrinkled lip, and sneer of cold command,
# Tell that its sculptor well those passions read
# Which yet survive, stamped on these lifeless things,
# The hand that mocked them, and the heart that fed;
# And on the pedestal, these words appear:
# 
# My name is Ozymandias, King of Kings;
# Look on my Works, ye Mighty, and despair!"
# 
# Nothing beside remains. Round the decay
# Of that colossal Wreck, boundless and bare
# The lone and level sands stretch far away.

# In[1054]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties
import os


# Most of these datasets do not have B field in them and the channel 2 output of transmissivity is not normalized, since it is in arbitrary units we would need a function to normalize them. 
# 
# Another thing to note is that fitting to the current readings of the sweep fields would not exactly be feasible because it's all jimble jambled, what I mean to say is that the current readings have high noise such that fitting a lorentzian is not exactly possible, but what we care about is only the peaks and that means we only need to know the value of current near the transmissivity peak and we can include uncertainties within it. 

# In[1055]:


# Converting the sweep values to b field values:
def add_magnetic_field(filename):
    ItB = (0.6*1e-4)#Teslas
    header_info = pd.read_csv(filename, skiprows=1, nrows=1, header=None)
    timeconst = float(header_info.iloc[0, 4])
    df = pd.read_csv(filename, skiprows=2, usecols=[0, 1, 2], names=["Sequence", "CH1", "CH2"])
    df["Bfield"] = (df["CH1"]*ItB)/10
    df["CH2norm"] = (df["CH2"] - min(df["CH2"]))/(max(df["CH2"])-min(df["CH2"]))
    df["time"] = df["Sequence"]*timeconst
    filename, ext = os.path.splitext(filename)
    new_filename = f"{filename}_B{ext}"
    df.to_csv(new_filename, index=False)
    print("New file:",new_filename) 
    return df


# Now using this function to polish the datasets. 

# In[1056]:


df_25 = add_magnetic_field("linear_zeeman_data/25kHz.csv")
df_50 = add_magnetic_field("linear_zeeman_data/50kHz.csv")
df_75 = add_magnetic_field("linear_zeeman_data/75kHz.csv")
df_100 = add_magnetic_field("linear_zeeman_data/100kHz.csv")
df_125 = add_magnetic_field("linear_zeeman_data/125kHz.csv")
df_150 = add_magnetic_field("linear_zeeman_data/150kHz.csv")
df_175 = add_magnetic_field("linear_zeeman_data/175kHz.csv")
df_195 = add_magnetic_field("linear_zeeman_data/195kHz.csv")
# df_300 = add_magnetic_field("linear_zeeman_data/300kHz_5s.csv")

df = [df_25,df_50,df_75,df_100,df_125,df_150,df_175,df_195]


# Now we gotta fit lorentzians to this data to extract the peak positions along with their uncertainties.
# #### Lorentzian fit of the data:
# 
# $$f(x;A,x_0,\gamma) = A\cdot\frac{\gamma^2}{\gamma^2+(x-x_0)^2}$$

# In[1057]:


# All kind of lorentzians that one might need, all in one place!!

def lorentzian(x, y0, a1, c1, w1):
    L1 = a1 / (1 + ((x - c1) / w1)**2)
    return y0 + L1

def duo_lorentzian(x, y0, a1, c1, w1, a2, c2, w2):
    L1 = a1 / (1 + ((x - c1) / w1)**2)
    L2 = a2 / (1 + ((x - c2) / w2)**2)
    return y0 + L1 + L2

def tri_lorentzian(x, y0, a1, c1, w1, a2, c2, w2, a3, c3, w3):
    L1 = a1 / (1 + ((x - c1) / w1)**2)
    L2 = a2 / (1 + ((x - c2) / w2)**2)
    L3 = a3 / (1 + ((x - c3) / w3)**2)
    return y0 + L1 + L2 + L3

def quad_lorentzian(x, y0, a1, c1, w1, a2, c2, w2, a3, c3, w3, a4, c4, w4):
    L1 = a1 / (1 + ((x - c1) / w1)**2)
    L2 = a2 / (1 + ((x - c2) / w2)**2)
    L3 = a3 / (1 + ((x - c3) / w3)**2)
    L4 = a4 / (1 + ((x - c4) / w4)**2)
    return y0 + L1 + L2 + L3 + L4

def penta_lorentzian(x, y0, a1, c1, w1, a2, c2, w2, a3, c3, w3, a4, c4, w4, a5, c5, w5):
    L1 = a1 / (1 + ((x - c1) / w1)**2)
    L2 = a2 / (1 + ((x - c2) / w2)**2)
    L3 = a3 / (1 + ((x - c3) / w3)**2)
    L4 = a4 / (1 + ((x - c4) / w4)**2)
    L5 = a5 / (1 + ((x - c5) / w5)**2)
    return y0 + L1 + L2 + L3 + L4 + L5

def hexa_lorentzian(x, y0, a1, c1, w1, a2, c2, w2, a3, c3, w3, a4, c4, w4, a5, c5, w5, a6, c6, w6):
    L1 = a1 / (1 + ((x - c1) / w1)**2)
    L2 = a2 / (1 + ((x - c2) / w2)**2)
    L3 = a3 / (1 + ((x - c3) / w3)**2)
    L4 = a4 / (1 + ((x - c4) / w4)**2)
    L5 = a5 / (1 + ((x - c5) / w5)**2)
    L6 = a6 / (1 + ((x - c6) / w6)**2)
    return y0 + L1 + L2 + L3 + L4 + L5 + L6 


# So the lorentzian peaks we would get are not the magnetic field values, what we have is time, we need to correlate the time with the magnetic field values and that means we need to write a new function that correlates the time to magnetic field using the current value measured on channel 1. 

# In[1058]:


def get_bfield_data(df, center_time, sigma_t, window_size=0.005, result_list=None):
    mask = (df['time'] >= center_time - window_size) & (df['time'] <= center_time + window_size)
    sub = df.loc[mask]

    # Dynamically determine the dataset's resolution (quantization step)
    # This captures the oscilloscope's actual Volts/Div zoom level
    if len(sub) > 1:
        b_diffs = np.abs(np.diff(sub['Bfield']))
        non_zero_diffs = b_diffs[b_diffs > 1e-12] # Filter floating-point zeroes

        if len(non_zero_diffs) > 0:
            dynamic_resolution = np.min(non_zero_diffs)
        else:
            dynamic_resolution = 0.32 * (0.6 * 1e-4) / 10 # Fallback
    else:
        dynamic_resolution = 0.32 * (0.6 * 1e-4) / 10 # Fallback

    vtb_unc_term = 0.0
    if 'voltage' in df.columns:
        voltage = df['voltage'].iloc[0]
        vtb_unc_term = (voltage * VtB[1])**2

    if sub.empty or len(sub) < 3:
        mean_b = np.nan if sub.empty else sub['Bfield'].mean()
        unc_b = np.sqrt(dynamic_resolution**2 + vtb_unc_term)
    else:
        mean_b = sub['Bfield'].mean()

        # Linear fit to isolate the sweep slope from sensor noise
        p = np.polyfit(sub['time'], sub['Bfield'], 1)
        dBdt = p[0]
        intercept = p[1]

        # Propagated uncertainty from the Lorentzian time center fit
        unc_propagated = abs(dBdt) * sigma_t

        # Statistical noise (Standard Error of the Mean of the residuals)
        residuals = sub['Bfield'] - (dBdt * sub['time'] + intercept)
        std_residuals = np.std(residuals, ddof=2)
        sem_residuals = std_residuals / np.sqrt(len(sub))

        # Combine dynamic instrumental resolution with statistical and propagated errors
        unc_b = np.sqrt(unc_propagated**2 + sem_residuals**2 + dynamic_resolution**2 + vtb_unc_term)

    if result_list is not None:
        result_list.append([mean_b, unc_b])

    return mean_b, unc_b


# In[1059]:


def fit_lorentzians(df, p0, num, result_list=None):
    x_data, y_data = df["time"], df["CH2norm"] 

    func_map = {
        1: lorentzian,
        2: duo_lorentzian,
        3: tri_lorentzian,
        4: quad_lorentzian,
        5: penta_lorentzian,
        6: hexa_lorentzian
    }

    if num not in func_map:
        print(f"Unknown lorentzian to fit for num={num}")
        return [], []

    fit_func = func_map[num]
    popt, pcov = curve_fit(fit_func, x_data, y_data, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    b_means = []
    b_uncs = []
    init_centers = []
    init_amps = []

    for i in range(num):
        center_idx = 2 + (i * 3)
        amp_idx = 1 + (i * 3)

        center_time = popt[center_idx]
        sigma_t = perr[center_idx]

        init_centers.append(p0[center_idx])
        init_amps.append(p0[amp_idx])

        # Extracts and stores B-field data for ALL peaks
        mean_b, unc_b = get_bfield_data(df, center_time, sigma_t)
        b_means.append(mean_b)
        b_uncs.append(unc_b)

        print(f"Lorentzian peak {i+1}: {center_time:.4f} Â± {sigma_t:.4f} | B-field: {mean_b:.4e} Â± {unc_b:.4e}")

    plt.figure(figsize=(7, 4), dpi=1200)
    plt.plot(x_data, y_data, label='Measured Data', alpha=0.7)
    plt.plot(x_data, fit_func(x_data, *popt), 'r-', linewidth=2, label=f'Fitted {num}-Lorentzian')
    plt.scatter(init_centers, init_amps, color='black', marker='x', label='Initial Guesses', zorder=5)
    plt.xlabel('Time (s)')
    plt.ylabel('CH2norm (Normalized Intensity)')
    # plt.title(f'{num}-Lorentzian Multi-Peak Fit')
    plt.grid(True, which='both', linestyle='--', alpha=0.1)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    if result_list is not None:
        result_list.append(b_means + b_uncs)

    return popt, perr


# In[1060]:


# List of optimal values with uncertainties
lopt = []
# List of all guesses for the linear regieme
p0guesses = [[1.0,0,2.025,0.018,0.2,2.5,0.023,0.6,2.3,0.022],
             [1.0,0,2.025,0.018,0.2,3.2,0.023,0.6,2.8,0.022],
             [1.0,0,2.025,0.018,0.2,3.6,0.023,0.6,3.1,0.022],
             [1.0,0,2.025,0.018,0.2,4.1,0.023,0.6,3.4,0.022],
             [1.0,0,2.025,0.018,0.2,4.5,0.023,0.6,3.7,0.022],
             [1.0,0,2.025,0.018,0.2,5.0,0.023,0.6,4.1,0.022],
             [1.0,0,2.025,0.018,0.2,5.5,0.023,0.6,4.4,0.022],
             [1.0,0,2.025,0.018,0.2,5.7,0.023,0.6,4.5,0.022]]


# #### Fitting all the lorentzians

# In[1061]:


for p0, df_ in zip(p0guesses, df):
    fit_lorentzians(df_,p0,3,lopt)


# In[1062]:


freq_list = [25,50,75,100,125,150,175,195]
lopt = np.array(lopt)
# Add freq_list as first column
lopt = np.column_stack((freq_list, lopt))


# In[1063]:


freq = lopt[:, 0]
mean_b2, unc_b2 = lopt[:, 2], lopt[:, 5]
mean_b3, unc_b3 = lopt[:, 3], lopt[:, 6]

plt.figure(figsize=(8, 5))

plt.errorbar(freq, mean_b2, yerr=unc_b2, fmt='o', color='blue', 
             capsize=4, elinewidth=1.5, alpha=0.8, label='Peak 2 B-field')

plt.errorbar(freq, mean_b3, yerr=unc_b3, fmt='s', color='red', 
             capsize=4, elinewidth=1.5, alpha=0.8, label='Peak 3 B-field')

plt.xlabel('Frequency (kHz)')
plt.ylabel('Magnetic Field (T)')
plt.title('Magnetic Field of Peaks 2 and 3 vs. Frequency')
plt.grid(True, linestyle='--', alpha=0.1)
plt.legend(loc='best')
plt.tight_layout()
plt.show()


# In[1064]:


def linear_model(x, m, b):
    return m * x + b

# Extract data using the index mapping
freq = lopt[:, 0]
mean_b2, unc_b2 = lopt[:, 2], lopt[:, 5]
mean_b3, unc_b3 = lopt[:, 3], lopt[:, 6]

# Perform weighted linear fits (sigma accounts for the uncertainties)
popt2, pcov2 = curve_fit(linear_model, freq, mean_b2, sigma=unc_b2, absolute_sigma=True)
popt3, pcov3 = curve_fit(linear_model, freq, mean_b3, sigma=unc_b3, absolute_sigma=True)

# Calculate parameter uncertainties
err2 = np.sqrt(np.diag(pcov2))
err3 = np.sqrt(np.diag(pcov3))

# Calculate residuals (Observed - Fitted)
res2 = mean_b2 - linear_model(freq, *popt2)
res3 = mean_b3 - linear_model(freq, *popt3)

# Calculate Chi-Squared and Reduced Chi-Squared
dof = len(freq) - 2  # Degrees of freedom: N points - 2 parameters (linear)

chi2_2 = np.sum((res2 / unc_b2)**2)
red_chi2_2 = chi2_2 / dof

chi2_3 = np.sum((res3 / unc_b3)**2)
red_chi2_3 = chi2_3 / dof

# Initialize figure. height_ratios changed from [3, 1] to [5, 1] to shrink residuals plot.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [5, 1]}, sharex=True)

# --- Top Subplot: Main Data and Fits ---
ax1.errorbar(freq, mean_b2, yerr=unc_b2, fmt='o', color='blue', capsize=4, alpha=0.8, label='Peak 2 Data')
ax1.errorbar(freq, mean_b3, yerr=unc_b3, fmt='s', color='red', capsize=4, alpha=0.8, label='Peak 3 Data')

# Generate smooth line for fits
freq_fit = np.linspace(min(freq), max(freq), 100)

# Format parameter labels to include Reduced Chi-Squared
fit_label2 = (f'Fit 2: y = ({popt2[0]:.2e} Â± {err2[0]:.1e})x + ({popt2[1]:.2e} Â± {err2[1]:.1e})\n'
              f'$\chi^2_\\nu$ = {red_chi2_2:.2f}')
fit_label3 = (f'Fit 3: y = ({popt3[0]:.2e} Â± {err3[0]:.1e})x + ({popt3[1]:.2e} Â± {err3[1]:.1e})\n'
              f'$\chi^2_\\nu$ = {red_chi2_3:.2f}')

ax1.plot(freq_fit, linear_model(freq_fit, *popt2), 'b-', linewidth=1.5, label=fit_label2)
ax1.plot(freq_fit, linear_model(freq_fit, *popt3), 'r-', linewidth=1.5, label=fit_label3)

ax1.set_ylabel('Magnetic Field (T)')
ax1.set_title('Magnetic Field of Peaks 2 and 3 vs. Frequency')
ax1.grid(True, linestyle='--', alpha=0.1)
ax1.legend(loc='best', fontsize=9)

# --- Bottom Subplot: Residuals ---
ax2.axhline(0, color='black', linestyle='--', linewidth=1, zorder=1)
ax2.errorbar(freq, res2, yerr=unc_b2, fmt='o', color='blue', capsize=4, alpha=0.8, zorder=2)
ax2.errorbar(freq, res3, yerr=unc_b3, fmt='s', color='red', capsize=4, alpha=0.8, zorder=3)

ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('Residuals (T)')
ax2.grid(True, linestyle='--', alpha=0.1)

plt.tight_layout()
plt.show()


# #### Now Entering the quadratic regime
# 
# What I am going to have to do is caliberate the voltage reading of the offset magnetic field to the voltage reading we took and add it to the whole magnetic field value as offset.

# In[1065]:


# Converting the sweep values to b field values:
def add_magnetic_field(filename):
    ItB = (0.6*1e-4)#Teslas
    header_info = pd.read_csv(filename, skiprows=1, nrows=1, header=None)
    timeconst = float(header_info.iloc[0, 4])
    df = pd.read_csv(filename, skiprows=2, usecols=[0, 1, 2], names=["Sequence", "CH1", "CH2"])
    df["Bfield"] = (df["CH1"]*ItB)
    df["CH2norm"] = (df["CH2"] - min(df["CH2"]))/(max(df["CH2"])-min(df["CH2"]))
    df["time"] = df["Sequence"]*timeconst
    filename, ext = os.path.splitext(filename)
    new_filename = f"{filename}_B{ext}"
    df.to_csv(new_filename, index=False)
    print("New file:",new_filename) 
    return df


# In[1066]:


c1 = add_magnetic_field("offsetfieldcalliberation/c1-330kHz_0p1mV.csv")
c2 = add_magnetic_field("offsetfieldcalliberation/c2-330kHz_25p1mV.csv")
c3 = add_magnetic_field("offsetfieldcalliberation/c3-330kHz_49p5mV.csv")
c4 = add_magnetic_field("offsetfieldcalliberation/c4-330kHz_25p4mV.csv")
c5 = add_magnetic_field("offsetfieldcalliberation/c5-330kHz_25p4mV.csv")


# In[1067]:


copt = []
fit_lorentzians(c1,[1.0,0.0,1.5,0.001],1,copt)
fit_lorentzians(c2,[1.0,0.5,0.5,0.001,0.0,5.1,0.001],2)
fit_lorentzians(c3,[1.0,0.0,3.5,0.001],1,copt)
fit_lorentzians(c4,[1.0,0.0,3.5,0.001],1,copt)
fit_lorentzians(c5,[1.0,0.0,3.5,0.001],1,copt)
print(copt)


# In[1068]:


copt = np.array(copt)
m0, u0 = copt[0]
m1, u1 = copt[1]
m2, u2 = copt[2]
m3, u3 = copt[3]

m_diff = m2 - m3
u_diff = np.sqrt(u2**2 + u3**2)

m_p1 = m0 + m_diff
u_p1 = np.sqrt(u0**2 + u_diff**2)

m_p2 = m1
u_p2 = u1

mean_VtB = (m_p1 - m_p2) / 49.5
unc_VtB = np.sqrt(u_p1**2 + u_p2**2) / 49.5

VtB = np.array([mean_VtB, unc_VtB])


# In[1069]:


# Converting the sweep values to b field values:
def add_magnetic_field_woff(filename,voltage):
    ItB = (0.6*1e-4)#Teslas
    header_info = pd.read_csv(filename, skiprows=1, nrows=1, header=None)
    timeconst = float(header_info.iloc[0, 4])
    df = pd.read_csv(filename, skiprows=2, usecols=[0, 1, 2], names=["Sequence", "CH1", "CH2"])
    df["voltage"] = voltage
    df["Bfield"] = (df["CH1"]*ItB)
    df["Bfield"] = df["Bfield"] + (VtB[0]*voltage)
    df["CH2norm"] = (df["CH2"] - min(df["CH2"]))/(max(df["CH2"])-min(df["CH2"]))
    df["time"] = df["Sequence"]*timeconst
    filename, ext = os.path.splitext(filename)
    new_filename = f"{filename}_B{ext}"
    df.to_csv(new_filename, index=False)
    print("New file:",new_filename) 
    return df


# In[1070]:


# df_300 = add_magnetic_field_woff("linear_zeeman_data/300kHz_5s.csv",20)
# df_400 = add_magnetic_field_woff("linear_zeeman_data/400kHz_33p36mV.csv",33.36)
# df_500 = add_magnetic_field_woff("linear_zeeman_data/500kHz_5s.csv",42)
# df_600 = add_magnetic_field_woff("linear_zeeman_data/600kHz_5s.csv",55)
# df_doub = [df_300,df_400,df_500,df_600]


# In[1071]:


# lopt2 = []
# p0gues0 = [[1.0,0.0,2.5,0.001,0.0,5.1,0.001],
#            [1.0,0.0,1.5,0.001,0.0,4.8,0.001],
#            [1.0,0.0,1.3,0.001,0.0,4.9,0.001],
#            [1.0,0.2,0.75,0.001,0.0,4.8,0.001]]

# for p0,df_ in zip(p0gues0,df_doub):
#     fit_lorentzians(df_,p0,2,lopt2)


# In[1072]:


# lopt2 = np.array(lopt2, ndmin=2)
# lopt2[:, [0, 1]] = lopt2[:, [1, 0]]

# freq_list2 = np.array([300, 400, 500, 600])
# lopt2 = np.column_stack((freq_list2, lopt2))


# In[1073]:


# loptk = lopt[:, [0, 2, 3, 5, 6]]
# loptk = np.vstack((loptk, lopt2))


# In[1074]:


# def quad_model(x, a, b, c):
#     return a * x**2 + b * x + c

# freq = loptk[:, 0]
# freq_fit = np.linspace(freq.min(), freq.max(), 100)

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [5, 1]}, sharex=True)
# ax2.axhline(0, color='black', linestyle='--', linewidth=1, zorder=1)

# # Loop handles Peak 2 (idx 1, 3) and Peak 3 (idx 2, 4) to elixminate code duplication
# for m_idx, u_idx, color, fmt, name in [(1, 3, 'blue', 'o', 'Peak 2'), (2, 4, 'red', 's', 'Peak 3')]:
#     mean_b, unc_b = loptk[:, m_idx], loptk[:, u_idx]

#     popt, pcov = curve_fit(quad_model, freq, mean_b, sigma=unc_b, absolute_sigma=True)
#     err = np.sqrt(np.diag(pcov))
#     res = mean_b - quad_model(freq, *popt)
#     red_chi2 = np.sum((res / unc_b)**2) / (len(freq) - 3)

#     fit_lbl = (f'{name} Fit: y = ({popt[0]:.2e} Â± {err[0]:.1e})xÂ² + ({popt[1]:.2e} Â± {err[1]:.1e})x + ({popt[2]:.2e} Â± {err[2]:.1e})\n'
#                f'$\chi^2_\\nu$ = {red_chi2:.2f}')

#     ax1.errorbar(freq, mean_b, yerr=unc_b, fmt=fmt, color=color, capsize=4, alpha=0.8, label=f'{name} Data')
#     ax1.plot(freq_fit, quad_model(freq_fit, *popt), color=color, linestyle='-', linewidth=1.5, label=fit_lbl)
#     ax2.errorbar(freq, res, yerr=unc_b, fmt=fmt, color=color, capsize=4, alpha=0.8, zorder=2)

# ax1.set_ylabel('Magnetic Field (T)')
# ax1.set_title('Magnetic Field of Peaks 2 and 3 vs. Frequency')
# ax1.grid(True, linestyle='--', alpha=0.1)
# ax1.legend(loc='best', fontsize=9)

# ax2.set_xlabel('Frequency (kHz)')
# ax2.set_ylabel('Residuals (T)')
# ax2.grid(True, linestyle='--', alpha=0.1)

# plt.tight_layout()
# plt.show()


# In[1075]:


df_9p02MHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile1.csv",1116)
df_7p75MHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile2.csv",960)
df_6p9MHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile3.csv",856.7)
df_5p3MHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile4.csv",659.7)
df_4p3MHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile5.csv",540.8)
df_2p9MHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile6.csv",362)
df_1p75MHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile7.csv",207.6)
df_800kHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile8.csv",94.9)
df_500kHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile9.csv",23.5)
df_300kHz = add_magnetic_field_woff("New Quadratic zeeman data/NewFile10.csv",25.9)


# In[ ]:





# In[1076]:


# New Quadratic Zeeman datasets - Rb85 only
# Absorption dips -> negative Lorentzians (amplitude < 0, y0 ~ 1)
# Peak count: <=800kHz -> 1 peak; 1.75MHz -> 2; 2.9MHz -> 3;
#             4.3MHz -> 4; 5.3MHz -> 5; >=6.9MHz -> 6

nqopt = []  # collects [b_mean, b_unc] per peak, one entry per dataset

# # 1-peak datasets
fit_lorentzians(df_300kHz,  [1.0, -0.6,  3.82,  0.15], 1, nqopt)
fit_lorentzians(df_500kHz,  [1.0, -0.4,  1.755, 0.15], 1, nqopt)
fit_lorentzians(df_800kHz,  [1.0, -0.85, 1.33,  0.10], 1, nqopt)


# In[1077]:


# 2-peak dataset
fit_lorentzians(df_1p75MHz,
    [1.0, -0.15, 2.34, 0.08,
          -0.20, 2.61, 0.08,
          0.20, 2.00, 0.08,
          0.40, 1.87, 0.08,
          0.60, 1.50, 0.08], 5, nqopt)


# In[1078]:


# 3-peak dataset
fit_lorentzians(df_2p9MHz,
    [1.0 , -0.10, 2.34, 0.10,
          -0.25, 2.85, 0.10,
          -0.20, 3.29, 0.10,
           0.50, 1.9, 0.10,
           0.60, 1.5, 0.10,
           0.70, 1.05, 0.10], 6, nqopt)


# In[1079]:


def process_dataset(df, t_min, t_max):
    # Truncate dataset
    df_trunc = df[(df['time'] > t_min) & (df['time'] < t_max)].copy()

    # Calculate boundary coordinates for linear correction
    x0, y0 = df_trunc['time'].iloc[0], df_trunc['CH2norm'].iloc[0]
    x1, y1 = df_trunc['time'].iloc[-1], df_trunc['CH2norm'].iloc[-1]

    # Compute slope and intercept
    m = (y1 - y0) / (x1 - x0)
    b = y0 - m * x0

    # Apply correction and normalize offset
    df_trunc['CH2norm'] = df_trunc['CH2norm'] - (m * df_trunc['time'] + b) + 1

    return df_trunc


# In[1080]:


# # 4-peak dataset,
df_4p3MHz_ = process_dataset(df_4p3MHz,0.7,5.2)
fit_lorentzians(df_4p3MHz_,
    [1.0,  0.50, 2.47, 0.12,
           0.30, 3.17, 0.12,
           0.10, 3.89, 0.12,
           0.00, 4.64, 0.12,
           0.60, 1.70, 0.12,
           0.75, 1.05, 0.12], 6, nqopt)


# In[1081]:


# # 5-peak dataset
df_5p3MHz_ = process_dataset(df_5p3MHz,1,5.2)
fit_lorentzians(df_5p3MHz_,
    [1.0, 0.65, 2.065, 0.12,
          0.45, 2.79,  0.12,
          0.30, 3.55,  0.12,
          0.15, 4.33,  0.12,
          0.00, 5.10,  0.12,
          0.76, 1.3,  0.12], 6, nqopt)


# In[1082]:


# # 6-peak datasets
df_6p9MHz_ = process_dataset(df_6p9MHz,0.9,5.3)

fit_lorentzians(df_6p9MHz_,
    [1.0, 0.78, 1.9, 0.12,
          0.70, 2.63,  0.12,
          0.60, 3.38,  0.12,
          0.45, 4.14,  0.12,
          0.35, 4.87,  0.12,
          0.85, 1.2,  0.12], 6, nqopt)


# In[1083]:


df_7p75MHz_ = process_dataset(df_7p75MHz,0.9,5.3)

fit_lorentzians(df_7p75MHz_,
    [1.0, 0.78, 1.9, 0.12,
          0.70, 2.63,  0.12,
          0.60, 3.38,  0.12,
          0.45, 4.14,  0.12,
          0.35, 4.87,  0.12,
          0.85, 1.2,  0.12], 6, nqopt)


# In[1084]:


df_9p02MHz_ = process_dataset(df_9p02MHz,0.4,5.785)

fit_lorentzians(df_9p02MHz_,
    [1.0, 0.78, 1.7, 0.12,
          0.70, 2.63,  0.12,
          0.60, 3.7,  0.12,
          0.45, 4.8,  0.12,
          0.35, 5.7,  0.12,
          0.85, 0.6,  0.12], 6, nqopt)


# In[1085]:


# Build frequency-tagged array from nqopt
# nqopt entries: one list per dataset, each list = [b1, b2, ..., u1, u2, ...]
# Datasets in order: 300kHz(1), 500kHz(1), 800kHz(1),
#                    1.75MHz(2), 2.9MHz(3), 4.3MHz(4), 5.3MHz(5),
#                    6.9MHz(6), 7.75MHz(6), 9.02MHz(6)

nq_freqs  = [300, 500, 800, 1750, 2900, 4300, 5300, 6900, 7750, 9020]
nq_npeaks = [1,   1,   1,   2,    3,    4,    5,    6,    6,    6   ]

# Flatten into rows of (freq_kHz, b_mean, b_unc)
nq_rows = []
for ds_idx, (freq, n) in enumerate(zip(nq_freqs, nq_npeaks)):
    entry = nqopt[ds_idx]          # [b1,...,bn, u1,...,un]
    b_means = entry[:n]
    b_uncs  = entry[n:]
    for b, u in zip(b_means, b_uncs):
        if not np.isnan(b):
            nq_rows.append([freq, b, u])

nq_arr = np.array(nq_rows)        # shape (N, 3): freq, b_mean, b_unc

# ── Combined plot: existing loptk + new Rb85 data ─────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 9),
                                gridspec_kw={'height_ratios': [5, 1]},
                                sharex=True)
ax2.axhline(0, color='black', linestyle='--', linewidth=1, zorder=1)

freq_all = np.concatenate([loptk[:, 0], nq_arr[:, 0]])
freq_fit = np.linspace(freq_all.min(), freq_all.max(), 200)

colors = {'loptk_p2': 'blue', 'loptk_p3': 'red', 'nq': 'green'}

# Existing peaks 2 & 3 from loptk
for m_idx, u_idx, color, fmt, name in [
        (1, 3, 'blue', 'o', 'Rb87 Peak 2'),
        (2, 4, 'red',  's', 'Rb87 Peak 3')]:
    mean_b = loptk[:, m_idx]
    unc_b  = loptk[:, u_idx]
    popt, pcov = curve_fit(quad_model, loptk[:, 0], mean_b,
                           sigma=unc_b, absolute_sigma=True)
    err = np.sqrt(np.diag(pcov))
    res = mean_b - quad_model(loptk[:, 0], *popt)
    red_chi2 = np.sum((res / unc_b)**2) / (len(loptk[:, 0]) - 3)
    fit_lbl = (f'{name}: ({popt[0]:.2e}±{err[0]:.1e})f² + '
               f'({popt[1]:.2e}±{err[1]:.1e})f + ({popt[2]:.2e}±{err[2]:.1e})\n'
               f'$\\chi^2_\\nu$ = {red_chi2:.2f}')
    ax1.errorbar(loptk[:, 0], mean_b, yerr=unc_b,
                 fmt=fmt, color=color, capsize=4, alpha=0.8, label=f'{name} Data')
    ax1.plot(freq_fit, quad_model(freq_fit, *popt),
             color=color, linestyle='-', linewidth=1.5, label=fit_lbl)
    ax2.errorbar(loptk[:, 0], res, yerr=unc_b,
                 fmt=fmt, color=color, capsize=4, alpha=0.8, zorder=2)

# New Rb85 data - fit quadratic to all points together
nq_freq = nq_arr[:, 0]
nq_b    = nq_arr[:, 1]
nq_u    = nq_arr[:, 2]

popt_nq, pcov_nq = curve_fit(quad_model, nq_freq, nq_b,
                              sigma=nq_u, absolute_sigma=True)
err_nq = np.sqrt(np.diag(pcov_nq))
res_nq = nq_b - quad_model(nq_freq, *popt_nq)
red_chi2_nq = np.sum((res_nq / nq_u)**2) / (len(nq_freq) - 3)

fit_lbl_nq = (f'Rb85 Fit: ({popt_nq[0]:.2e}±{err_nq[0]:.1e})f² + '
              f'({popt_nq[1]:.2e}±{err_nq[1]:.1e})f + ({popt_nq[2]:.2e}±{err_nq[2]:.1e})\n'
              f'$\\chi^2_\\nu$ = {red_chi2_nq:.2f}')

ax1.errorbar(nq_freq, nq_b, yerr=nq_u,
             fmt='^', color='green', capsize=4, alpha=0.8, label='Rb85 Data')
ax1.plot(freq_fit, quad_model(freq_fit, *popt_nq),
         color='green', linestyle='-', linewidth=1.5, label=fit_lbl_nq)
ax2.errorbar(nq_freq, res_nq, yerr=nq_u,
             fmt='^', color='green', capsize=4, alpha=0.8, zorder=2)

ax1.set_ylabel('Magnetic Field (T)')
ax1.set_title('Quadratic Zeeman: Rb87 Peaks 2 & 3 and Rb85 vs. Frequency')
ax1.grid(True, linestyle='--', alpha=0.1)
ax1.legend(loc='best', fontsize=8)

ax2.set_xlabel('Frequency (kHz)')
ax2.set_ylabel('Residuals (T)')
ax2.grid(True, linestyle='--', alpha=0.1)

plt.tight_layout()
plt.show()


# ## Getting images for paper

# In[1086]:


x_data, y_data = df_195["time"], df_195["CH2norm"] 
plt.figure(figsize=(7, 4), dpi=1200)
plt.plot(x_data, y_data, label='Measured Data', alpha=0.7)
# plt.plot(x_data, fit_func(x_data, *popt), 'r-', linewidth=2, label=f'Fitted {num}-Lorentzian')
# plt.scatter(init_centers, init_amps, color='black', marker='x', label='Initial Guesses', zorder=5)
plt.xlabel('Time (s)')
plt.ylabel('CH2norm (Normalized Intensity)')
plt.grid(True, which='both', linestyle='--', alpha=0.1)
plt.legend(loc='best')

plt.tight_layout(pad=0)

plt.savefig('high_res_plot.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()


# In[1087]:


def linear_model(x, m, b):
    return m * x + b

freq = lopt[:, 0]

scale_factor = 1e5
mean_b2, unc_b2 = lopt[:, 2] * scale_factor, lopt[:, 5] * scale_factor
mean_b3, unc_b3 = lopt[:, 3] * scale_factor, lopt[:, 6] * scale_factor

popt2, pcov2 = curve_fit(linear_model, freq, mean_b2, sigma=unc_b2, absolute_sigma=True)
popt3, pcov3 = curve_fit(linear_model, freq, mean_b3, sigma=unc_b3, absolute_sigma=True)

err2 = np.sqrt(np.diag(pcov2))
err3 = np.sqrt(np.diag(pcov3))

res2 = mean_b2 - linear_model(freq, *popt2)
res3 = mean_b3 - linear_model(freq, *popt3)

dof = len(freq) - 2

chi2_2 = np.sum((res2 / unc_b2)**2)
red_chi2_2 = chi2_2 / dof

chi2_3 = np.sum((res3 / unc_b3)**2)
red_chi2_3 = chi2_3 / dof

print(f"--- 85Rb (Peak 2) ---")
print(f"Slope: {popt2[0]:.4e} ± {err2[0]:.4e} (10^-5 T / kHz)")
print(f"Intercept: {popt2[1]:.4e} ± {err2[1]:.4e} (10^-5 T)")
print(f"Reduced Chi-Squared: {red_chi2_2:.4f}\n")

print(f"--- 87Rb (Peak 3) ---")
print(f"Slope: {popt3[0]:.4e} ± {err3[0]:.4e} (10^-5 T / kHz)")
print(f"Intercept: {popt3[1]:.4e} ± {err3[1]:.4e} (10^-5 T)")
print(f"Reduced Chi-Squared: {red_chi2_3:.4f}")

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=1200)

ax1.errorbar(freq, mean_b2, yerr=unc_b2, fmt='o', color='blue', capsize=4, alpha=0.8, label=r'$^{85}$Rb Data')
ax1.errorbar(freq, mean_b3, yerr=unc_b3, fmt='s', color='red', capsize=4, alpha=0.8, label=r'$^{87}$Rb Data')

freq_fit = np.linspace(0, max(freq) * 1.05, 100)

ax1.plot(freq_fit, linear_model(freq_fit, *popt2), 'b-', linewidth=1.5, label=r'$^{85}$Rb Fit')
ax1.plot(freq_fit, linear_model(freq_fit, *popt3), 'r-', linewidth=1.5, label=r'$^{87}$Rb Fit')

ax1.set_xlabel('Frequency (kHz)', fontsize=18)
ax1.set_ylabel(r'Magnetic Field ($10^{-5}$ T)', fontsize=18)
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.grid(True, linestyle='--', alpha=0.1)

ax1.legend(loc='upper left', fontsize=14)
ax1.set_xlim(left=0)

axins = ax1.inset_axes([0.60, 0.10, 0.35, 0.30])
axins.axhline(0, color='black', linestyle='--', linewidth=1, zorder=1)
axins.errorbar(freq, res2, yerr=unc_b2, fmt='o', color='blue', capsize=3, alpha=0.8, zorder=2)
axins.errorbar(freq, res3, yerr=unc_b3, fmt='s', color='red', capsize=3, alpha=0.8, zorder=3)

axins.set_xlabel('Freq (kHz)', fontsize=12)
axins.set_ylabel('Residuals', fontsize=12)
axins.tick_params(axis='both', which='major', labelsize=10)
axins.grid(True, linestyle='--', alpha=0.1)

plt.tight_layout(pad=0)
plt.savefig('zeeman_effect_fits.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
plt.show()


# In[ ]:




