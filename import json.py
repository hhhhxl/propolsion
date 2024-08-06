import json
import numpy as np
import os
import tqdm
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import srm_solve as srm

def set_directory():
    SRM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_ROOT_DIR = f"{SRM_ROOT_DIR}/settings"
    burning_rate_dir = f"{SETTINGS_ROOT_DIR}/burning_rate"
    grain_dir = f"{SETTINGS_ROOT_DIR}/grain"
    A_e_dir = f"{SETTINGS_ROOT_DIR}/A_e"
    nozzle_dir = f"{SETTINGS_ROOT_DIR}/nozzle"
    return SRM_ROOT_DIR, SETTINGS_ROOT_DIR, burning_rate_dir, grain_dir, A_e_dir, nozzle_dir

def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_and_extract_data(file_path):
    data1 = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            try:
                e_val = float(parts[0])
                Ab_val = float(parts[1])
                data1.append((e_val, Ab_val))
            except ValueError:
                continue
    data1 = np.array(data1)
    return data1

def init_computation():
    SRM_ROOT_DIR, SETTINGS_ROOT_DIR, burning_rate_dir, grain_dir, A_e_dir, nozzle_dir= set_directory()
    data_grain = read_json(f"{grain_dir}/grain_test.json")
    data_burning_rate = read_json(f"{burning_rate_dir}/burning_rate.json")
    data_A_e = read_and_extract_data(f"{A_e_dir}/test.txt")
    data_nozzle = read_json(f"{nozzle_dir}/nozzle_test.json")
    return data_grain, data_burning_rate, data_A_e, data_nozzle

def load_Ab():
    SRM_ROOT_DIR, SETTINGS_ROOT_DIR, burning_rate_dir, grain_dir, A_e_dir, nozzle_dir= set_directory()
    data1 = []
    with open(f"{A_e_dir}/test.txt", 'r') as file:
        for line in file:
            parts = line.split()
            try:
                e_val = float(parts[0])
                Ab_val = float(parts[1])
                data1.append((e_val, Ab_val))
            except ValueError:
                continue
    data1 = np.array(data1)
    e_data = data1[:, 0]
    Vc_data = data1[:, 1]
    first_column_diff = np.diff(e_data)
    e_average = np.mean(first_column_diff)
    Vc_data_trans = Vc_data * 0.0000164
    Vc_data_diff = np.diff(Vc_data)
    Ab_data= -1*Vc_data_diff
    np.append(Ab_data, [0])
    Ab_data = (Ab_data / e_average)*0.0006452
    e_average_trans = e_average * 0.0254
    return Ab_data, e_data, e_average_trans, Vc_data_trans

def compute_Pt(P0):
    i = 0
    data_grain, data_burning_rate, data_A_e, data_nozzle = init_computation()
    Ab_data, e_data, e_step, Vc_data = load_Ab()
    Vc_data = 1e-6*data_grain["Volum"] - Vc_data
    rho_b = 1000*data_grain["density"]
    c_star = data_grain["C*"]
    gamma = data_grain["gamma"]
    Gamma = srm.SRM_Solver.Gamma(gamma)
    At = 1e-4*data_nozzle["At"]
    Pc = np.array([])
    Pc = np.append(Pc,P0)
    t = np.array([])
    t = np.append(t,[0])
    dP = 0
    for i in tqdm(range(len(Ab_data)), desc="Computing"):
        rate_a = srm.SRM_Solver.rate_a(Pc[i], data_burning_rate)
        rate_n = srm.SRM_Solver.rate_n(Pc[i], data_burning_rate)
        Kb = Ab_data[i] / At
        C1 = (rho_b*c_star*rate_a)**(1/(1-rate_n))
        C2 = rate_a / (c_star*(Gamma**2)*At)
        Pceq = C1*(Kb**(1/(1-rate_n)))
        pp1 = Pceq**(1-rate_n)
        pp2 = C2*Vc_data[i]*(dP/e_step)
        pp3 = 1/(1-rate_n)
        Pc_unit = ((pp1 - pp2)**pp3 + Pc[i]) / 2
        Pc = np.append(Pc, Pc_unit)
        P_cold = Pc[i]
        dp = Pc_unit - P_cold
        t_unit = e_step/(rate_a*(Pc_unit)**(rate_n))
        tt = t[i] + t_unit
        t = np.append(t, tt)
        i = i+1
        if i == (len(Ab_data)):
            break
    return Pc, t

def calculate_pe(p_c):
    gamma = 1.4
    Ma = 0.9999
    return p_c / (1 + (gamma - 1) / 2 * Ma**2) ** (gamma / (gamma - 1))

def calculate_C_F(p_c, p_e):
    gamma = 1.4
    p_a = 101325
    term1 = (2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))
    term2 = np.sqrt(gamma) * np.sqrt(
        (2 * gamma / (gamma - 1)) * (1 - (p_e / p_c) ** ((gamma - 1) / gamma))
    )
    term3 = (p_e / p_c) - (p_a / p_c)
    return term1 * term2 + term3

def calculate_F(p_c, A_t, eta):
    p_e = calculate_pe(p_c)
    C_F = calculate_C_F(p_c, p_e)
    return C_F * eta * A_t * p_c

if __name__ == '__main__':
    Pc, t = compute_Pt(1000000)
    eta = 0.95
    A_t = 23.76e-2
    F_values = [calculate_F(p_c, A_t, eta) for p_c in Pc]
    plt.figure(figsize=(12, 6))
    plt.plot(t, F_values, marker='o', linestyle='-', color='b')
    plt.xlabel('时间 t (秒)')
    plt.ylabel('推力 F (牛顿)')
    plt.title('推力 vs 时间')
    plt.grid(True)
    plt.show()
