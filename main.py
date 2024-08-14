import json
import numpy as np
import os
import tqdm
from scipy.interpolate import interp1d
import srm_solve as srm
from scipy.misc import derivative
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys



def set_directory():
    """
    这个函数的作用是设置目录。

    参数：
    无。

    返回值：
    元组，包含六个字符串：SRM_ROOT_DIR、SETTINGS_ROOT_DIR、burning_rate_dir、grain_dir、A_e_dir、nozzle_dir。

    用法示例：
    >>> SRM_ROOT_DIR, SETTINGS_ROOT_DIR, burning_rate_dir, grain_dir, A_e_dir, nozzle_dir = set_directory()
    """
    # SRM_ROOT_DIR = os.getcwd()
    SRM_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SETTINGS_ROOT_DIR = f"{SRM_ROOT_DIR}/settings"
    burning_rate_dir = f"{SETTINGS_ROOT_DIR}/burning_rate"
    grain_dir = f"{SETTINGS_ROOT_DIR}/grain"
    A_e_dir = f"{SETTINGS_ROOT_DIR}/A_e"
    nozzle_dir = f"{SETTINGS_ROOT_DIR}/nozzle"
    return SRM_ROOT_DIR, SETTINGS_ROOT_DIR, burning_rate_dir, grain_dir, A_e_dir, nozzle_dir

def read_json(file_path):
    """
    从文件中读取 JSON 数据

    参数：
    file_path (str)：JSON 文件的路径

    返回值：
    dict：从文件中读取的 JSON 数据转换成的字典

    使用示例：
    >>> data = read_json('file.json')
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def read_and_extract_data(file_path):
    """
    读取指定文件路径下的文件，提取每一行的前两个数字，将它们转换为浮点数，
    并作为元组添加到列表中。列表被转换为 NumPy 数组后，提取第一列和
    第二列数据。对第二列数据使用三次样条插值，并返回插值函数。

    参数：
    file_path (str)：数据文件的路径

    返回值：
    Ab_interpolated (interp1d 对象)：对于提取出的 `Ab` 数据进行三次样条插值得到的函数对象

    异常处理：
    如果文件无法打开或者内容格式不正确，函数会打印错误信息并终止运行。
    """
    data1 = []
    # 读取文件并过滤有效数据
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            # 仅保留数字部分
            try:
                e_val = float(parts[0])
                Ab_val = float(parts[1])
                data1.append((e_val, Ab_val))
            except ValueError:
                continue

    # 将数据转换为 NumPy 数组
    data1 = np.array(data1)
    # 提取e和Ab
    e_data = data1[:, 0]
    Ab_data = data1[:, 1]

    # Ab_interpolated = interp1d(e_data, Ab_data, kind='cubic', fill_value='extrapolate')
    return data1

def init_computation():
    """
    初始化计算过程，包括读取 Grain、燃烧速率、A_e 和喷管数据，并读取和提取设置文件中的变量

    返回值:
    grain_data (dict): 从 grain_dir 读取的 JSON 格式的数据
    burning_rate_data (dict): 从 burning_rate_dir 读取的 JSON 格式的燃烧速率数据
    A_e_data (dict): 从 A_e_dir 读取的 JSON 格式的 A_e 数据
    nozzle_data (dict): 从 nozzle_dir 读取的 JSON 格式的喷管数据
    """

    SRM_ROOT_DIR, SETTINGS_ROOT_DIR, burning_rate_dir, grain_dir, A_e_dir, nozzle_dir= set_directory()

    data_grain = read_json(f"{grain_dir}/grain_test.json")
    data_burning_rate = read_json(f"{burning_rate_dir}/burning_rate.json")
    data_A_e = read_and_extract_data(f"{A_e_dir}/test.txt")
    data_nozzle = read_json(f"{nozzle_dir}/nozzle_test.json")

    return data_grain, data_burning_rate, data_A_e, data_nozzle

def load_Ab():

    """
    加载并处理固体火箭发动机性能数据。

    该函数从指定的目录中读取燃烧室面积比（Ab）与排气孔喉部比（e）的数据，
    并进行必要的数据处理，包括过滤无效数据、计算平均肉厚以及计算Ab的变化量。
    最后，返回处理后的Ab数据和平均肉厚。
    """


    SRM_ROOT_DIR, SETTINGS_ROOT_DIR, burning_rate_dir, grain_dir, A_e_dir, nozzle_dir= set_directory()

    #data_A_e = read_json(f"{A_e_dir}/test.txt")
    data1 = []
    # 读取文件并过滤有效数据
    with open(f"{A_e_dir}/test.grt", 'r') as file:
        for line in file:
            parts = line.split()
            # 仅保留数字部分
            try:
                e_val = float(parts[0])
                Ab_val = float(parts[1])
                data1.append((e_val, Ab_val))
            except ValueError:
                continue

    # 将数据转换为 NumPy 数组
    data1 = np.array(data1)

    # 提取e和Ab
    # e_data = 0.0254*data1[:, 0]
    e_data = data1[:, 0]
    # Vc_data = 16.387064*1e-6*data1[:, 1]
    Vc_data = data1[:, 1]


    # 计算平均肉厚（求解步长）
    first_column_diff = np.diff(e_data)
    e_average = np.mean(first_column_diff)

    # 将Vc_data转换为m^3
    Vc_data_trans = Vc_data * 0.0000164

    Vc_data_diff = np.diff(Vc_data)

    Ab_data= -1*Vc_data_diff
    # Ab_data.insert(0, 0)  # 在列表开头插入0
    np.append(Ab_data, [0])

    Ab_data = (Ab_data / e_average)*0.0006452

    e_average_trans = e_average * 0.0254

    return Ab_data, e_data, e_average_trans, Vc_data_trans

def compute_Pt(P0):
    #循环迭代计算
    i = 0

    # 初始化计算过程，包括读取 Grain、燃烧速率、A_e 和喷管数据，并读取和提取设置文件中的变量
    data_grain, data_burning_rate, data_A_e, data_nozzle = init_computation()
    Ab_data, e_data, e_step, Vc_data = load_Ab()



    Vc_data = 1e-6*data_grain["Volum"] - Vc_data


    # 获取计算参数
    rho_b = 1000*data_grain["density"]

    P_atm = data_nozzle["P_atm"]

    c_star = data_grain["C*"]
    gamma = data_grain["gamma"]

    Gamma = srm.SRM_Solver.Gamma(gamma)
    At = 1e-4*data_nozzle["At"]


    K0 = Ab_data[0] / At

    #初始化numpy列表
    Pc = np.array([])
    #Pc = np.append(Pc,[0])

    t = np.array([])


    Pc = np.append(Pc,P0)

    rate_a = srm.SRM_Solver.rate_a(P0, data_burning_rate)
    rate_n = srm.SRM_Solver.rate_n(P0, data_burning_rate)

    #计算点火段长度
    t_ig = np.append(t,compute_init_ig(P0, Vc_data[0], Gamma, c_star, At, rho_b, K0, rate_a, rate_n))
    t = np.append(t,[t_ig])


    #设置初值
    dP = 0


    #进入循环
    #while True:
    for i in tqdm(range(len(Ab_data)), desc="Computing"):

        #获取a，n
        rate_a = srm.SRM_Solver.rate_a(Pc[i], data_burning_rate)
        rate_n = srm.SRM_Solver.rate_n(Pc[i], data_burning_rate)


        #获取Kb
        Kb = Ab_data[i] / At

        #定义计算算子C1
        C1 = (rho_b*c_star*rate_a)**(1/(1-rate_n))

        #定义计算算子C2
        C2 = rate_a / (c_star*(Gamma**2)*At)

        #计算平衡压强
        Pceq = C1*(Kb**(1/(1-rate_n)))

        #计算实际压强
        pp1 = Pceq**(1-rate_n)
        pp2 = C2*Vc_data[i]*(dP/e_step)

        pp3 = 1/(1-rate_n)

        # pp1 = round(pp1, 4)

        # pp2 = round(pp2, 4)
        # pp3 = round(pp3, 4)

        Pc_unit = ((pp1 - pp2)**pp3 + Pc[i]) / 2

        #将Pc_unit加入列表
        Pc = np.append(Pc, Pc_unit)

        #计算P_code
        P_cold = Pc[i]

        #计算deltaP
        dp = Pc_unit - P_cold

        #计算时间并加入列表
        t_unit = e_step / (rate_a*(Pc_unit)**(rate_n))

        tt = t[i] + t_unit

        t = np.append(t, tt)

        #循环变量自增
        i = i+1


        #循环结束判断
        if i == (len(Ab_data)):

            # Vc = 1e-6*data_grain["Volum"]
            # t_post = compute_post(Vc, Gamma, Pceq, c_star, At,Pc)

            # t = np.append(t, t_post)
            # Pc = np.append(Pc, 0)
            break

        # if (i != 0) and (Pc_unit < P_atm):
        #     break


    t = np.insert(t, 0, 0)
    Pc = np.insert(Pc, 0, 0)

    # Vc = 1e-6*data_grain["Volum"]
    # t_post = compute_post(Vc, Gamma, Pceq, c_star, At,Pc_unit)

    # t = np.append(t, t_post)
    # Pc = np.append(Pc, 0)

    return Pc, t

def calculate_C_F(p_c):

    data_grain, data_burning_rate, data_A_e, data_nozzle = init_computation()

    p_a = 101325
    Ma = 0.999

    Ae = 1e-4*data_nozzle["Ae"]
    At = 1e-4*data_nozzle["At"]
    gamma = data_grain["gamma"]

    Gamma = srm.SRM_Solver.Gamma(gamma)

    p_e = srm.SRM_Solver.calculate_pe(p_c, gamma, Ma)

    term1 = 2*gamma/(gamma-1)*(1 - (p_e / p_c) ** ((gamma - 1) / gamma))
    term1 = Gamma*np.sqrt(term1)

    term2 = (Ae/At)*((p_e / p_c) - (p_a / p_c))



    return term1 + term2

def calculate_F(p_c):

    data_grain, data_burning_rate, data_A_e, data_nozzle = init_computation()
    C_F = calculate_C_F(p_c)

    At = 1e-4*data_nozzle["At"]

    eta = data_nozzle["eta"]


    F = C_F * eta * At * p_c

    return F

def plot_chart_P(t, Pc):
    """
    绘制 Pc 随时间变化的图表。

    :param t: 时间序列数据
    :param Pc: 对应的 Pc 值序列
    """

    # t = np.insert(t, 0, t_ig)
    # t = np.insert(t, 0, t_ig)

    plt.figure(num='Pc vs Time',figsize=(8, 6))

    # 设置图表的标题和坐标轴标签
    plt.title('Pc-----Time')
    plt.xlabel('Time (s)')  # 横坐标单位
    plt.ylabel('Pc (Pa)')  # 纵坐标单位

    # 绘制数据点，并使用圆点形状，同时减小圆点大小
    plt.plot(t, Pc, 'o-', markersize=3, label='Data Points')  # 'o-' 表示圆形标记和线连接

    # 显示网格
    plt.grid(True)

    # 添加图例
    plt.legend()

    # 显示图表
    #plt.show()

def plot_chart_F(t, Pc):
    """
    绘制 Pc 随时间变化的图表。

    :param t: 时间序列数据
    :param Pc: 对应的 Pc 值序列
    """


    plt.figure(num='F vs Time',figsize=(8, 6))

    # 设置图表的标题和坐标轴标签
    plt.title('F-----Time')
    plt.xlabel('Time (s)')  # 横坐标单位
    plt.ylabel('F (N)')  # 纵坐标单位

    # 绘制数据点，并使用圆点形状，同时减小圆点大小
    plt.plot(t, Pc, 'o-', markersize=3, label='Data Points')  # 'o-' 表示圆形标记和线连接

    # 显示网格
    plt.grid(True)

    # 添加图例
    plt.legend()

    # 显示图表
    plt.show()

def write_to_csv_with_pandas(t, Pc, filename):
    """
    使用 Pandas 将时间序列 t 和对应的 Pc 值写入 CSV 文件。

    :param t: 时间序列数据
    :param Pc: 对应的 Pc 值序列
    :param filename: 输出 CSV 文件的名称，默认为 'pt_data.csv'
    """
    # 创建 DataFrame
    data = {
        'Time (s)': t,
        'Pc (Pa)': Pc
    }
    df = pd.DataFrame(data)

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(filename, index=False)

def compute_init_ig(P0, Vc, Gamma, c_star, At, rho_p, K0, rate_a, rate_n):
    #计算点火段

    C_eq = (rho_p*c_star*rate_a)**(1/(1-rate_n))
    Pceq = C_eq*(K0**(1/(1-rate_n)))


    C1 = (1/(1-rate_n))*(Vc/((Gamma**2)*c_star*At))

    cc2 = rho_p*c_star*K0-P0**(1-rate_n)
    ccc2 = rho_p*c_star*K0-Pceq**(1-rate_n)
    C2 = np.log(cc2 / ccc2)

    t = C1*C2

    if C2 != C2:
        print("\n*******************************************************************")
        print("点火压设置错误!!!")
        print("*******************************************************************\n")
        sys.exit(0)

    return t

def Compute_M(rho_b, c_star, Kn, Ctp, phi_a, rate_a, rate_n):
    #计算M
    C1 = (rho_b*c_star*rate_a*Kn*phi_a)/Ctp
    # M = C1**(1/(1.000835-rate_n))

    return C1

def compute_Pt_s(P0):
    #循环迭代计算
    i = 0

    # 初始化计算过程，包括读取 Grain、燃烧速率、A_e 和喷管数据，并读取和提取设置文件中的变量
    data_grain, data_burning_rate, data_A_e, data_nozzle = init_computation()
    Ab_data, e_data, e_step, Vc_data = load_Ab()



    Vc_data = 1e-6*data_grain["Volum"] - Vc_data


    # 获取计算参数
    rho_b = 1000*data_grain["density"]

    P_atm = data_nozzle["P_atm"]

    c_star = data_grain["C*"]
    gamma = data_grain["gamma"]

    rho_s = 1000*data_grain["rho_s"]
    dc0 = data_grain["dc0"]

    ep = data_grain["ep"]
    phi_a =data_grain["phi_a"]


    Gamma = srm.SRM_Solver.Gamma(gamma)
    At = 1e-4*data_nozzle["At"]


    K0 = Ab_data[0] / At

    #初始化numpy列表
    Pc = np.array([])
    #Pc = np.append(Pc,[0])

    t = np.array([])


    Pc = np.append(Pc,P0)

    rate_a = srm.SRM_Solver.rate_a(P0, data_burning_rate)
    rate_n = srm.SRM_Solver.rate_n(P0, data_burning_rate)

    #计算点火段长度
    t_ig = np.append(t,compute_init_ig(P0, Vc_data[0], Gamma, c_star, At, rho_b, K0, rate_a, rate_n))
    t = np.append(t,[t_ig])


    #设置初值
    dP = 0


    #进入循环
    #while True:
    for i in tqdm(range(len(Ab_data)), desc="Computing"):

        #获取a，n
        rate_a = srm.SRM_Solver.rate_a(Pc[i], data_burning_rate)
        rate_n = srm.SRM_Solver.rate_n(Pc[i], data_burning_rate)

        rate = rate_a*((Pc[i])**(rate_n))


        #获取Kb
        Kb = Ab_data[i] / At

        #获取Ctp
        Ctp = srm.SRM_Solver.Get_Ctp(ep, rho_s, Vc_data[i], At, dc0, rate)

        #计算M
        M = Compute_M(rho_b, c_star, Kb, Ctp, phi_a, rate_a, rate_n)



        #计算平衡压强
        Pceq = ((1-ep)*M)**(1/(1.000835-rate_n))

        #定义计算算子C2
        C2 = rate_a / (c_star*(Gamma**2)*At)


        #计算实际压强
        pp1 = Pceq**(1-rate_n)
        pp2 = C2*Vc_data[i]*(dP/e_step)

        pp3 = 1/(1-rate_n)

        # pp1 = round(pp1, 4)

        # pp2 = round(pp2, 4)
        # pp3 = round(pp3, 4)

        Pc_unit = ((pp1 - pp2)**pp3 + Pc[i]) / 2

        #将Pc_unit加入列表
        Pc = np.append(Pc, Pc_unit)

        #计算P_code
        P_cold = Pc[i]

        #计算deltaP
        dp = Pc_unit - P_cold

        #计算时间并加入列表
        t_unit = e_step / (rate_a*(Pc_unit)**(rate_n))

        tt = t[i] + t_unit

        t = np.append(t, tt)

        #循环变量自增
        i = i+1


        #循环结束判断
        if i == (len(Ab_data)):

            # Vc = 1e-6*data_grain["Volum"]
            # t_post = compute_post(Vc, Gamma, Pceq, c_star, At,Pc)

            # t = np.append(t, t_post)
            # Pc = np.append(Pc, 0)
            break


    t = np.insert(t, 0, 0)
    Pc = np.insert(Pc, 0, 0)

    return Pc, t

def compute_post(Vc, Gamma, Pceq, c_star, At, Pc):
    #后效段求解




    ttt1 = Vc/(Gamma*c_star*At)
    ttt2 = np.log(Pceq/Pc)
    t = ttt1*ttt2


    return t

if __name__ == '__main__':

    # Pc, t= compute_Pt(1000000)
    Pc, t = compute_Pt_s(1000000)

    filename ='Pt_data.csv'

    # write_to_csv_with_pandas(t, Pc, filename)

    F = calculate_F(Pc)

    plot_chart_P(t, Pc)

    plot_chart_F(t, F)


