import json
import numpy as np
import os
import tqdm
from scipy.interpolate import interp1d
import srm_solve
from scipy.misc import derivative
from scipy.integrate import solve_ivp

class SRM_Solver:

    def rate_a(pc,data):
        pc = pc / 1000000

        for key in data:
            # 检查pc值是否在JSON文件中的pc范围内
            if float(key.split('-')[0]) <= pc <= float(key.split('-')[1]):
                return data[key]['a']
        print(f"Info: 找不到相应于 pc 值 {pc} 的条目，使用默认值")  
        return 0  # 默认返回值
    
    def rate_n(pc,data):
        pc = pc / 1000000

        for key in data:
            if float(key.split('-')[0]) <= pc <= float(key.split('-')[1]):
                return data[key]['n']
        print(f"Info: 找不到相应于 pc 值 {pc} 的条目，使用默认值")  
        return 0  # 默认返回值
    
    def Gamma(gamma):

        rest = ((gamma)**0.5)*((2/(gamma+1))**((gamma+1)/(2*(gamma-1))))

        return rest

    
    def init_val(self,P0,e_max):
        # 初始化参数
        P0 = P0

        e_span = (0, e_max) 
        P_atm = 1.01325e-1  # 标准大气压

        return P0,e_span
    


    def calculate_pe(p_c, gamma, Ma):
        #Ma = 0.9999
        Rest = p_c / (1 + ((gamma - 1) / 2 * Ma**2)) ** ((gamma - 1)/gamma)

        return Rest

    
    def solve_ode(self,ode_function,P0,e_max):

        # 定义微分方程
        pc0,e_span = self.init_val(P0,e_max)


        solution = solve_ivp(ode_function, e_span, [pc0], method='RK45', t_eval=np.linspace(e_span[0], e_span[1], 100))
        
        return solution