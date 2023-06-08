import matplotlib.pyplot as plt
import scipy.io as sio
import math as m
import numpy as np
import pandas as pd


class ParametersLoadFromMATLAB:
    parameters = sio.loadmat('parameters.mat')
    y_x = parameters['y_x'].flatten()
    x = parameters['x'].flatten()
    F_cr = parameters['F_cr'].flatten()
    gam = parameters['gam'].flatten()
    Mg = parameters['M_g'].flatten()
    Rg = parameters['R_g'].flatten()
    T0 = parameters['T_0'].flatten()
    P0 = parameters.get('p0').flatten()
    L_con = parameters.get('L_con').flatten()
    R_cc = parameters.get('R_cc').flatten()
    R_cr = parameters.get('R_cr').flatten()
    L_cc = parameters.get('L_cc').flatten()
    R1 = parameters.get('R1').flatten()
    R2 = parameters.get('R2').flatten()
    P = parameters.get('P').flatten()
    m_flow = parameters.get('m_flow').flatten()


class CalculationParametersEachSection(ParametersLoadFromMATLAB):
    def __init__(self):
        F_cr = self.F_cr
        gam = self.gam
        F = m.pi * self.y_x ** 2
        equation = lambda L, Fi: ((gam + 1) / 2) ** (1 / (gam - 1)) * (1 - (gam - 1) / (gam + 1) * L ** 2) ** \
                                 (1 / (gam - 1)) * L - F_cr / Fi
        self.eq = equation
        try:
            F_con_i, x_con_i = self.get_F_con_and_X_con(self.R_cc, self.R_cr,
                                                        self.L_con)  # Расчет ведется от критического сечения
            F = np.append(F_con_i, F)  # Площадь сечений начиная с сужающейся части
            self.x = np.append(x_con_i[::-1] * -1, self.x)
            self.F = F
            self.x_con_i = x_con_i[::-1]

            # self.plot_lambda_speed(F) # В этой функции я построил графики. Из которых нашел Лямбда
            lambda_speed = np.array(
                [0.3, 1, 1, 1, 1.8, 2.2, 2.4, 2.5, 2.6, 2.7, 2.7, 2.8, 2.9, 3, 3, 3, 3, 3.1, 3.1, 3.1, 3.1, 3.1, 3.1])

            cr_section_speed_sound = self.cr_section_speed_sound(gam, self.Rg, self.T0)
            speed_x = self.speed_i(cr_section_speed_sound, lambda_speed)

            pressure_x = self.pressure_i(gam, self.P0, lambda_speed)
            temperature_x = self.temperature_i(gam, self.T0, lambda_speed)
            den_0 = self.density_0(self.Rg, self.T0, self.P0)
            density_x = self.density_i(den_0, gam, lambda_speed)
            self.speed_x = speed_x
            self.density_x = density_x

            # self.show_parameters(speed_x, pressure_x, temperature_x, density_x)
            # self.plot_parameters(self.x, speed_x, pressure_x, temperature_x, density_x)

        except OverflowError as err:
            print(err)

    def get_F_con_and_X_con(self, R_cc, R_cr, L_con):
        """X и F_con - считаются от начала сужения камеры сгорания"""
        F_con = []
        x_con = []
        tg_alpha = (R_cc - R_cr) / L_con
        Radius_i = lambda R, L: R - L * tg_alpha

        R_1 = R_cc
        for l in np.arange(L_con / 3, L_con + 1e-5, L_con / 3):
            R_1 = Radius_i(R_1, l)
            F_con.append(np.pi * R_1 ** 2)
            x_con.append(l)
        return np.array(F_con), np.array(x_con)

    def plot_lambda_speed(self, F):
        x = np.arange(0, 4.1, 0.5)
        for Fi in F:
            y = self.eq(x, Fi)
            plt.plot(x, y)
            plt.xlabel(r'$lambda$')
            plt.ylabel(r'$f(lambda)$')
            plt.title(r'Нахождение Лямбда')
            plt.grid(True)
            plt.show()

    def cr_section_speed_sound(self, gam, Rg, T0):
        return m.sqrt(2 * gam * Rg * T0 / (gam + 1))

    def speed_i(self, cr_section_speed_sound, lam_x) -> np.array:
        """Нахождение скорости в каждом сечении"""
        W = lam_x * cr_section_speed_sound
        return W

    def pressure_i(self, gam, P0, lam) -> np.array:
        Pi = P0 * (1 - (gam - 1) / (gam + 1) * lam ** 2) ** (gam / (gam - 1))
        return Pi

    def temperature_i(self, gam, T0, lam) -> np.array:
        Ti = T0 * (1 - (gam - 1) / (gam + 1) * lam ** 2)
        return Ti

    def density_0(self, Rg, T0, P0):
        return Rg * T0 / P0

    def density_i(self, den_0, gam, lam) -> np.array:
        den_i = den_0 * (1 - (gam - 1) / (gam + 1) * lam ** 2) ** (1 / (gam - 1))
        return den_i

    def show_parameters(self, W, P, T, Den):
        print(f'Section Speed - {W}\n'
              f'Section Pressure - {P}\n'
              f'Section Temperature- {T}\n'
              f'Section Density - {Den}')

    def plot_parameters(self, x, *args):
        titles = ['Скорость', 'Давление', 'Температура', 'Плотность']
        for i, param in enumerate(args):
            plt.plot(x, param)
            plt.xlabel(fr'X')
            plt.title(fr'{titles[i]}')
            plt.grid(True)
            plt.show()


# calculate = CalculationParametersEachSection()


class ConvectiveHeatTransferCoefficient(CalculationParametersEachSection):
    """Convective heat transfer coefficient"""
    coolant_properties = {'Cp_cool': 2300, 'L_cool': 0.11, 'D_cool': 780, 'Mu_cool': 5e-4, 'T_cool': 700}

    def __init__(self):
        super().__init__()
        ksi2 = self.L_cc
        ksi_i = self.KSI_i()
        ksi_i = np.append(ksi2, ksi_i)  # Эффективная длинна в каждом счечении

        TPCP = {'Cp': 1810, 'Lg': 0.17, 'Mu_g': 6.3 * 10 ** -5}  # Thermophysical properties combustion products
        # print(len(self.density_x[3:]), len(self.speed_x[3:]), len(ksi_i), sep='\n')
        Re = self.Renolds_number(ksi_i, TPCP.get('Mu_g'))
        Pr = self.Prandtl_number(TPCP.get('Mu_g'), TPCP.get('Cp'), TPCP.get('Lg'))
        alpha_ki = self.ALPHA_ki(TPCP.get('Cp'), Re, Pr)  # Коэффициент конвективной теплоотдачи !!!!!!!!
        self.alpha_ki = alpha_ki
        self.Re = Re
        self.Pr = Pr
        # print(self.density_x)

    def KSI_i(self):
        # Нахождение кси - эффективной длинны
        Ri = np.sqrt(self.F / np.pi)[1:]  # Радиусы без радиуса цилиндра КС
        betta_con = np.arcsin((self.R1 - self.R2) / self.x_con_i[0])
        betta_expanding = np.radians(30) * -1
        betta_a = betta_expanding + np.radians(7)

        R_con = Ri[:3]  # 3ий элемент - критическое сечение
        R_expanding = Ri[3:-1]
        Ra = Ri[-1:-3]
        effective_length = lambda R, betta: self.L_cc * R[1:] ** 0.75 / self.R_cc ** 0.75 + \
                                            4 * R[1:] ** 0.75 * np.sum((R[:-1] - R[1:]) / np.sin(betta))
        ksi = np.append(effective_length(R_con, betta_con), effective_length(R_expanding, betta_expanding))
        return np.append(ksi, effective_length(Ra, betta_a))

    def ALPHA_ki(self, Cp, Re, Pr):
        A = 0.0296
        m = 0.2
        n = 0.6
        alpha = Cp * self.density_x[3:] * self.speed_x[3:] * A * Re ** -m * Pr ** n
        return alpha

    def Renolds_number(self, ksi_i, Mu_g):
        return self.density_x[3:] * self.speed_x[3:] * ksi_i / Mu_g

    def Prandtl_number(self, Mu_g, Cp, Lg):
        return Mu_g * Cp / Lg


class ChoiceNearWallLayerParameters(ParametersLoadFromMATLAB):
    """Choice of near-wall layer parameters"""

    def __init__(self):
        m_flow_cp = self.m_flow * 0.15
        self.m_flow_cp = m_flow_cp

    def get_m_flow(self):
        return self.m_flow_cp


def decorator_d_middle_cc(method_decorated):
    def wrapper(self):
        sigma_walf = 0.001
        hr = 0.002
        return method_decorated(self) + sigma_walf + hr

    return wrapper


def decorator_ribs_cr(function):
    def wrapper(self, D_av_cr):
        return np.floor(function(self, D_av_cr))

    return wrapper


def decorator_D_kn(function):
    def wrapper(self, n):
        betta = np.array([60, 50, 40, 30])
        return function(self, n) / (np.pi * np.cos(np.radians(betta)))

    return wrapper


class DeterminingParametersCoolingPath(ConvectiveHeatTransferCoefficient):
    sigma_walf = 1e-3
    sigma_walf_outer = 4e-3
    sigma_ribs = 2e-3
    t_N_min = 2.5e-3
    t_N_max = 7e-3
    h_ribs = 5e-3
    betta_cr = np.radians(60)

    def __init__(self):
        super().__init__()
        d_average_i = self.D_middle_cc()
        d_average_cr = d_average_i[2]  # Средний диаметр в критике

        n_ribs_cr = self.N_ribs_cr(d_average_cr)  # Число ребер в критике
        d_kn = self.D_kn(n_ribs_cr)  # Диаметр где происходит удвоения числа ребер
        n_ribs_i = self.N_ribs_i(d_average_i)

        t_Ni = np.array(
            [el for el in
             self.step_between_edges_in_current_section(d_average_i, n_ribs_i, d_kn)])  # Шаг между ребер в i сечении
        f_i = self.passage_area_in_cooling_path(t_Ni, n_ribs_i)  # Площадь проходного сечения охладающего тракта !!!!!
        m_cool = ChoiceNearWallLayerParameters().get_m_flow()
        W_cool_i = self.cooling_path_speed(m_cool, f_i, self.coolant_properties.get('D_cool'))
        d_hct = self.hydraulical_diameter_cooling_tractor(self.h_ribs, t_Ni,
                                                          self.sigma_ribs)  # Гидравлический диаметр
        betta_i = self.betta_in_current_section(d_average_i, d_kn)
        parameters_cooling_tracts = pd.DataFrame({'d_average': d_average_i,
                                                  'n_ribs': n_ribs_i,
                                                  'W_cool_i': W_cool_i,
                                                  'betta_i': betta_i,
                                                  't_Ni': t_Ni,
                                                  'fi': f_i,
                                                  'd_hct': d_hct}, index=range(1, 24))
        self.parameters_cooling_tracts = parameters_cooling_tracts
        self.W_cool_i = W_cool_i
        # parameters_cooling_tracts.to_excel('cooling_tract.xlsx') # !!!!!
        # print(parameters_cooling_tracts)

    @decorator_d_middle_cc
    def D_middle_cc(self):
        d = 2 * np.sqrt(self.F / np.pi)
        return d * 2

    @decorator_ribs_cr
    def N_ribs_cr(self, d_average_cr):
        betta = 60  # Угол наклона ребер к образующей
        t_N_min = 2.5e-3  # Минимальный шаг между ребрами
        return np.pi * d_average_cr * np.cos(np.radians(betta)) / t_N_min

    @decorator_D_kn
    def D_kn(self, n_ribs_cr):
        Kn = np.array([1, 2, 4, 8])
        t_N_max = 7e-3
        return Kn * n_ribs_cr * t_N_max

    def N_ribs_i(self, d_average_i):
        t_N_min = 2.5e-3
        betta = 60
        return np.pi * d_average_i * np.cos(np.radians(betta)) / t_N_min

    def step_between_edges_in_current_section(self, d_average_i, n_ribs_i, d_kn):
        betta = (60, 50, 40, 30)
        fun = lambda di, ni, bet: np.pi * di / ni * np.cos(np.radians(bet))

        for di, ni in zip(d_average_i, n_ribs_i):
            if di < d_kn[0]:
                yield fun(di, ni, betta[0])
            elif di < d_kn[1]:
                yield fun(di, ni, betta[1])
            elif di < d_kn[2]:
                yield fun(di, ni, betta[2])
            else:
                yield fun(di, ni, betta[3])

    def betta_in_current_section(self, d_average_i, d_kn):
        betta = (60, 50, 40, 30)
        betta_i = []
        for di in d_average_i:
            if di < d_kn[0]:
                betta_i.append(betta[0])
            elif di < d_kn[1]:
                betta_i.append(betta[1])
            elif di < d_kn[2]:
                betta_i.append(betta[2])
            else:
                betta_i.append(betta[3])
        return np.array(betta_i)

    def passage_area_in_cooling_path(self, t_Ni, n_ribs):
        return t_Ni * self.h_ribs * (1 - self.sigma_ribs / t_Ni) * n_ribs

    def cooling_path_speed(self, m_cool, fi, density_cool):
        return m_cool / fi / density_cool

    def hydraulical_diameter_cooling_tractor(self, h_ribs, t_Ni, sigma_ribs):
        return 2 * h_ribs * (t_Ni - sigma_ribs) / (t_Ni - sigma_ribs + h_ribs)


# Determining the temperature of the fire wall from the gas side in the first approximation
class DeterminingTemperatureFireWall(DeterminingParametersCoolingPath):
    fire_wall_materials = {'Bronze': {'Lw': 340, 'Tw_max': 850}, 'Steel': {'Lw': 26, 'Tw_max': 1200}}
    sigma0 = 5.67 * 1e-8  # Постоянная Стефана-Больцмана
    E_effective = 1 / (0.6 / .5 + 10 / 7 - 1)

    def __init__(self):
        super().__init__()
        Tw_g = 1e3
        T0 = self.T0
        q_ki = self.alpha_ki * (T0 - Tw_g)  # Плотность конвективного теплового потока
        q_k1 = 0.2 * q_ki[0]  # Плотность конвективного теплового потока в 1ом сечении
        q_k1i = np.array([q_k1 for _ in range(3)]).flatten()

        q_r_cc = self.sigma0 * self.E_effective * (
                    T0 ** 4 - Tw_g ** 4)  # Плотность лучистого теплового потока в КС
        q_r_cci = np.array(
            [q_r_cc for _ in range(3)]).flatten()  # Плотность лучистого теплового потока в КС и в сужающщейся части

        Ri = np.sqrt(self.F / np.pi)
        q_ri = .5 * q_r_cc * (Ri[3] / Ri[3:]) ** 2
        parameters_thermal_load_on_fire_wall = pd.DataFrame({'q_кi': np.append(q_k1i, q_ki),
                                                             'q_лi': np.append(q_r_cci, q_ri),
                                                             'q_sum_i': np.append(q_k1i, q_ki) + np.append(q_r_cci,
                                                                                                           q_ri)},
                                                            index=range(1, 24))
        self.parameters_thermal_load_on_fire_wall = parameters_thermal_load_on_fire_wall
        # parameters_thermal_load_on_fire_wall.to_excel('thermal_load_fire_wall.xlsx')
        # print(parameters_thermal_load_on_fire_wall) # !!!!


# Расчет прогрева охладителя
class CoolerWarmUpCalculation(DeterminingTemperatureFireWall):
    def __init__(self):
        super().__init__()
        xi = self.x + self.x[0]*-1
        dx_i = xi[1:] - xi[:-1] # Расстояние до следующего сечения

        di = np.sqrt(self.F / np.pi) * 2
        di[8] = (di[7] + di[9])/2
        dx_s = np.sqrt(dx_i**2 + (di[1:] - di[:-1])**2/4) # Расстояние до следущего сечения
        dS_i = .5 * np.pi * (di[1:] + di[:-1]) * dx_s # Площадь поверхности усеченого конуса

        Cp_cool = self.coolant_properties['Cp_cool']
        m_flow_cool = ChoiceNearWallLayerParameters().get_m_flow()
        q_sum_i = np.array(self.parameters_thermal_load_on_fire_wall.q_sum_i)

        coefficients = np.round(.5 * (q_sum_i[1:] + q_sum_i[:-1]) * dS_i /(Cp_cool * m_flow_cool))
        conclusion = []
        T_cool_i = 300
        for index in range(len(coefficients)-1, -1, -1):
            T_cool_i = T_cool_i + coefficients[index]
            conclusion.append(T_cool_i)
        T_cool_i = np.array(conclusion[::-1])

        cooler_warm_up_calculation = pd.DataFrame({'T_cool_i': T_cool_i}, index=range(1, 23))
        self.cooler_warm_up_calculation = cooler_warm_up_calculation
        cooler_warm_up_calculation.to_excel('cooler warm up calculation.xlsx')
        print(cooler_warm_up_calculation) # !!!!


# Расчет коэффициента конвективной теплоотдачи в охлаждающем тракте
class CoefficientConvectiveHeatTransferCoolingPath(CoolerWarmUpCalculation):
    def __init__(self):
        super().__init__()
        # Nu = .023 * self.Re **.8 * self.Pr **.4
        Ki = self.coolant_properties['L_cool'] ** .6 * (self.coolant_properties['Cp_cool']/self.coolant_properties['Mu_cool'])**.4
        m_flow = ChoiceNearWallLayerParameters().get_m_flow()

        alpha_cool_i = .023 * (m_flow/self.parameters_cooling_tracts['fi'])**.8 * Ki / self.parameters_cooling_tracts['d_hct']
        coefficient_convective = pd.DataFrame({'alpha_cool_i': alpha_cool_i,
                                               'pW': m_flow/self.F,
                                               'W_cool_i': self.W_cool_i})
        self.coefficient_convective = coefficient_convective
        # print(coefficient_convective)
        # coefficient_convective.to_excel('coefficient_convective.xlsx')


# Расчет коэффициента эффективности оребрения
class CalculateCoefficientFinEfficiency(CoefficientConvectiveHeatTransferCoolingPath):
    def __init__(self):
        super().__init__()
        betta_i = np.radians(self.parameters_cooling_tracts['betta_i'])
        t_Ni = self.parameters_cooling_tracts.t_Ni
        Bi = self.coefficient_convective.alpha_cool_i * self.sigma_ribs / self.fire_wall_materials['Bronze']['Lw']
        Wi = self.h_ribs / self.sigma_ribs * np.sqrt(2*Bi)
        Ei = np.tanh(Wi) / Wi

        nu_ribs_i = 1 + 1/np.cos(betta_i) * (2 * self.h_ribs * Ei - self.sigma_ribs)/t_Ni
        coefficient_fin_efficiency = pd.DataFrame({'nu_ribs': np.round(nu_ribs_i, 3)})
        self.coefficient_fin_efficiency = np.abs(coefficient_fin_efficiency)
        # coefficient_fin_efficiency.to_excel('coefficient_fin_efficiency.xlsx')


# Расчет коэффициента эффективности оребрения
class FinningEfficientFactorCalculation(CalculateCoefficientFinEfficiency):
    def __init__(self):
        super().__init__()
        Tw_cool = self.parameters_thermal_load_on_fire_wall['q_sum_i'] / self.coefficient_convective['alpha_cool_i'] /\
             self.coefficient_fin_efficiency.nu_ribs + self.cooler_warm_up_calculation.T_cool_i

        Tw_cool_i = np.round(Tw_cool[:-1], 3)
        Tw_gas = self.parameters_thermal_load_on_fire_wall.q_sum_i / self.fire_wall_materials['Bronze']['Lw'] * \
                   self.sigma_walf + Tw_cool_i
        Tw_gas_i = np.round(Tw_gas[:-1], 3)
        finning_effeicient_facor = pd.DataFrame({'Tw_cool_i': Tw_cool_i,
                                                 'Tw_gas_i': Tw_gas_i})
        finning_effeicient_facor.to_excel('finning_effeicient_facor.xlsx')
        print(finning_effeicient_facor) # !!!!!


if __name__ == '__main__':
    test = FinningEfficientFactorCalculation()

