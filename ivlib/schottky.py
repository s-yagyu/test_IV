from typing import Tuple, Union, List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pynverse import inversefunc

def line_fit(xdata: np.ndarray, ydata: np.ndarray) -> tuple:
    """Perform linear fitting using numpy polyfit.

    Args:
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.

    Returns:
        tuple: Slope and intercept of the fitted line.
    """
    slope, intercept = np.polyfit(xdata, ydata, 1)
    return slope, intercept


def abs_line_fit(xdata: np.ndarray, ydata: np.ndarray, params_init: np.ndarray = None) -> tuple:
    """Perform absolute error fitting using minimization.

    Args:
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.
        params_init (np.ndarray, optional): Initial parameters for fitting. Defaults to None.

    Returns:
        tuple: Slope and intercept of the fitted line.
    """
    if params_init is None:
        params_init = np.polyfit(xdata, ydata, 1)

    result = minimize(_absolute_error, params_init, args=(xdata, ydata))
    slope, intercept = result.x

    return slope, intercept


def _m_fit(xdata: np.ndarray, aa: float, bb: float) -> np.ndarray:
    """Linear function for fitting purposes.

    Args:
        xdata (np.ndarray): Independent variable data.
        aa (float): Slope of the line.
        bb (float): Intercept of the line.

    Returns:
        np.ndarray: Computed y values.
    """
    return aa * xdata + bb


def _absolute_error(params: np.ndarray, xdata: np.ndarray, ydata: np.ndarray) -> float:
    """Compute the absolute error between model predictions and actual data.

    Args:
        params (np.ndarray): Parameters (slope, intercept).
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.

    Returns:
        float: Sum of absolute errors.
    """
    y_pred = _m_fit(xdata, *params)
    return np.sum(np.abs(ydata - y_pred))


def select_xarray(xdata: np.ndarray, ydata: np.ndarray, 
                  xmin: float = 0.5, xmax: float = 1) -> tuple:
    """Select a range of data points.

    Args:
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.
        xmin (float, optional): Minimum x value. Defaults to 0.5.
        xmax (float, optional): Maximum x value. Defaults to 1.

    Returns:
        tuple: Filtered xdata and ydata.
    """
    indices = np.where((xdata >= xmin) & (xdata <= xmax))
    xd2 = xdata[indices]
    yd2 = ydata[indices]

    return xd2, yd2

"""
 Schottky analysis method developed by Cheung
(Appl. Phys. Lett.Vol.49, (2), 1986, p.85)

Input:
Ga2O3    APPLIED PHYSICS LETTERS 110, 093503 (2017)
J: Current density[A/cm2]   30A/cm2 at 1.5 V
V: Voltage[V]
A_eff: Dioad area [cm2] 
-> diode diameter: 100micron --> pi*(0.01/2)**2 = 7.85e-5cm2
T: temperetur 300K
AS: Richadson 36 A/cm2K2

Calculate
phi = 1.38 eV
n= 1.1
Rs = 110 ohm

generally
1 < n < 2
0.2 < phi < 1.5

"""

def I2J(I: float, A_eff: float) -> float:
    """Convert current to current density.

    Args:
        I (float): Current in Amperes. 
        A_eff (float): Effective diode area in cm^2.

    Returns:
        float: Current density in A/cm^2.
    """
    J = I / A_eff
    return J


def schottky_func(V: np.ndarray, J: np.ndarray, R: float = 110, phi: float = 1.38, 
                  n: float = 1.1, A_eff: float = 7.85e-5,
                  AS: float = 36, T: float = 300, 
                  IorJ='J', plot: bool = True) -> tuple:
    """Perform Schottky analysis and compute several parameters.

    Args:
        V (np.ndarray): Voltage data.
        J (np.ndarray): Current density data.
        R (float, optional): Resistance in Ohms. Defaults to 110.
        phi (float, optional): Barrier height in electron volts. Defaults to 1.38.
        n (float, optional): Ideality factor. Defaults to 1.1.
        A_eff (float, optional): Effective diode area in cm^2. Defaults to 7.85e-5.
        AS (float, optional): Richardson constant in A/cm^2K^2. Defaults to 36.
        T (float, optional): Temperature in Kelvin. Defaults to 300.
        plot (bool, optional): Whether to plot the results. Defaults to True.

    Returns:
        tuple: Voltage, derivative of voltage with respect to ln(J), H(J), and current density.
    """
    if IorJ == 'I':
        J = J / A_eff
        
    kb_ev = 8.62e-5
    beta = 1 / (kb_ev * T)
    ASTT = AS * T * T

    sh_v = R * A_eff * J + n * phi + (n / beta) * np.log(J / ASTT)
    dV_dlnJ = R * A_eff * J + n / beta
    H_J = V - (n / beta) * np.log(J / ASTT)

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(dV_dlnJ, J, label='dv/dlnJ')
        ax1.set_ylabel('dv/dlnJ')
        ax1.set_xlabel('J')
        ax2.plot(H_J, J, label='H(J)')
        ax2.set_ylabel('H(J)')
        ax2.set_xlabel('J')
        ax1.grid()
        ax2.grid()
        fig.suptitle('Schottky Analysis')
        plt.show()

    return sh_v, dV_dlnJ, H_J, J


def schottky_v(J: np.ndarray, R: float = 10, phi: float = 0.56, 
               n: float = 1.3, A_eff: float = 1, AS: float = 8.6, T: float = 300) -> float:
    """Calculate voltage from Schottky model.

    Args:
        J (float): Current [Amperes/cm2].
        R (float, optional): Resistance in Ohms. Defaults to 10.
        phi (float, optional): Barrier height in electron volts. Defaults to 0.56.
        n (float, optional): Ideality factor. Defaults to 1.3.
        A_eff (float, optional): Effective diode area in cm^2. Defaults to 1.
        AS (float, optional): Richardson constant in A/cm^2K^2. Defaults to 8.6.
        T (float, optional): Temperature in Kelvin. Defaults to 300.

    Returns:
        float: Calculated voltage.
    """
    # J = I / A_eff
    kb_ev = 8.62e-5
    beta = 1 / (kb_ev * T)
    ASTT = AS * T * T

    sh_v = R * A_eff * J + n * phi + (n / beta) * np.log(J / ASTT)
    return sh_v


def inv_schottky_i(V: np.ndarray, R: float = 10e-3, phi: float = 1.3, 
                   n: float = 1.1, A_eff: float = 1, AS: float = 8.6, T: float = 500) -> float:
    """Calculate current from Schottky model using inverse function.

    Args:
        V (float): Voltage in Volts.
        R (float, optional): Resistance in Ohms. Defaults to 10e-3.
        phi (float, optional): Barrier height in electron volts. Defaults to 1.3.
        n (float, optional): Ideality factor. Defaults to 1.1.
        A_eff (float, optional): Effective diode area in cm^2. Defaults to 1.
        AS (float, optional): Richardson constant in A/cm^2K^2. Defaults to 8.6.
        T (float, optional): Temperature in Kelvin. Defaults to 500.

    Returns:
        float: Calculated current in Amperes.
    """
    inv_scv = inversefunc(schottky_v, args=(R, phi, n, A_eff, AS, T))
    return inv_scv(V)


def dJ_dlnJ_fit_plot(V: np.ndarray, J: np.ndarray,
                     A_eff: float = 1, AS: float = 8.6, 
                     T: float = 300, IorJ='J', opt: str = 'abs', plot: bool = True) -> dict:
    """Fit dV/dlnJ vs J and plot results.

        Make graph of $\dfrac{dV_A}{dlnJ} -J$
        slope: $AR_s$, 
        slice: $\dfrac{nk_BT}{q}$
    Args:
        V (np.ndarray): Voltage data.
        I (np.ndarray): Current data.
        A_eff (float, optional): Effective diode area in cm^2. Defaults to 1.
        AS (float, optional): Richardson constant in A/cm^2K^2. Defaults to 8.6.
        T (float, optional): Temperature in Kelvin. Defaults to 300.
        opt (str, optional): Fitting option ('abs' for absolute fit). Defaults to ''.
        plot (bool, optional): Whether to plot the results. Defaults to True.

    Returns:
        dict: Computed and fitted data.
    """
    
    if IorJ=='I':
        J = I / A_eff
    kb_ev = 8.62e-5
    beta = 1 / (kb_ev * T)
    ASTT = AS * T * T
    dV = np.gradient(V)
    dJ = np.gradient(J)
    dV_dJ = dV / dJ
    dV_dlnJ = J * dV_dJ

    try:
        if opt == 'abs':
            fit_p = abs_line_fit(J, dV_dlnJ)
        else:
            fit_p = np.polyfit(J, dV_dlnJ, 1)

        a, b = fit_p[0], fit_p[1]
        fit_line = np.poly1d(fit_p)(J)
        R = a / A_eff
        n = b / (kb_ev * T)
        print(f'R=slope/A_eff: {R:.2e}, n : {n:.3f}')
    except:
        R = np.nan
        n = np.nan
        fit_p = np.nan
        fit_line = np.nan
        print('Error')

    if plot:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(J, dV_dlnJ, 'ro', label='dv/dlnJ')
        try:
            ax1.plot(J, fit_line, 'g-',
                     label=f'Fit\nslice:{fit_p[1]:.2f}\nslope:{fit_p[0]:.2e}')
        except:
            pass
        ax1.set_ylabel('dV/dlnJ')
        ax1.set_xlabel('J')
        ax1.grid()
        ax1.legend(title=f'R=slope/A_eff: {R:.2e}\nn : {n:.3f}')
        fig.suptitle('Schottky Analysis dv/dlnJ-J')
        plt.show()

    return {"dV_dlnJ": dV_dlnJ, "J": J, "R": R, "n": n, "fit": fit_line}


def H_fit_plot(V: np.ndarray, J: np.ndarray, 
               n: float = 1.3, A_eff: float = 1, 
               AS: float = 8.6, T: float = 300, IorJ='J',
               opt: str = 'abs', plot: bool = True) -> dict:
    """Fit H(J) vs J and plot results.

    Args:
        V (np.ndarray): Voltage data.
        J (np.ndarray): Current data [A/cm2].
        n (float, optional): Ideality factor. Defaults to 1.3.
        A_eff (float, optional): Effective diode area in cm^2. Defaults to 1.
        AS (float, optional): Richardson constant in A/cm^2K^2. Defaults to 8.6.
        T (float, optional): Temperature in Kelvin. Defaults to 300.
        opt (str, optional): Fitting option ('abs' for absolute fit). Defaults to ''.
        plot (bool, optional): Whether to plot the results. Defaults to True.

    Returns:
        dict: Computed and fitted data.
    """
    if IorJ=='I':
        J = I / A_eff
        
    kb_ev = 8.62e-5
    beta = 1 / (kb_ev * T)
    ASTT = AS * T * T
    J_p = (n / beta) * np.log(J / ASTT)
    H_J = V - J_p

    try:
        if opt == 'abs':
            fit_p = abs_line_fit(J, H_J)
        else:
            fit_p = np.polyfit(J, H_J, 1)

        a, b = fit_p[0], fit_p[1]
        fit_line = np.poly1d(fit_p)(J)
        R = a / A_eff
        phi = b / n
        print(f'R : {R:.2e}, phi : {phi:.3f}')
    except:
        R = np.nan
        phi = np.nan
        fit_p = np.nan
        print('Error')

    if plot:
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.plot(J, H_J, 'ro', label='H(J)')
        try:
            ax1.plot(J, fit_line, 'g-',
                     label=f'Fit\nslice:{fit_p[1]:.2f}\nslope:{fit_p[0]:.2f}')
        except:
            pass
        ax1.set_ylabel('H(J)')
        ax1.set_xlabel('J')
        ax1.grid()
        ax1.legend(title=f'R : {R:.2e}\nphi : {phi:.3f}')
        fig.suptitle('Schottky Analysis H(J)-J')
        plt.show()

    return {"H_J": H_J, "J": J, "R": R, "phi": phi, "fit": fit_line}


def analysis_all(V: np.ndarray, J: np.ndarray, 
                 A_eff: float = 7.85e-5, AS: float = 36, T: float = 300, 
                 IorJ='J', opt: str ='abs', plot: bool = True) -> tuple:
    """Perform comprehensive Schottky analysis.

    Args:
        V (np.ndarray): Voltage data.
        I (np.ndarray): Current data.
        A_eff (float, optional): Effective diode area in cm^2. Defaults to 7.85e-5.
        AS (float, optional): Richardson constant in A/cm^2K^2. Defaults to 36.
        T (float, optional): Temperature in Kelvin. Defaults to 300.
        plot (bool, optional): Whether to plot the results. Defaults to True.

    Returns:
        tuple: Results of dJ/dlnJ and H(J) analyses.
    """
    dlnJ_dict = dJ_dlnJ_fit_plot(V=V, J=J, A_eff=A_eff, T=T, IorJ=IorJ, opt=opt, plot=False)
    H_dict = H_fit_plot(
        V=V, J=J, n=dlnJ_dict["n"], A_eff=A_eff, AS=AS, T=T, IorJ=IorJ, opt=opt, plot=False)

    if plot:
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(dlnJ_dict['J'], dlnJ_dict['dV_dlnJ'], 'ro', label='dv/dlnJ')
        ax1.plot(dlnJ_dict['J'], dlnJ_dict['fit'], 'g-',
                 label='Fit')
        ax1.set_ylabel('dv/dlnJ')
        ax1.set_xlabel('J')
        ax1.legend(title=f'R : {dlnJ_dict["R"]:.2e}\nn : {dlnJ_dict["n"]:.3f}')
        ax1.grid()

        ax2.plot(H_dict['J'], H_dict['H_J'], 'ro', label='H(J)')
        ax2.plot(H_dict['J'], H_dict['fit'], 'g-',
                 label=f'Fit')
        ax2.set_ylabel('H(J)')
        ax2.set_xlabel('J')
        ax2.legend(title=f'R : {H_dict["R"]:.2e}\nphi : {H_dict["phi"]:.3f}')
        # ax2.set_yscale('log')
        ax2.grid()

        fig.suptitle(f'Schottky Analysis')
        plt.show()

    summary_dict = {'dln_R': dlnJ_dict['R'],
                    'n': dlnJ_dict['n'],
                    'H_R': H_dict['R'],
                    'phi': H_dict['phi']}

    return dlnJ_dict, H_dict, summary_dict


def liner_log_plot2(xdata: np.ndarray, ydata: np.ndarray) -> None:
    """Plot linear and logarithmic data.

    Args:
        xdata (np.ndarray): Independent variable data.
        ydata (np.ndarray): Dependent variable data.
    """
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(xdata, ydata, 'ro-')
    ax1.set_ylabel('J')
    ax1.set_xlabel('V')
    ax2.plot(xdata, ydata, 'ro-')
    ax2.set_ylabel('ln J')
    ax2.set_xlabel('V')
    ax2.set_yscale('log')
    ax1.grid()
    ax2.grid()
    fig.suptitle('Schottky Calculation')
    plt.show()


if __name__ == '__main__':
    pass
