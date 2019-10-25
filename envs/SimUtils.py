import os
import pandas as pd
import cantera as ct
import numpy as np
from numba import jit

def premix(phi=0.4, fuel={'CH4': 1}, ox={'N2': 0.79, 'O2': 0.21}, mech='gri30.xml', P=25*101325, T_fuel=300, T_ox=650, M_total=1):

    """Function that premixes air and fuel at a prescribed equivalence ratio, phi

    Keyword Arguments:
        phi {float} -- [Equivalence ratio at which fuel and oxidizer are mixed] (default: {0.4})
        fuel {dict} -- [Dictionary containing the fuel composition] (default: {{'CH4':1}})
        ox {dict} -- [Dictionary containing air composition] (default: {{'N2':0.79, 'O2':0.21}})
        mech {str} -- [String that contains the mechanism file. Don't change unless absolutely necessary!] (default: {'gri30.xml'})
        P {float} -- [Pressure in Pa] (default: {25*101325})
        T_fuel {int} -- [Preheat temperature of the fuel in Kelvin] (default: {300})
        T_ox {int} -- [Preheat temperature of the oxidizer in Kelvin. Normally this is the temperature of the air coming in from the compressor] (default: {650})
        M_total {int} -- [Total mass; arbitrarily set to 1] (default: {1})

    Returns:
        [type] -- [description]
    """
    airGas = ct.Solution(mech)
    airGas.TPX = [T_ox, P, ox]
    fuelGas = ct.Solution(mech)
    fuelGas.TPX = T_fuel, P, fuel

    # Temporary ThermoPhase object to get mass flow rates:
    temp = ct.Solution('gri30.xml')
    temp.set_equivalence_ratio(phi, fuel, ox)

    # Find the fuel and oxidizer mass flow rates for the given equivalence ratio:
    # here, temp[fuel.keys()].Y returns an array containing the mass fraction of all fuel species:
    mdot_fuel = M_total * sum(temp[fuel.keys()].Y)
    mdot_ox = M_total * sum(temp[ox.keys()].Y)

    fuel = ct.Quantity(fuelGas)
    fuel.mass = mdot_fuel

    air = ct.Quantity(airGas)
    air.mass = mdot_ox

    fuel.constant = air.constant = 'HP'  # keep enthalpy and pressure constant

    fuel_air_mixture = fuel + air  # mix at constant HP

    # Output mixer gas:
    return fuel_air_mixture.phase

@jit(nopython=True, fastmath=True, cache=True)
def correctNOx(X_i, X_H2O, X_O2):
    dry_i = X_i/(1 - X_H2O)
    dry_O2 = X_O2/(1 - X_H2O)
    corrected = dry_i*(20.9 - 15)/(20.9 - dry_O2*100)
    corrected_ppm = corrected*1e6
    return corrected_ppm     

def getStateAtTime(flame, tList, tRequired, mech='gri30.xml'):
    '''A function that gets the state at a desired point in time of a flame simulation performed using runFlame. 
        Takes in a flame object, its associated time series, and the desired point in time (in seconds).
        Returns a new Cantera gas object with the desired state, and the corresponding index in the flame at which the point was found. 

        Example usage: gas, t_ind = getStateAtTime(flame, time, t_req)'''
    mainBurnerDF = flame
    columnNames = mainBurnerDF.columns.values
    vel_final = mainBurnerDF['u'].iloc[-1]
    moleFracs = mainBurnerDF.columns[mainBurnerDF.columns.str.contains('X_')]
    assert (tRequired > 0)
    newGas = ct.Solution(mech)

    if (tRequired >= max(tList)):
        tau_vit = tRequired - max(tList)
        newGas.TPX = flame['T'].iloc[-1], flame['P'].iloc[-1], flame[moleFracs].iloc[-1]
        tau_vit_start = tList[-1]
    else:
        dt_list = abs(tList - tRequired)
        # Find index at which required time is closest to an entry in tList:     
        minIndex = next(ind for ind, val in enumerate(dt_list) if abs(val - min(dt_list)) < 1e-6)
        flameTime_closestTreq = tList[minIndex] 
        if ((flameTime_closestTreq - tRequired) > 1e-6):
            newGas.TPX = flame['T'].iloc[minIndex-1], flame['P'].iloc[minIndex-1], flame[moleFracs].iloc[minIndex-1]
            assert((newGas['NO'].X - flame['X_NO'].iloc[minIndex-1] <= 1e-6))    
            tau_vit = tRequired - tList[minIndex - 1]
            tau_vit_start = tList[minIndex - 1]
            assert(tau_vit > 0)# if the closest flameTime is larger than tRequired, the second closest should be less than, otherwise /that/ would be the closest...?
        elif ((tRequired - flameTime_closestTreq) > 1e-6): # if closest time is more than 1e-6 s less than tReq
            newGas.TPX = flame['T'].iloc[minIndex], flame['P'].iloc[minIndex], flame[moleFracs].iloc[minIndex] 
            assert((newGas['NO'].X - flame['X_NO'].iloc[minIndex] <= 1e-6))    
            tau_vit = tRequired - tList[minIndex] 
            tau_vit_start = tList[minIndex]
        else:
            newGas.TPX = flame['T'].iloc[minIndex], flame['P'].iloc[minIndex], flame[moleFracs].iloc[minIndex]
            tau_vit = 0
            assert((newGas['NO'].X - flame['X_NO'].iloc[minIndex] <= 1e-6))    
    if tau_vit > 0:
        vitiator = ct.ConstPressureReactor(newGas)
        vitRN = ct.ReactorNet([vitiator])
        dt = 0.00001 * 1e-3
        vit_tList = np.arange(0, tau_vit, dt)
        vitArray = np.array([None] * len(vit_tList) * len(columnNames)).reshape(len(vit_tList), len(columnNames))
        # performance note: we don't need to do this. we can simply just advance to the desired tau_main
        for i in range(0, len(vit_tList)):
            MWi = vitiator.thermo.mean_molecular_weight
            vitArray[i, :] = np.hstack([vel_final * dt, vel_final, vitiator.thermo.T, 0, MWi, vitiator.thermo.Y, vitiator.thermo.X, vitiator.thermo.P])
            vitRN.advance(vit_tList[i])
        vit_tList += tau_vit_start             
        vitDF = pd.DataFrame(data=vitArray, index=vit_tList, columns=columnNames, dtype=np.float64)
        # print("Vitiator end time:", vit_tList[-1]/1e-3, "milliseconds") 
        vitiator.syncState()
        '''Call syncState so that newGas has the right state for use in later functions.'''
        minIndex = -1  # last index in time list 
        mainBurnerDF = mainBurnerDF[np.array(mainBurnerDF.index > 0) & np.array(mainBurnerDF.index <= tRequired)]
        # print("Initial mainBurnerDF length:", len(mainBurnerDF.index.values))
        mainBurnerDF = pd.concat([mainBurnerDF, vitDF])        
        # print("New mainBurnerDF length:", len(mainBurnerDF.index.values))
    else:
        mainBurnerDF = mainBurnerDF[np.array(mainBurnerDF.index > 0) & np.array(mainBurnerDF.index <= tRequired)]
    mainBurnerDF['NOppmvd'] = correctNOx(mainBurnerDF['X_NO'].values, mainBurnerDF['X_H2O'].values, mainBurnerDF['X_O2'].values)
    mainBurnerDF['COppmvd'] = correctNOx(mainBurnerDF['X_CO'].values, mainBurnerDF['X_H2O'].values, mainBurnerDF['X_O2'].values)    
    return newGas, minIndex, mainBurnerDF

def runFlame(gas, slope=0.01, curve=0.01):
    # Simulation parameters
    width = 0.06  # m
    # Flame object
    f = ct.FreeFlame(gas, width=width)
    f.set_refine_criteria(ratio=2, slope=slope, curve=curve,
                          prune=min(slope, curve)/1e3)
    f.transport_model = 'Multi'
    f.soret_enabled = True
    f.max_grid_points = 10000
    f.solve(loglevel=1, auto=True, refine_grid=True)
    # Convert distance into time:
    CH2O = gas.species_index('CH2O')
    X_CH2O = f.X[CH2O]
    maxIndex = np.arange(0, len(X_CH2O))[X_CH2O == max(X_CH2O)][0]
    maxIndex2 = X_CH2O.argmax()
    assert maxIndex == maxIndex2
#     startingIndex = np.arange(0, len(X_CH2O))[X_CH2O >= X_CH2O[0] + 5][0]
    startingIndex = maxIndex
    #     startingIndex = np.arange(0, len(f.heat_release_rate))[f.heat_release_rate == max(f.heat_release_rate)][0]
    u_avg = np.array(f.u[startingIndex:] + f.u[startingIndex - 1:-1]) * 0.5
    dx = np.array(f.grid[startingIndex:] - f.grid[startingIndex - 1:-1])
    dt = dx / u_avg

    # need to add some padding to the time array to account for dist = 0.0
    pre_time = [-999] * (startingIndex-1)
    pre_time.extend([0])
    time = np.hstack(
        (np.array(pre_time), np.cumsum(dt)))  # numerically integrate (read: sum) dt to get array of times for each x location

    return f, time

def runMainBurner(phi_main, tau_main, T_fuel=300, T_ox=650, P=25*101325, mech="gri30.xml", slope=0.01, curve=0.01, filename=None):
    flameGas = premix(phi_main, P=P, mech=mech, T_fuel=T_fuel, T_ox=T_ox)

    if filename == None:
        filename = '{0}_{1:.4f}.pickle'.format('phi_main', phi_main)
    if os.path.isfile(filename):
        mainBurnerDF = pd.read_parquet(filename)
        flameTime = mainBurnerDF.index.values
    else:
        flame, flameTime = runFlame(flameGas, slope=slope, curve=curve)
        columnNames = ['x', 'u', 'T', 'n', 'MW'] + ["Y_" + sn for sn in flameGas.species_names] + ["X_" + sn for sn in
                                                                                                   flameGas.species_names]
        flameData = np.concatenate(
            [np.array([flame.grid]), np.array([flame.u]), np.array([flame.T]), np.array([[0] * len(flame.T)]), np.array([[0] * len(flame.T)]), flame.Y, flame.X], axis=0)
        mainBurnerDF = pd.DataFrame(
            data=flameData.transpose(), index=flameTime, columns=columnNames)
        mainBurnerDF.index.name = 'Time'
        mainBurnerDF['P'] = flame.P
        mainBurnerDF.to_parquet(filename, compression='gzip')

    vitiatedProd, flameCutoffIndex, mainBurnerDF = getStateAtTime(
        mainBurnerDF, flameTime, tau_main)
    vitReactor = ct.ConstPressureReactor(vitiatedProd, name='MainBurner')
    return vitReactor, mainBurnerDF

@jit(nopython=True, fastmath=True)
def solvePhi_airSplit(phiGlobal, phiMain, airSplit=1):
    fs = 0.058387057492574147288255659304923028685152530670166015625
    mfm = 1
    mam = 1/(phiMain*fs)
    if airSplit == 0: 
        mam = 0
        mas = 1
    elif airSplit < 1:
        mas = mam * (1-airSplit)/airSplit
    else: 
        mas = 0
    mfs = phiGlobal * fs * (mam + mas) - mfm    
    m_total = mfm + mam + mfs + mas
    return mfm/m_total, mam/m_total, mfs/m_total, mas/m_total

def equil(phi, T_air = 650, T_fuel = 300, P = 25*ct.one_atm, mech="gri30.xml"): 
    gas = ct.Solution(mech)  
    fs_CH4 = 0.058387057492574147288255659304923028685152530670166015625

    fuelGas = ct.Solution('gri30.xml')
    fuelGas.TPX = T_fuel, P, {'CH4':1}
    fuel = ct.Quantity(fuelGas)
    fuel.mass = phi*fs_CH4

    airGas = ct.Solution('gri30.xml')
    airGas.TPX = T_air, P, {'N2':0.79, 'O2':0.21}
    air = ct.Quantity(airGas)
    air.mass = 1

    fuel.constant = air.constant = 'HP'  # keep enthalpy and pressure constant
    mixture = fuel + air  # mix at constant HP
    mixture = mixture.phase

    mixture.equilibrate('HP');  
    CO_ppmvd = correctNOx(mixture['CO'].X, mixture['H2O'].X, mixture['O2'].X) 
    NO_ppmvd = correctNOx(mixture['NO'].X, mixture['H2O'].X, mixture['O2'].X) 
    return np.hstack([mixture.T, CO_ppmvd, NO_ppmvd]) 

def create_combustor_network():
    pass