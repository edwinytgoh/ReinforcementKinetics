import gym
import cantera as ct
from gym import spaces
from gym.utils import EzPickle
from SimUtils import create_combustor_network, solvePhi_airSplit

milliseconds = 1e-3
TAU_MAIN = 15 * milliseconds  # constant main burner stage length
PHI_MAIN = 0.3719  # i.e., m_fuel/m_air = 0.3719 * stoichiometric fuel-air-ratio
PHI_GLOBAL = 0.635  # for 1975 K final temperature
T_eq, CO_eq, NO_eq = equil(phi_global)

T_FUEL = 300
T_AIR = 650
P = 25*101325
# chemical mechanism containing kinetic rates and thermodynamic properties
MECH = 'gri30.xml'

# calculate total "reservoir" of each reactant stream based on equivalence ratio and air split
# airSplit = 0.75 means 25% of the air coming from compressor is diverted into secondary stage
M_fuel_main, M_air_main, M_fuel_sec, M_air_sec = solvePhi_airSplit(
    PHI_GLOBAL, PHI_MAIN, airSplit=0.75)


main_burner_reactor, main_burner_df = SimUtils.runMainBurner(
    PHI_MAIN, TAU_MAIN, T_fuel=T_FUEL, T_ox=T_AIR, P=P)


class SimEnv(gym.Env, EzPickle):
    """
    Simulation Environment using Cantera that follows the Open AI gym interface
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """Constructor for the SimEnv class. Call SimEnv(...) to create a SimEnv object.

        Arguments:
            args {[type]} -- [description]
        """

        # super(SimEnv) returns the superclass, in this case gym.Env. Construct a gym.Env instance
        # gym.Env.__init__(self)
        # EzPickle.__init__(self)

        # Create main burner objects
        self.main_burner_gas = ct.Solution(MECH)
        self.main_burner_gas.TPX = main_burner_reactor.thermo.TPX
        self.main_burner_reactor = ct.ConstPressureReactor(main_burner_gas)
        self.main_burner_reservoir = ct.Reservoir(contents=main_burner_gas)
        self.remaining_main_burner_mass = M_air_main + M_fuel_main

        # Create secondary stage objects
        self.sec_fuel = ct.Solution(MECH)
        self.sec_fuel.TPX = T_FUEL, P, {'CH4': 1.0}
        self.sec_fuel_reservoir = ct.Reservoir(contents=sec_fuel)
        self.sec_fuel_remaining = M_fuel_sec
        self.sec_air = ct.Solution(MECH)
        self.sec_air.TPX = T_AIR, P, {'N2': 0.79, 'O2': 0.21}
        self.sec_air_reservoir = ct.Reservoir(contents=sec_air)
        self.sec_air_remaining = M_air_sec

        # Create simulation network model
        self.sec_stage_gas = ct.Solution(MECH)
        self.sec_stage_gas.TPX = 300, P, {'AR': 1.0}
        self.sec_stage_reactor = ct.ConstPressureReactor(self.sec_stage_gas)
        self.sec_stage_reactor.mass = 1e-6
        self.network = ct.ReactorNet([self.sec_stage_reactor])

        # Create main and secondary mass flow controllers and connect them to the secondary stage
        self.mfc_main = ct.MassFlowController(
            self.main_burner_reservoir, self.sec_stage_reactor)
        self.mfc_main.set_mass_flow_rate(0)

        self.mfc_sec_air = ct.MassFlowController(
            self.sec_reservoir, self.sec_stage_reactor)
        self.mfc_sec_air.set_mass_flow_rate(0)

        self.mfc_sec_fuel = ct.MassFlowController(
            self.sec_reservoir, self.sec_stage_reactor)
        self.mfc_sec_fuel.set_mass_flow_rate(0)        

        # Define action and observation spaces; must be gym.spaces
        # we're controlling three things: mdot_main, mdot_fuel_sec, mdot_air_sec
        self.action_space = spaces.Box(low=0, high=1, shape=(1, 3))
        self.observation_space = spaces.Box(low=0, high=1, shape=)

    def step(self, action):
        return observation, reward, game_over, {}

    def reset(self): 
        return

    def render(self, mode='human'):
        return

    def close(self):
        return