import gym
import cantera as ct
import numpy as np
from gym import spaces
from gym.utils import EzPickle
from SimUtils import solvePhi_airSplit, equil, runMainBurner

milliseconds = 1e-3
DT = 0.001*milliseconds # this is the time step used in the simulation
TAU_MAIN = 15 * milliseconds  # constant main burner stage length
PHI_MAIN = 0.3719  # i.e., m_fuel/m_air = 0.3719 * stoichiometric fuel-air-ratio
PHI_GLOBAL = 0.635  # for 1975 K final temperature
T_eq, CO_eq, NO_eq = equil(PHI_GLOBAL)

T_FUEL = 300
T_AIR = 650
P = 25*101325
# chemical mechanism containing kinetic rates and thermodynamic properties
MECH = 'gri30.xml'

# calculate total "reservoir" of each reactant stream based on equivalence ratio and air split
# airSplit = 0.75 means 25% of the air coming from compressor is diverted into secondary stage
M_fuel_main, M_air_main, M_fuel_sec, M_air_sec = solvePhi_airSplit(
    PHI_GLOBAL, PHI_MAIN, airSplit=0.75)


main_burner_reactor, main_burner_df = runMainBurner(
    PHI_MAIN, TAU_MAIN, T_fuel=T_FUEL, T_ox=T_AIR, P=P)


class SimEnv(gym.Env, EzPickle):
    """
    Description:
        Three streams are entering a constant pressure reactor at variable rates. The goal is to control the flow rates so that a target temperature is achieved while NO and CO are minimized.

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
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

        self.dt = DT
        self.age = 0
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
        self.network = ct.ReactorNet([self.main_burner_reactor, self.sec_stage_reactor])

        # Create main and secondary mass flow controllers and connect them to the secondary stage
        self.mfc_main = ct.MassFlowController(
            self.main_burner_reservoir, self.sec_stage_reactor)
        self.mfc_main.set_mass_flow_rate(0)

        self.mfc_air_sec = ct.MassFlowController(
            self.sec_reservoir, self.sec_stage_reactor)
        self.mfc_air_sec.set_mass_flow_rate(0)

        self.mfc_fuel_sec = ct.MassFlowController(
            self.sec_reservoir, self.sec_stage_reactor)
        self.mfc_fuel_sec.set_mass_flow_rate(0)        

        # Define action and observation spaces; must be gym.spaces
        # we're controlling three things: mdot_main, mdot_fuel_sec, mdot_air_sec
        self.action_space = spaces.Box(low=0, high=1, shape=(1, 3))
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, len(self.sec_stage_gas.state)) + len(self.sec_stage_gas.net_production_rates), dtype=np.float64)

        # set initial observation to be 0 or time history from the flame?
        self._next_observation = 

    @property 
    def dt(self):
        return self.__dt

    @dt.setter
    def dt(self, dt=0.001*milliseconds): 
        self.__dt = dt

    def step(self, action):
        # Calculate mdots based on action input (ideally predicted by model)
        action = action[0] # action is a 1x3 array, so take the first row first
        mdot_main = action[0] * self.remaining_main_burner_mass
        mdot_fuel_sec = action[1] * self.sec_fuel_remaining
        mdot_air_sec = action[2] * self. sec_air_remaining

        self.mfc_main.set_mass_flow_rate(mdot_main)
        self.mfc_fuel_sec.set_mass_flow_rate(mdot_fuel_sec)
        self.mfc_air_sec.set_mass_flow_rate(mdot_air_sec)

        # Advance the reactor network (sec_stage_reactor and main_burner_reactor) by dt
        self.age += self.dt
        self.network.advance(self.age)

        # sync the main burner reservoir state to match the updated main burner reactor 
        self.main_burner_reservoir.syncState() #TODO: check if main burner reservoir contents are being updated 

        self.remaining_main_burner_mass -= mdot_main
        self.sec_air_remaining -= mdot_air_sec
        self.sec_fuel_remaining -= mdot_fuel_sec   

        # get observations and calculate rewards 
        observation = self._get_observation

        return observation, reward, game_over, {}

    def reset(self): 
        return

    def render(self, mode='human'):
        return

    def close(self):
        return