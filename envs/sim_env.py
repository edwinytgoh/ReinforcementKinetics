import gym
import cantera as ct
import numpy as np
from gym import spaces
from gym.utils import EzPickle
from envs.SimUtils import solvePhi_airSplit, equil, runMainBurner, correctNOx
milliseconds = 1e-3
DT = 0.001*milliseconds  # this is the time step used in the simulation
MAX_STEPS = 15*milliseconds/DT  # 15 ms total time gives max steps of 15,000
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
    PHI_GLOBAL, PHI_MAIN, airSplit=0.9)

M_fuel_main *= 100
M_air_main *= 100
M_fuel_sec *= 100
M_air_sec *= 100

main_multiplier = 0.15
sec_multiplier = 0.75

main_burner_reactor, main_burner_df = runMainBurner(
PHI_MAIN, TAU_MAIN, T_fuel=T_FUEL, T_ox=T_AIR, P=P, mech=MECH)
# main_burner_backup_state = ct.Solution(MECH)
# main_burner_backup_state.TPX = main_burner_reactor.thermo.TPX

NO_idx = main_burner_reactor.thermo.species_index('NO')
CO_idx = main_burner_reactor.thermo.species_index('CO')
O2_idx = main_burner_reactor.thermo.species_index('O2')
H2O_idx = main_burner_reactor.thermo.species_index('H2O')


class SimEnv(gym.Env, EzPickle):
    """
    Description:
        Three streams are entering a constant pressure reactor at variable rates. The goal is to control the flow rates so that a target temperature is achieved while NO and CO are minimized.

    Observation:
        Type: Box(2(n+1), 5) where n is the number of species
        Num	        Observation                 Min         Max
        ---         ----------------            ---         ----
        0	        Temperature (K)             0           3000
        1	        Density (kg/m3)             0           10
        2-n+1	    Mole fractions              0           1
        n+2-2n+1	Net production rates        -Inf        Inf

    Actions:
        Type: Box(3) - entrain a fraction of remaining fluid streams
        Num         Action                  Min         Max
        ---         ------                  ---         ---
        0           Entrain main burner     0           1
        1           Entrain sec fuel        0           1
        2           Entrain sec air         0           1

        Note: 0 means nothing is entrained, 1 means entrain all remaining fluid
    Reward is a function of:
        - Distance between current temperature and target temperature: smaller distance = higher reward.
        - CO
        - NO
    Starting State:
        Choose final 10 rows from main burner simulation, append density and net production rates
    Episode Termination:
        1. Number of steps is greater than MAX_STEPS
            OR
        2. Temperature is within a certain threshold of the final temperature
            AND
        2. CO is within a certain threshold of the equilibrium CO value 
            OR
        3. Remaining reactants is less than 1e-9
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
        self.steps_taken = 0
        self.reward = 0
        # Create main burner objects
        self.main_burner_gas = ct.Solution(MECH)
        self.main_burner_gas.TPX = main_burner_reactor.thermo.TPX
        self.main_burner_reactor = ct.ConstPressureReactor(
            self.main_burner_gas)
        self.main_burner_reservoir = ct.Reservoir(
            contents=self.main_burner_gas)
        self.remaining_main_burner_mass = M_air_main + M_fuel_main

        # Create secondary stage objects
        self.sec_fuel = ct.Solution(MECH)
        self.sec_fuel.TPX = T_FUEL, P, {'CH4': 1.0}
        self.sec_fuel_reservoir = ct.Reservoir(contents=self.sec_fuel)
        self.sec_fuel_remaining = M_fuel_sec
        self.sec_air = ct.Solution(MECH)
        self.sec_air.TPX = T_AIR, P, {'N2': 0.79, 'O2': 0.21}
        self.sec_air_reservoir = ct.Reservoir(contents=self.sec_air)
        self.sec_air_remaining = M_air_sec

        # Create simulation network model
        self.sec_stage_gas = ct.Solution(MECH)
        self.sec_stage_gas.TPX = 300, P, {'AR': 1.0}
        self.sec_stage_reactor = ct.ConstPressureReactor(self.sec_stage_gas)
        self.sec_stage_reactor.volume = 1e-8
        self.network = ct.ReactorNet(
            [self.main_burner_reactor, self.sec_stage_reactor])

        # Create main and secondary mass flow controllers and connect them to the secondary stage
        self.mfc_main = ct.MassFlowController(
            self.main_burner_reservoir, self.sec_stage_reactor)
        self.mfc_main.set_mass_flow_rate(0)

        self.mfc_air_sec = ct.MassFlowController(
            self.sec_air_reservoir, self.sec_stage_reactor)
        self.mfc_air_sec.set_mass_flow_rate(0)

        self.mfc_fuel_sec = ct.MassFlowController(
            self.sec_fuel_reservoir, self.sec_stage_reactor)
        self.mfc_fuel_sec.set_mass_flow_rate(0)

        # Define action and observation spaces; must be gym.spaces
        # we're controlling three things: mdot_main, mdot_fuel_sec, mdot_air_sec
        self.action_space = spaces.Box(
            low=np.array([[0,0,0]]), 
            high=np.array([[
                main_multiplier * self.remaining_main_burner_mass, 
                sec_multiplier * self.sec_fuel_remaining, 
                sec_multiplier * self.sec_air_remaining]]), 
            dtype=np.float32)
        low = np.array([0]*55 + [-np.finfo(np.float32).max]*53)
        high = np.array([3000, 10] + [1]*53 + [np.finfo(np.float64).max]*53)
        num_cols = len(self.sec_stage_gas.state) + \
            len(self.sec_stage_gas.net_production_rates)
        self.observation_space = spaces.Box(
            low=np.tile(low, (10, 1)),
            high=np.tile(high, (10, 1)),
            dtype=np.float64)

        self.observation_array = self._next_observation(init=True)

    def _next_observation(self, init=False):
        if init:
            #? set initial observation to be 0 or time history from the flame?
            final_ten_rows = main_burner_df.iloc[-10:].copy()
            # append density by assuming that density after the flame is constant
            final_ten_rows['density'] = [main_burner_reactor.thermo.state[1]]*10
            obs_cols = ['T', 'density'] + [col for col in final_ten_rows.columns if 'X_' in col]
            return np.hstack([
                final_ten_rows.loc[:, obs_cols].values,
                np.tile(self.main_burner_gas.net_production_rates, (10, 1)) # repmat 10 rows
            ])        
        else:
            # get observations and calculate rewards by stacking last 9 observations on top of current observation
            return np.vstack((
                self.observation_array[1:],
                np.hstack([self.sec_stage_gas.state, self.sec_stage_gas.net_production_rates])
            ))            

    def step(self, action):
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        # Calculate mdots based on action input (ideally predicted by model)
        # action is a 1x3 array, so take the first row first
        action = action[0]
        mdot_main = action[0]/self.dt # note: need to do integral. see MassCalc.docx
        mdot_fuel_sec = action[1]/self.dt
        mdot_air_sec = action[2]/self.dt


        self.mfc_main.set_mass_flow_rate(mdot_main)
        self.mfc_fuel_sec.set_mass_flow_rate(mdot_fuel_sec)
        self.mfc_air_sec.set_mass_flow_rate(mdot_air_sec)

        # Advance the reactor network (sec_stage_reactor and main_burner_reactor) by dt
        self.network.set_initial_time(self.age) # important!! 
        self.age += self.dt
        self.steps_taken += 1
        self.network.advance(self.age)

        # sync the main burner reservoir state to match the updated main burner reactor
        # TODO: check if main burner reservoir contents are being updated
        self.main_burner_reservoir.syncState()

        self.remaining_main_burner_mass -= action[0]
        self.sec_fuel_remaining -= action[1]
        self.sec_air_remaining -= action[2]
        
        # update action space
        #TODO: find a way to set entrainment rate based on physical limitations, i.e., can't be infinitely fast
        self.action_space = spaces.Box(
            low=np.array([[0,0,0]]), 
            high=np.array([[
                main_multiplier*self.remaining_main_burner_mass, 
                sec_multiplier*self.sec_fuel_remaining, 
                sec_multiplier*self.sec_air_remaining]]), 
            dtype=np.float32)
        
        self.observation_array = self._next_observation() # update observation array

        # convert NO and CO from mole fractions into volumetric ppm
        NO_ppmvd = correctNOx(
            self.sec_stage_gas.X[NO_idx],
            self.sec_stage_gas.X[H2O_idx],
            self.sec_stage_gas.X[O2_idx]) 
        
        CO_ppmvd = correctNOx(
            self.sec_stage_gas.X[CO_idx],
            self.sec_stage_gas.X[H2O_idx],
            self.sec_stage_gas.X[O2_idx])
        
        T_distance = np.abs(self.sec_stage_gas.T - T_eq)
        T_threshold = 0.15*T_eq

        CO_distance = CO_ppmvd - CO_eq
        CO_threshold = 0.25*CO_eq
        reward_T = -10*(T_distance/T_threshold)**3 + 10
        reward_NO = -5*(NO_ppmvd/25)**3 + 5
        reward_CO = -5*(CO_distance/CO_threshold)**3 + 5 #TODO: Check whether this increases for CO_ppmvd < CO_eq
        reward = reward_T + reward_NO + reward_CO - 0.5*self.steps_taken # penalize for every extra step taken
        self.reward = reward
        game_over = self.steps_taken > MAX_STEPS \
                    or (\
                        T_distance <= T_threshold \
                        and np.abs(CO_distance) <= CO_threshold \
                        and self.remaining_main_burner_mass <= 1e-9 \
                        and self.sec_air_remaining <= 1e-9 \
                        and self.sec_fuel_remaining <= 1e-9                         
                    )
        return self.observation_array, reward, game_over, {}

    def reset(self):
        self.__init__() # nuclear option: possibly slower but safe 
        return self._next_observation(init=True)

    def render(self, mode='human'):
        # convert NO and CO from mole fractions into volumetric ppm
        NO_ppmvd = correctNOx(
            self.sec_stage_gas.X[NO_idx],
            self.sec_stage_gas.X[H2O_idx],
            self.sec_stage_gas.X[O2_idx]) 
        
        CO_ppmvd = correctNOx(
            self.sec_stage_gas.X[CO_idx],
            self.sec_stage_gas.X[H2O_idx],
            self.sec_stage_gas.X[O2_idx])

        T_distance = np.abs(self.sec_stage_gas.T - T_eq)
        T_threshold = 0.15*T_eq

        CO_distance = CO_ppmvd - CO_eq
        CO_threshold = 0.25*CO_eq

        reward_T = -10*(T_distance/T_threshold)**3 + 10
        reward_NO = -5*(NO_ppmvd/25)**3 + 5
        reward_CO = -5*(CO_distance/CO_threshold)**3 + 5 #TODO: Check whether this increases for CO_ppmvd < CO_eq
        
        phi = self.sec_stage_gas.get_equivalence_ratio()
        phi_norm = phi/(1 + phi)
        T = self.sec_stage_gas.T
        if self.steps_taken == 0:
            print(f"step\tage (ms)\tT\tphi_norm\tNO\tCO\tReward\tReward T\tReward NO\tReward CO")
            print(f"=============================================================")
        print(
        f"{self.steps_taken}\t",
        f"{self.age/milliseconds:.2f}\t",
        f"{T:.2f}\t",
        f"{phi_norm:.2f}\t",
        f"{NO_ppmvd:.2f}\t",
        f"{CO_ppmvd:.2f}\t", 
        f"{self.reward:.2f}\t",
        f"{reward_T:.2f}\t{reward_NO:.2f}\t{reward_CO:.2f}"
        )        
        # return 

    def close(self):
        return
