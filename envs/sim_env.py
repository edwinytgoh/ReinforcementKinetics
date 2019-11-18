import gym
import cantera as ct
import numpy as np
from gym import spaces
from gym.utils import EzPickle
from envs.SimUtils import solvePhi_airSplit, equil, runMainBurner, correctNOx, sigmoid

milliseconds = 1e-3
DT = 0.001*milliseconds  # this is the time step used in the simulation
MAX_STEPS = 16*milliseconds/DT  # 15 ms total time gives max steps of 15,000
TAU_MAIN = 15 * milliseconds  # constant main burner length
# definition: phi = (m_fuel/m_air)/(m_fuel/m_air)_stoich; phi > 1 means excess fuel; < 1 means excess air
PHI_MAIN = 0.3719  # i.e., m_fuel/m_air = 0.3719 * stoichiometric fuel-air-ratio
PHI_GLOBAL = 0.635  # for 1975 K final temperature
T_eq, CO_eq, NO_eq = equil(PHI_GLOBAL)
CO_threshold = 1.25*CO_eq 

T_FUEL = 300
T_AIR = 650
P = 25*101325
# chemical mechanism containing kinetic rates and thermodynamic properties
MECH = 'gri30.xml' # GRI-MECH 3.0 contains 53 species and 325 reactions, and is optimized for NOx production rates


# calculate total "reservoir" of each reactant stream based on equivalence ratio and air split
# airSplit = 0.75 means 25% of the air coming from compressor is diverted into secondary stage
M_fuel_main, M_air_main, M_fuel_sec, M_air_sec = solvePhi_airSplit(
    PHI_GLOBAL, PHI_MAIN, airSplit=0.9)

M_fuel_main *= 100
M_air_main *= 100
M_fuel_sec *= 100
M_air_sec *= 100

tau_ent_main = 5 * milliseconds
tau_ent_sec = 1 * milliseconds

main_burner_reactor, main_burner_df = runMainBurner(
PHI_MAIN, TAU_MAIN, T_fuel=T_FUEL, T_ox=T_AIR, P=P, mech=MECH)
delta_T = T_eq - main_burner_reactor.thermo.T

NO_idx = main_burner_reactor.thermo.species_index('NO')
CO_idx = main_burner_reactor.thermo.species_index('CO')
O2_idx = main_burner_reactor.thermo.species_index('O2')
H2O_idx = main_burner_reactor.thermo.species_index('H2O')


class SimEnv(gym.Env, EzPickle):
    """
    Description
    -----------
    Three streams are entering a constant pressure reactor at variable rates. The goal is to control the flow rates so that a target temperature is achieved while NO and CO are minimized.

    Observation
    -----------
    Type: Box(2(n+1), 10) where n is the number of species (53 in the default GRI-MECH 3.0 chemical mechanism)
        Num	            Observation                 Min         Max
        ---             -----------                 ---         ---
        0	            Temperature (K)             0           3000
        1	            Density (kg/m3)             0           10
        2 - n+1         Mole fractions              0           1
        n+2 - 2n+1      Net production rates        -Inf        Inf
                    
    
    Actions
    -------
    Type: Box(3) - set an entrainment (i.e., mass flow) rate for each fluid stream in kilograms per second (kg/s) 
    Note: This rate is constraint by physical limits, which are specified through entrainment time scales tau_ent_main and tau_ent_sec

        Num         Action                  Min         Max
        ---         ------                  ---         ---
        0           Entrain main burner     0           self.remaining_main_burner_mass/tau_ent_main
        1           Entrain sec fuel        0           sec_fuel_remaining/tau_ent_sec
        2           Entrain sec air         0           sec_air_remaining/tau_ent_sec

        Note #2: This is to constrain the agent to cases where entrainment isn't infinite; it is physically impossible to entrain remaining mass in one timestep (dt = 0.001 ms)
    
    Reward
    ------
        Depends on:
            - Distance between current temperature and target temperature: smaller distance = higher reward.
            - CO
            - NO

    Starting State
    --------------
    Choose final 10 rows from main burner simulation, append density and net production rates
    
    Episode Termination
    -------------------
    1. Number of steps is greater than MAX_STEPS
        OR
    2. Temperature is within a certain threshold of the final temperature
        AND Remaining reactants is less than 1e-9
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """Constructor for the SimEnv class. Call SimEnv(...) to create a SimEnv object.

        Arguments:
            None
        """

        # super(SimEnv) returns the superclass, in this case gym.Env. Construct a gym.Env instance
        # gym.Env.__init__(self)
        # EzPickle.__init__(self)

        self.dt = DT
        self.age = 0
        self.steps_taken = 0
        self.reward = 0
        self.T_within_threshold = False
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
        self.sec_stage_reactor.volume = 1e-8 # second stage starts off small, and grows as mass is entrained
        self.network = ct.ReactorNet(
            [self.main_burner_reactor, self.sec_stage_reactor])

        # Create main and secondary mass flow controllers and connect them to the secondary stage
        self.mfc_main = ct.MassFlowController(
            self.main_burner_reservoir, self.sec_stage_reactor)
        self.mfc_main.set_mass_flow_rate(0) # zero initial mass flow rate

        self.mfc_air_sec = ct.MassFlowController(
            self.sec_air_reservoir, self.sec_stage_reactor)
        self.mfc_air_sec.set_mass_flow_rate(0)

        self.mfc_fuel_sec = ct.MassFlowController(
            self.sec_fuel_reservoir, self.sec_stage_reactor)
        self.mfc_fuel_sec.set_mass_flow_rate(0)

        # Define action and observation spaces; must be gym.spaces
        # we're controlling three things: mdot_main, mdot_fuel_sec, mdot_air_sec
        #TODO: check to see if chosen tau_ent and self.action_space.high definition allow for complete entrainment of mass
        self.action_space = spaces.Box(
            low=np.array([0,0,0]), 
            high=np.array([
                self.remaining_main_burner_mass/tau_ent_main, 
                self.sec_fuel_remaining/tau_ent_sec, 
                self.sec_air_remaining/tau_ent_sec]), 
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
        
        # Reward variables for T, NO, and CO
        self.reward_T = 0
        self.reward_NO = 0 
        self.reward_CO = 0

    def _next_observation(self, init=False):
        """Return the next observation, i.e., 10 states up to and including this time step
        """

        if init: # initial observation is final 10 rows from main burner
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
            # get observations and calculate rewards by stacking previous 9 observations on top of current observation
            return np.vstack((
                self.observation_array[1:],
                np.hstack([self.sec_stage_gas.state, self.sec_stage_gas.net_production_rates])
            ))            

    def calculate_reward(self):
        
        # penalise SUPER HEAVILY if agent doesn't use up all reactants within 16 ms
        if self.steps_taken == MAX_STEPS: 
            self.reward = np.finfo(np.float64).min
            return self.reward
        else:
            # convert NO and CO from mole fractions into volumetric ppm
            NO_ppmvd = correctNOx(
                self.sec_stage_gas.X[NO_idx],
                self.sec_stage_gas.X[H2O_idx],
                self.sec_stage_gas.X[O2_idx]) 
           
            # Temperature reward
            T = self.sec_stage_gas.T
            T_distance_percent = np.abs(T - T_eq)/T_eq
            T_threshold_percent = 0.9*0.005 # +-10K for 1975K 
            self.reward_T = 100*sigmoid(-T_threshold_percent*1000*(T_distance_percent - T_threshold_percent)) # see reward shaping.ipynb
            self.T_within_threshold = T_distance_percent < 0.005
            # Remaining reactants 
            reactants_left = self.remaining_main_burner_mass + self.sec_air_remaining + self.sec_fuel_remaining
            reactants_left_percent = 100*reactants_left/(M_fuel_main + M_air_main + M_fuel_sec + M_air_sec) # initial mass should be 100 
            self.reward_reactants = (100 - reactants_left_percent)**2

            if reactants_left_percent <= 5:     
                CO_ppmvd = correctNOx(
                    self.sec_stage_gas.X[CO_idx],
                    self.sec_stage_gas.X[H2O_idx],
                    self.sec_stage_gas.X[O2_idx])                            
                self.reward_CO = 100*sigmoid(-3*(CO_ppmvd - CO_threshold - 2)) #Note: CO_threshold = 1.25 CO_eq
            else:
                self.reward_CO = 0                 
            self.reward_NO = 100*sigmoid(-0.4*(NO_ppmvd-15))
            self.reward = (self.reward_T + self.reward_NO + self.reward_CO - (self.age/milliseconds)**3) # penalize for long times        

    def step(self, action):
        """
        Advance the state of simulation environment by one timestep (given by self.dt). 

        Parameters
        ----------
        action : array_like
            A collection/sequence of actions as described in the SimEnv docstring. 
            action[0] — the mass flow rate in kilograms per second (kg/s) of the main burner fluid/products
            action[1] — mass flow rate (kg/s) of the secondary fuel
            action[2] — mass flow rate (kg/s) of the secondary air/oxidizer
        
        Returns
        -------
        self.observation_array : np.ndarray
            A [2(n+1) x 10] array where n is the number of species (53 in GRI-MECH 3.0)
            This represents the 10 latest thermodynamic states up to and including the current time step.
        self.reward : float
            A value representing the reward/score of the environment at its current state.

        Notes
        -----
            1. The action space may eventually be expanded to include more secondary reactants, e.g., H2O, He, etc.
            2. self.reward may or may not be cumulative across the entire history of the environment. This behavior is TBD.        
            3. Internally, the substeps are as follows: 
                i.   Set secondary stage reactor's mass flow controllers to the input mass flow rates. 
                ii.  Advance the reactor network by self.dt, taking as many internal integrator steps as necessary.
                iii. Update remaining reservoir mass and increment relevant counters 
                iv.  Update action space to ensure that flow rate does not exceed remaining mass in reservoirs
                v.   Calculate reward, get observation, and check if episode is complete 
        """

        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))

        # Calculate mdots based on action input (typically predicted by model/agent policy)
        # action is a 1x3 array, so take the first row first
        mdot_main = action[0] 
        mdot_fuel_sec = action[1]
        mdot_air_sec = action[2]


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

        self.remaining_main_burner_mass -= action[0] * self.dt
        self.sec_fuel_remaining -= action[1] * self.dt
        self.sec_air_remaining -= action[2] * self.dt

        # update action space
        self.action_space = spaces.Box(
            low=np.array([0,0,0]), 
            high=np.array([
                self.remaining_main_burner_mass/tau_ent_main, 
                self.sec_fuel_remaining/tau_ent_sec, 
                self.sec_air_remaining/tau_ent_sec]), 
            dtype=np.float32)
        
        self.observation_array = self._next_observation() # update observation array

        self.calculate_reward()

        game_over = self.steps_taken > MAX_STEPS \
                    or (\
                        self.T_within_threshold \
                        and self.remaining_main_burner_mass <= 1e-9 \
                        and self.sec_air_remaining <= 1e-9 \
                        and self.sec_fuel_remaining <= 1e-9                         
                    )
        return self.observation_array, self.reward, game_over, {}

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

        phi = self.sec_stage_gas.get_equivalence_ratio()
        phi_norm = phi/(1 + phi)
        T = self.sec_stage_gas.T
        if self.steps_taken < 2:
            print(f"step|age_(ms)|T|phi_norm|NO|CO|Rem_Main|Rem_SecFuel|Rem_SecAir|Mdot_Main|Mdot_SecFuel|Mdot_SecAir|Max_Main|Max_SecFuel|Max_SecAir|Reward|Reward_T|Reward_NO|Reward_CO")
            print(f"=============================================================")
        print(
            f"{self.steps_taken}|",
            f"{self.age/milliseconds:.2f}|",
            f"{T:.2f}|",
            f"{phi_norm:.2f}|",
            f"{NO_ppmvd:.2f}|",
            f"{CO_ppmvd:.2f}|", 
            f"{self.remaining_main_burner_mass:.2f}|",
            f"{self.sec_fuel_remaining:.2f}|",
            f"{self.sec_air_remaining:.2f}|",
            f"{self.mfc_main.mdot(0):.2f}|",
            f"{self.mfc_fuel_sec.mdot(0):.2f}|",
            f"{self.mfc_air_sec.mdot(0):.2f}|",            
            f"{self.action_space.high[0]:.2f}|",
            f"{self.action_space.high[1]:.2f}|",
            f"{self.action_space.high[2]:.2f}|",  
            f"{self.reward:.2f}|",
            f"{self.reward_T:.2f}|{self.reward_NO:.2f}|{self.reward_CO:.2f}"
        )              
        # return 

    def close(self):
        return
