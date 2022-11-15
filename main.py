import json
import os
# import random
import numpy as np
import copy
# import time as tm
import matplotlib.pyplot as plt

from bonsai_common import SimulatorSession, Schema
# import dotenv
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

from Chuveiro_Turbinado import MalhaFechada, ChuveiroTurbinado

class ChuveiroTurbinadoSimulation(SimulatorSession):
# class ChuveiroTurbinadoSimulation():
    
    def reset(
        self,
        Sr_0: float = 50,
        Sa_0: float = 50,
        xq_0: float = 0.3,
        xf_0: float = 0.5,
        xs_0: float = 0.4672,
        Fd_0: float = 0,
        Td_0: float = 25,
        Tinf: float = 25,
        T0: list = [50,  30 ,  30,  30],
        SPh_0: float = 60,
    ):
        """
        Chuveiro turbinado para simulação.
        ---------------------------------
        Parâmetros:
        Sr_0: seletor que define a fração da resistência elétrica utilizada no aquecimento do tanque de mistura.
        Sa_0: seletor que define a fração para aquecimento à gás do boiler.
        xq_0: abertura da válvula para entrada da corrente quente aquecida pelo boiler.
        xf_0: abertura da válvula para entrada da corrente fria.
        xs_0: abertura da válvula para saída da corrente na tubulação final.
        Fd_0: vazão da corrente de distúrbio.
        Td_0: temperatura da corrente de distúrbio.
        Tinf: temperatura ambiente.
        T0: condições iniciais da simulação (nível do tanque, temperatura do tanque de mistura, 
            temperatura de aquecimento do boiler, temperatura de saída.
        SPh_0: setpoint inicial do nível do tanque de mistura da corrente fria com quente.
        """
        
        self.Sr = Sr_0
        self.Sa = Sa_0
        self.xq = xq_0
        self.xf = xf_0
        self.xs = xs_0
        self.Fd = Fd_0
        self.Td = Td_0
        self.Tinf = Tinf
        self.T0 = T0
        self.SPh = SPh_0
        
        self.time = 10
        self.dt = 0.01
        self.time_sample = 10 # minutos
        
        # Definindo TU: tempo, Sr, Sa, xq, SP(h), xs, Fd, Td, Tinf
        TU = np.array(
        [   
              [0, self.Sr, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf],
              [self.time, self.Sr, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]
        ])
        # print(TU)

        # Simulação malha fechada com controladores PID no nível do tanque h: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, TU, Kp_T4a = [37, 4.5], Ti_T4a = [1, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = self.dt)

        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas:
        self.TT, self.YY, self.UU = malha_fechada.solve_system()
        
        # Valores finais das variáveis de estado (no tempo final):
        self.h = self.YY[:,0][-1]
        self.T4a = self.YY[:,-1][-1]
        self.T3 = self.YY[:,1][-1]
        self.Tq = self.YY[:,2][-1]
        
        # Valores finais das variáveis manipuladas e distúrbios:
        self.Sr = self.UU[:,0][-1]
        self.Sa = self.UU[:,1][-1]
        self.xq = self.UU[:,2][-1]
        self.xf = self.UU[:,3][-1]
        self.xs = self.UU[:,4][-1]
        self.Fd = self.UU[:,5][-1]
        self.Td = self.UU[:,6][-1]
        self.Tinf = self.UU[:,7][-1]
    
        # Cálculo do índice de qualidade do banho:
        self.iqb = malha_fechada.compute_iqb(self.YY[:,-1], # T4a
                                             self.UU[:,4], # xs
                                             self.TT)
        if np.isnan(self.iqb) or self.iqb == None or np.isinf(abs(self.iqb)):
            self.iqb = 0

        # Cálculo do custo do banho:
        self.custo_eletrico, self.custo_gas = malha_fechada.custo_banho(self.UU[:,0], # Sr
                                                                        # self.YY[:,2], # Tq
                                                                        # self.UU[:,7], # Tinf
                                                                        self.UU[:,1]) # Sa

        # Cálculo do custo da água:
        self.custo_agua = malha_fechada.custo_agua(self.UU[:,4]) # xs

        # Vazão final Fs:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))

        self.TU = TU
        self.TU_list = copy.deepcopy(TU)

        # Salva o estado atual:
        self.last_TU = copy.deepcopy(np.array([[self.time, self.Sr, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])

    def episode_start(self, config: Schema) -> None:
    # def episode_start(self, config) -> None:
        
        self.reset(
            Sr_0 = config.get('initial_electrical_resistence_fraction') or 50,
            Sa_0 = config.get('initial_gas_boiler_fraction') or 50,
            xq_0 = config.get('initial_hot_valve_opening') or 0.3,
            xf_0 = config.get('initial_cold_valve_opening') or 0.5,
            xs_0 = config.get('initial_out_valve_opening') or 0.4672,
            Fd_0 = config.get('initial_disturbance_current_flow') or 0,
            Td_0 = config.get('initial_disturbance_temperature') or 25,
            Tinf = config.get('initial_room_temperature') or 25, 
            # T0 = config.get('initial_conditions') or [50,  30 ,  30,  30],
            SPh_0 = config.get('initial_setpoint_tank_level') or 60
        )
        
    def step(self):
        
        self.time += self.time_sample 

        # Need to replace the values of the old TU for the values of the next TU
        # This is needed because we are simulating from 2 to 2 minutes, and for each 2 minutes, we can have only ONE action for each variable
        # For example, if in minute 2, SPh = 70, and in minute 4, SPh = 30, SPh will only by 30 in minute 4
        # This is not what we want, because SPh = 30 was selected to be the setpoint from minute 2 to minute 4
        # But the action we selected was valid for the whole episode (2 to 4 minutes), so we need to change SPh in minute 2 to 30
        # The results from the last episode are taken into account in the initial conditions
        self.last_TU[0][1] = self.Sr
        self.last_TU[0][2] = self.Sa
        self.last_TU[0][3] = self.xq
        self.last_TU[0][4] = self.SPh
        self.last_TU[0][5] = self.xs
        self.last_TU[0][6] = self.Fd 
        self.last_TU[0][7] = self.Td
        self.last_TU[0][8] = self.Tinf

        # Atribui novo TU e T0:
        self.TU = np.append(self.last_TU, np.array([[self.time, self.Sr, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        self.T0 = self.last_T0 # h, T3, Tq, T4a
        self.TU_list = np.append(self.TU_list, np.array([[self.time, self.Sr, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        # print(self.TU)
        # print(self.T0)
        # print(self.time)
        # print(self.TU_list)

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, self.TU, Kp_T4a = [37, 4.5], Ti_T4a = [1, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = self.dt)
                    
        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas:
        self.TT, self.YY, self.UU = malha_fechada.solve_system()

        # Valores finais das variáveis de estado (no tempo final):
        self.h = self.YY[:,0][-1]
        self.T4a = self.YY[:,-1][-1]
        self.T3 = self.YY[:,1][-1]
        self.Tq = self.YY[:,2][-1]

        # plt.plot(self.TT, self.YY[:,-1])
        # plt.show()
        
        # Valores finais das variáveis manipuladas e distúrbios:
        self.Sr = self.UU[:,0][-1]
        self.Sa = self.UU[:,1][-1]
        self.xq = self.UU[:,2][-1]
        self.xf = self.UU[:,3][-1]
        self.xs = self.UU[:,4][-1]
        self.Fd = self.UU[:,5][-1]
        self.Td = self.UU[:,6][-1]
        self.Tinf = self.UU[:,7][-1]
    
        # Cálculo do índice de qualidade do banho:
        self.iqb = malha_fechada.compute_iqb(self.YY[:,-1], # T4a
                                             self.UU[:,4], # xs
                                             self.TT) or 0
        if np.isnan(self.iqb) or self.iqb == None or np.isinf(abs(self.iqb)):
            self.iqb = 0

        # Cálculo do custo do banho:
        self.custo_eletrico, self.custo_gas = malha_fechada.custo_banho(self.UU[:,0], # Sr
                                                                        # self.YY[:,2], # Tq
                                                                        # self.UU[:,7], # Tinf
                                                                        self.UU[:,1]) # Sa

        # Cálculo do custo da água:
        self.custo_agua = malha_fechada.custo_agua(self.UU[:,4]) # xs

        # Vazão final Fs:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))
                
        # Salvar o estado atual:
        self.last_TU = copy.deepcopy(np.array([[self.time, self.Sr, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])

    def episode_step(self, action: Schema) -> None:
    # def episode_step(self, action) -> None:
        
        self.xq = action.get('hot_valve_opening')
        self.xs = action.get('out_valve_opening')
        self.Sr = action.get('electrical_resistence_fraction')
        self.Sa = action.get('gas_boiler_fraction')
        self.SPh = action.get('setpoint_tank_level')
        self.Fd = action.get('disturbance_current_flow')
        self.Td = action.get('disturbance_temperature')
        self.Tinf = action.get('room_temperature')

        self.step()

    def get_state(self):
        
        return {  
            'electrical_resistence_fraction': self.Sr,
            'gas_boiler_fraction': self.Sa,
            'hot_valve_opening': self.xq,
            'cold_valve_opening': self.xf,
            'out_valve_opening': self.xs,
            'disturbance_current_flow': self.Fd,
            'disturbance_temperature': self.Td,
            'room_temperature': self.Tinf,
            'setpoint_tank_level': self.SPh,
            'tank_level': self.h,
            'final_temperature': self.T4a,
            'flow_out': self.Fs,
            'final_boiler_temperature': self.Tq,
            'final_temperature_tank': self.T3,
            'quality_of_shower': self.iqb,
            'electrical_cost_shower': self.custo_eletrico,
            'gas_cost_shower': self.custo_gas,
            'cost_water': self.custo_agua,
        }
    
    def halted(self) -> bool:
        
        if self.T4a > 70:
            return True
        else:
            return False

    def get_interface(self) -> SimulatorInterface:
        # Register sim interface.

        with open('interface.json', 'r') as infile:
            interface = json.load(infile)

        return SimulatorInterface(
            name=interface['name'],
            timeout=interface['timeout'],
            simulator_context=self.get_simulator_context(),
            description=interface['description'],
        )

def main():

    workspace = os.getenv('SIM_WORKSPACE')
    access_key = os.getenv('SIM_ACCESS_KEY')

    # values in `.env`, if they exist, take priority over environment variables
    # dotenv.load_dotenv('.env', override=True)

    if workspace is None:
         raise ValueError('The Bonsai workspace ID is not set.')
    if access_key is None:
        raise ValueError('The access key for the Bonsai workspace is not set.')

    config = BonsaiClientConfig(workspace=workspace, access_key=access_key)
    # config = None

    chuveiro_sim = ChuveiroTurbinadoSimulation(config)

    chuveiro_sim.reset()

    while chuveiro_sim.run():
        continue

if __name__ == "__main__":
    main()