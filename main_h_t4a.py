import json
import os
import random
import numpy as np
import copy
import time as tm
import matplotlib.pyplot as plt

from bonsai_common import SimulatorSession, Schema
import dotenv
from microsoft_bonsai_api.simulator.client import BonsaiClientConfig
from microsoft_bonsai_api.simulator.generated.models import SimulatorInterface

from chuveiro_h_t4a import MalhaFechada, ChuveiroTurbinado

# class ChuveiroTurbinadoSimulation(SimulatorSession):
class ChuveiroTurbinadoSimulation():
    
    def reset(
        self,
        Sr_0: float = 50,
        Sa_0: float = 50,
        xq_0: float = 0.3,
        xf_0: float = 0.2,
        xs_0: float = 0.4672,
        Fd_0: float = 0,
        Td_0: float = 25,
        Tinf: float = 25,
        T0: list = [50,  30 ,  30,  30],
        SPh_0: float = 60,
        SPT4a_0: float = 38
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
        SPT4a_0: setpoint inicial da temperatura de saída do sistema.
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
        self.SPT4a = SPT4a_0
        
        self.time = 10
        self.dt = 0.01
        self.time_sample = 10 #minutos
        
        # Definindo TU:
        # Time, SP(T4a), Sa, xq, SP(h), xs, Fd, Td, Tinf
        TU = np.array(
        [   
              [0, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf],
              [self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]
        ])
        # print(TU)

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        # Parâmetros antigos: Kp_T4a = [20.63, 1], Ti_T4a = [1, 1e6]
        # Parâmetros novos para t = 10 minutos: Kp_T4a = [37, 4.5], Ti_T4a = [1, 1e6]
        # Para t = 10 minutos, Kp_T4a = [100, 500], Ti_T4a = [100, 10]
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, TU, Kp_T4a = [37, 4.5], Ti_T4a = [1, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = self.dt)

        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas
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
                                             self.TT) or 0
        if np.isnan(self.iqb) or self.iqb == None or np.isinf(abs(self.iqb)):
            self.iqb = 0

        # Cálculo do custo do banho:
        self.custo_eletrico, self.custo_gas = malha_fechada.custo_banho(self.UU[:,0], # Sr
                                                                        self.UU[:,1]) # Sa

        # Cálculo do custo da água:
        self.custo_agua = malha_fechada.custo_agua(self.UU[:,4]) # xs

        # Diferença entre setpoint e T4a:
        self.deltaspt4a = abs(self.SPT4a - self.T4a)

        # Vazão final Fs:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))

        self.TU = TU
        self.TU_list = copy.deepcopy(TU)

        # Salvar o estado atual:
        self.last_TU = copy.deepcopy(np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])

    # def episode_start(self, config: Schema) -> None:
    def episode_start(self, config) -> None:
        
        self.reset(
            Sr_0 = config.get('initial_electrical_resistence_fraction'),
            Sa_0 = config.get('initial_gas_boiler_fraction'),
            xq_0 = config.get('initial_hot_valve_opening'),
            xf_0 = config.get('initial_cold_valve_opening'),
            xs_0 = config.get('initial_out_valve_opening'),
            Fd_0 = config.get('initial_disturbance_current_flow'),
            Td_0 = config.get('initial_disturbance_temperature'),
            Tinf = config.get('initial_room_temperature'),
            T0 = config.get('initial_conditions'),
            SPh_0 = config.get('initial_setpoint_tank_level'),
            SPT4a_0 = config.get('initial_setpoint_final_temperature')
        )
        
    def step(self):
        
        self.time += self.time_sample 

        # Need to replace the values of the old TU for the values of the next TU
        # This is needed because we are simulating from 2 to 2 minutes, and for each 2 minutes, we can have only ONE action for each variable
        # For example, if in minute 2, SPh = 70, and in minute 4, SPh = 30, SPh will only by 30 in minute 4
        # This is not what we want, because SPh = 30 was selected to be the setpoint from minute 2 to minute 4
        # But the action we selected was valid for the whole episode (2 to 4 minutes), so we need to change SPh in minute 2 to 30
        # The results from the last episode are taken into account in the initial conditions
        print('')
        #print(self.last_TU)
        self.last_TU[0][1] = self.SPT4a
        self.last_TU[0][2] = self.Sa
        self.last_TU[0][3] = self.xq
        self.last_TU[0][4] = self.SPh
        self.last_TU[0][5] = self.xs
        self.last_TU[0][6] = self.Fd 
        self.last_TU[0][7] = self.Td
        self.last_TU[0][8] = self.Tinf
        #print(self.last_TU)
        #print('')
        self.TU = np.append(self.last_TU, np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        self.T0 = self.last_T0 # h, T3, Tq, T4a
        self.TU_list = np.append(self.TU_list, np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]), axis=0)
        # print(self.TU)
        # print(self.T0)
        #print(self.time)
        #print(self.TU_list)

        # Simulação malha fechada com controladores PID no nível do tanque h e na temperatura final T4a: 
        malha_fechada = MalhaFechada(ChuveiroTurbinado, self.T0, self.TU, Kp_T4a = [37, 4.5], Ti_T4a = [1, 1e6], 
                                     Td_T4a = [0.0, 0.0], b_T4a = [1, 1], Kp_h = 1, Ti_h = 0.3, Td_h = 0.0, b_h = 1, 
                                     Ruido = 0.005, U_bias_T4a = 50, U_bias_h = 0.5, dt = self.dt)
                    
        # TT = tempo, YY = variáveis de estado, UU = variáveis manipuladas
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
                                                                        self.UU[:,1]) # Sa

        # Cálculo do custo da água:
        self.custo_agua = malha_fechada.custo_agua(self.UU[:,4]) # xs

        # Diferença entre setpoint e T4a:
        self.deltaspt4a = abs(self.SPT4a - self.T4a)

        # Vazão final Fs:
        self.Fs = (5 * self.xs ** 3 * np.sqrt(30) * np.sqrt(-15 * self.xs ** 6 + np.sqrt(6625 * self.xs ** 12 + 640 * self.xs ** 6 + 16)) / (20 * self.xs** 6 + 1))
                
        # Salvar o estado atual:
        self.last_TU = copy.deepcopy(np.array([[self.time, self.SPT4a, self.Sa, self.xq, self.SPh, self.xs, self.Fd, self.Td, self.Tinf]]))
        self.last_T0 = copy.deepcopy([self.h, self.T3, self.Tq, self.T4a])

    # def episode_step(self, action: Schema) -> None:
    def episode_step(self, action) -> None:
        
        self.xq = action.get('hot_valve_opening')
        self.xs = action.get('out_valve_opening')
        self.SPT4a = action.get('setpoint_final_temperature')
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
            'setpoint_final_temperature': self.SPT4a,
            'final_temperature': self.T4a,
            'flow_out': self.Fs,
            'final_boiler_temperature': self.Tq,
            'final_temperature_tank': self.T3,
            'quality_of_shower': self.iqb,
            'electrical_cost_shower': self.custo_eletrico,
            'gas_cost_shower': self.custo_gas,
            'cost_water': self.custo_agua,
            'difference_setpoint_t4a': self.deltaspt4a
        }
    
    def halted(self) -> bool:
        
        if self.T4a > 63:
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

def main_test():
    
    chuveiro_sim = ChuveiroTurbinadoSimulation()
    chuveiro_sim.reset()
    state = chuveiro_sim.get_state()

    q_list = []
    T4_list = []
    SPT4_list = []
    h_list = []
    SPh_list = []
    Sr_list = []
    Sa_list = []
    time_list = []
    xq_list = []
    xf_list = [] 
    xs_list = [] 

       #Time,  SP(T4a),   Sa,    xq,  SP(h),      Xs,   Fd,  Td,  Tinf
    TU=[[20,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
        [30,       38,   50,   0.3,     60,   0.4672,   0,  25,   25],
        [40,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [50,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [60,       42,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [70,       42,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [80,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [90,       38,   50,   0.3,     70,   0.4672,   0,  25,   25],
        [100,       42,   100,   0.3,     40,   0.4672,   0,  25,   25],
        [110,       42,   100,   0.3,     40,   0.4672,   0,  25,   25],
        [120,       42,   100,   0.3,     40,   0.4672,   0,  25,   25],
        [130,       42,   100,   0.3,     40,   0.4672,   0,  25,   25]]   
    
    for i in range(0, 12):
        
        if chuveiro_sim.halted():
            break
            
        action = {
            'hot_valve_opening': TU[i][3],
            'out_valve_opening': TU[i][-4],
            'setpoint_final_temperature': TU[i][1],
            'gas_boiler_fraction': TU[i][2],
            'setpoint_tank_level': TU[i][4],
            'disturbance_current_flow': TU[i][6],
            'disturbance_temperature': TU[i][7],
            'room_temperature': TU[i][8],
        }
            
        chuveiro_sim.episode_step(action)
        state = chuveiro_sim.get_state()
        print('')
        print(state)
        q_list.append(state['quality_of_shower'])
        SPT4_list.append(state['setpoint_final_temperature'])
        T4_list.append(state['final_temperature'])
        SPh_list.append(state['setpoint_tank_level'])
        h_list.append(state['tank_level'])
        Sr_list.append(state['electrical_resistence_fraction'])
        Sa_list.append(state['gas_boiler_fraction'])
        # time_list.append(TU[i][0])
        xq_list.append(state['hot_valve_opening'])
        xf_list.append(state['cold_valve_opening'])
        xs_list.append(state['out_valve_opening'])

    time_list = range(0, 12)
    plt.figure(figsize=(20, 15))
    plt.subplot(4,2,1)
    plt.plot(time_list, q_list, label='IQB')
    plt.legend()
    plt.subplot(4,2,2)
    plt.plot(time_list, T4_list, label='T4a')
    plt.plot(time_list, SPT4_list, label='Setpoint T4a')
    plt.legend()
    plt.subplot(4,2,3)
    plt.plot(time_list, h_list, label='h')
    plt.plot(time_list, SPh_list, label='Setpoint h')
    plt.legend()
    plt.subplot(4,2,8)
    plt.plot(time_list, Sr_list, label='Sr')
    plt.legend()
    plt.subplot(4,2,4)
    plt.plot(time_list, Sa_list, label='Sa')
    plt.legend()
    plt.subplot(4,2,5)
    plt.plot(time_list, xq_list, label='xq')
    plt.legend()
    plt.subplot(4,2,6)
    plt.plot(time_list, xf_list, label='xf')
    plt.legend()
    plt.subplot(4,2,7)
    plt.plot(time_list, xs_list, label='xs')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main_test()
    # main()