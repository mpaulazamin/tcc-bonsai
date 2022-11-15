inkling "2.0"
using Goal

type SimConfig {
    initial_electrical_resistence_fraction: number,
    initial_gas_boiler_fraction: number,
    initial_hot_valve_opening: number,
    initial_cold_valve_opening: number,
    initial_out_valve_opening: number,
    initial_disturbance_current_flow: number,
    initial_disturbance_temperature: number,
    initial_room_temperature: number,
    initial_setpoint_tank_level: number
}

type SimState {
    electrical_resistence_fraction: number,
    gas_boiler_fraction: number,
    hot_valve_opening: number,
    cold_valve_opening: number,
    out_valve_opening: number,
    disturbance_current_flow: number,
    disturbance_temperature: number,
    room_temperature: number,
    setpoint_tank_level: number,
    tank_level: number,
    final_temperature: number,
    flow_out: number,
    final_boiler_temperature: number,
    final_temperature_tank: number,
    quality_of_shower: number,
    electrical_cost_shower: number,
    gas_cost_shower: number,
    cost_water: number
}

type Action {
    hot_valve_opening: number <0.3 .. 0.5>,
    out_valve_opening: number <0.3 .. 0.5>,
    electrical_resistence_fraction: number <50 .. 100>,
    gas_boiler_fraction: number <50 .. 100>,
    setpoint_tank_level: number <50 .. 70>,
    disturbance_current_flow: number <0 .. 1>,
    disturbance_temperature: number <24 .. 26>,
    room_temperature: number <24 .. 26>
}

graph (input: SimState): Action {
    concept Concept(input): Action {
        curriculum {
            source simulator (Action: Action, Config: SimConfig): SimState {
            }

            training {
                EpisodeIterationLimit: 1,
                TotalIterationLimit: 10000
            }

            goal (state: SimState) {
                drive Goal: state.quality_of_shower in Goal.Range(0.9, 1)
            }

            lesson `learn 1` {
                scenario { 
                    initial_electrical_resistence_fraction: number <49 .. 51>,
                    initial_gas_boiler_fraction: number <49 .. 51>,
                    initial_hot_valve_opening: number <0.29 .. 0.31>,
                    initial_cold_valve_opening: number <0.49 .. 0.51>,
                    initial_out_valve_opening: number <0.46 .. 0.47>,
                    initial_disturbance_current_flow: number <0 .. 0.1>,
                    initial_disturbance_temperature: number <24.9 .. 25.1>,
                    initial_room_temperature: number <24.9 .. 25.1>,
                    initial_setpoint_tank_level: number <59 .. 61>
                }
            }
        }
    }
}