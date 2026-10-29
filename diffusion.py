import torch
from time import perf_counter
import state_config as stype


def diffusion(state,red_cells,black_cells,diffusion_steps=10,diffusion_overrelaxation = 1.0, density = 1.0, cell_width = 1.0, timestep = 1.0):

    divergence_tensor = (state[stype.VELOCITY_X, 1:-1, 2:] * state[stype.CELL_TYPE, 1:-1, 1:-1]
                            - state[stype.VELOCITY_X, 1:-1, 1:-1] * state[stype.CELL_TYPE, 1:-1, :-2]
                            + state[stype.VELOCITY_Y, 2:, 1:-1] * state[stype.CELL_TYPE, 1:-1, 1:-1]
                            - state[stype.VELOCITY_Y, 1:-1, 1:-1] * state[stype.CELL_TYPE, :-2, 1:-1])
    
    
    neighbors_matrix = state[stype.CELL_TYPE, 1:-1, :-2] + state[stype.CELL_TYPE, 1:-1, 2:] + state[stype.CELL_TYPE, :-2, 1:-1] + state[stype.CELL_TYPE, 2:, 1:-1]
    
    for _ in range(diffusion_steps):
    
        pressure_solve_step(state, divergence_tensor, neighbors_matrix,red_cells,black_cells,diffusion_overrelaxation, density, cell_width, timestep)
            
    updatevelocity(state, density, cell_width, timestep)

#times how long the pressure solver takes
def diffusion_timed(state,red_cells,black_cells, device, diffusion_steps=10,diffusion_overrelaxation = 1.0, density = 1.0, cell_width = 1.0, timestep = 1.0):

    divergence_tensor = (state[stype.VELOCITY_X, 1:-1, 2:] * state[stype.CELL_TYPE, 1:-1, 1:-1]
                            - state[stype.VELOCITY_X, 1:-1, 1:-1] * state[stype.CELL_TYPE, 1:-1, :-2]
                            + state[stype.VELOCITY_Y, 2:, 1:-1] * state[stype.CELL_TYPE, 1:-1, 1:-1]
                            - state[stype.VELOCITY_Y, 1:-1, 1:-1] * state[stype.CELL_TYPE, :-2, 1:-1])
    
    
    neighbors_matrix = state[stype.CELL_TYPE, 1:-1, :-2] + state[stype.CELL_TYPE, 1:-1, 2:] + state[stype.CELL_TYPE, :-2, 1:-1] + state[stype.CELL_TYPE, 2:, 1:-1]

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()

    for _ in range(diffusion_steps):
    
        pressure_solve_step(state, divergence_tensor, neighbors_matrix,red_cells,black_cells,diffusion_overrelaxation, density, cell_width, timestep)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    diffusion_solve_ms = (perf_counter() - start) * 1000

    updatevelocity(state, density, cell_width, timestep)

    return diffusion_solve_ms

    
def pressure_solve_step(state, divergence_tensor, neighbors_matrix, red_cells, black_cells,diffusion_overrelaxation, density, cell_width, timestep):

    fluid_mask = state[stype.CELL_TYPE, 1:-1, 1:-1] > 0
    active_cells = fluid_mask & (neighbors_matrix > 0)
    safe_neighbors = neighbors_matrix.clamp_min(1)

     #gauss-seidel iteration
        #potential optimizations - multigrid solver
        #                        - store edge data within tensor                   
    
        #pressure equation: average neighbor pressure - (density)(width / timestep) * (divergence)
        #update red cells
    
    average_pressure_tensor = (state[stype.PRESSURE, 1:-1, :-2] + state[stype.PRESSURE, 1:-1, 2:] + state[stype.PRESSURE, :-2, 1:-1] + state[stype.PRESSURE, 2:, 1:-1]) * fluid_mask
    
    new_pressure_tensor = (average_pressure_tensor - (density * cell_width / timestep) * divergence_tensor) / safe_neighbors
    pressure = state[stype.PRESSURE, 1:-1, 1:-1]
    pressure[...] = torch.where(
        red_cells & active_cells,
        pressure + diffusion_overrelaxation * (new_pressure_tensor - pressure),
        torch.where(active_cells, pressure, torch.zeros_like(pressure)),
    )
    
        #update black cells
    
    average_pressure_tensor = (state[stype.PRESSURE, 1:-1, :-2] + state[stype.PRESSURE, 1:-1, 2:] + state[stype.PRESSURE, :-2, 1:-1] + state[stype.PRESSURE, 2:, 1:-1]) * fluid_mask
    
    new_pressure_tensor = (average_pressure_tensor - (density * cell_width / timestep) * divergence_tensor) / safe_neighbors
    pressure = state[stype.PRESSURE, 1:-1, 1:-1]
    pressure[...] = torch.where(
        black_cells & active_cells,
        pressure + diffusion_overrelaxation * (new_pressure_tensor - pressure),
        torch.where(active_cells, pressure, torch.zeros_like(pressure)),
    )

def updatevelocity(state, density, cell_width, timestep):

    state[stype.VELOCITY_X, 1:-1, 1:-1] -= (timestep / (density * cell_width)) * (state[stype.PRESSURE, 1:-1, 1:-1] - state[stype.PRESSURE, 1:-1, :-2]) * state[stype.CELL_TYPE,1:-1,1:-1] * state[stype.CELL_TYPE,1:-1,:-2] #making sure velocity is not coming from or going to solid cell
    state[stype.VELOCITY_Y, 1:-1, 1:-1] -= (timestep / (density * cell_width)) * (state[stype.PRESSURE, 1:-1, 1:-1] - state[stype.PRESSURE, :-2, 1:-1]) * state[stype.CELL_TYPE,1:-1,1:-1] * state[stype.CELL_TYPE,:-2,1:-1]

