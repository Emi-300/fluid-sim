import torch
import state_config as stype

def velocity_advection(state, pos_tensor, pos_middle_tensor, x_cells, y_cells, timestep=0.1, ):
    #semi-lagrangian advection
    
        #previous location can be approximated as x - t*u  (assume straight path)
        #find average velocity at this position
        #use that average velocity as new u
    
        vy_avg_tensor = 0.25 * (state[stype.VELOCITY_Y, 1:-1, 1:-1] + state[stype.VELOCITY_Y, 1:-1, :-2] + state[stype.VELOCITY_Y, 2:, 1:-1] + state[stype.VELOCITY_Y, 2:, :-2])
        vx_avg_tensor = 0.25 * (state[stype.VELOCITY_X, 1:-1, 1:-1] + state[stype.VELOCITY_X, 1:-1, 2:] + state[stype.VELOCITY_X, :-2, 1:-1] + state[stype.VELOCITY_X, :-2, 2:])
    
        prev_x = (pos_middle_tensor[0, 1:-1, 1:-1] - timestep * state[stype.VELOCITY_X, 1:-1, 1:-1]) - 0.5
        prev_y = pos_tensor[1, 1:-1, 1:-1] - timestep * vy_avg_tensor
    
        prev_x = prev_x.clamp(1.0, x_cells - 2.0)
        prev_y = prev_y.clamp(1.0, y_cells - 2.0)
    
        #interpolate x-vel at previous position
    
        vx = interpolate(state[stype.VELOCITY_X], state[stype.CELL_TYPE], prev_x, prev_y)
    
        #interpolate y-vel at previous position
    
        prev_x = pos_tensor[0,1:-1,1:-1] - timestep * vx_avg_tensor
        prev_y = (pos_middle_tensor[1,1:-1,1:-1] - timestep * state[stype.VELOCITY_Y, 1:-1, 1:-1]) - 0.5
    
        prev_x = prev_x.clamp(1.0, x_cells - 2.0)
        prev_y = prev_y.clamp(1.0, y_cells - 2.0)

        vy = interpolate(state[stype.VELOCITY_Y], state[stype.CELL_TYPE], prev_x, prev_y)

        #update velocities
    
        state[stype.VELOCITY_X, 1:-1, 1:-1] = vx * state[stype.CELL_TYPE, 1:-1, 1:-1]
        state[stype.VELOCITY_Y, 1:-1, 1:-1] = vy * state[stype.CELL_TYPE, 1:-1, 1:-1]

def smoke_advection(state, pos_tensor, pos_middle_tensor, x_cells, y_cells,timestep=0.1 ):

    vx_avg_tensor = 0.5 * (state[stype.VELOCITY_X, 1:-1, 1:-1] + state[stype.VELOCITY_X, 1:-1, 2:])
    vy_avg_tensor = 0.5 * (state[stype.VELOCITY_Y, 1:-1, 1:-1] + state[stype.VELOCITY_Y, 2:, 1:-1])

    prev_x = pos_tensor[0, 1:-1, 1:-1] - timestep * vx_avg_tensor
    prev_y = pos_tensor[1, 1:-1, 1:-1] - timestep * vy_avg_tensor

    prev_x = prev_x.clamp(1.0, x_cells - 2.0)
    prev_y = prev_y.clamp(1.0, y_cells - 2.0)

    density = interpolate(state[stype.DENSITY], state[stype.CELL_TYPE], prev_x, prev_y)

    state[stype.DENSITY, 1:-1, 1:-1] = density * state[stype.CELL_TYPE, 1:-1, 1:-1]

def interpolate(grid,cell_type, x, y):
    x0 = x.floor().long()
    x1 = x0 + 1
    xfrac = x - x0.float()

    y0 = y.floor().long()
    y1 = y0 + 1
    yfrac = y - y0.float()

    w00 = (1 - xfrac) * (1 - yfrac)
    w10 = xfrac * (1 - yfrac)
    w01 = (1 - xfrac) * yfrac
    w11 = xfrac * yfrac
        
    m00 = (cell_type[y0, x0] > 0).to(grid.dtype)
    m10 = (cell_type[y0, x1] > 0).to(grid.dtype)
    m01 = (cell_type[y1, x0] > 0).to(grid.dtype)
    m11 = (cell_type[y1, x1] > 0).to(grid.dtype)
        
    valid_weights = torch.stack(
        (w00 * m00, w10 * m10, w01 * m01, w11 * m11)
    )
    weight_sum = valid_weights.sum(dim=0)
        
    weighted_density = (
        w00 * m00 * grid[y0, x0]
        + w10 * m10 * grid[y0, x1]
        + w01 * m01 * grid[y1, x0]
        + w11 * m11 * grid[y1, x1]
    )
        
    return torch.where(
                weight_sum > 0,
                weighted_density / weight_sum.clamp_min(1e-8),
                torch.zeros_like(weighted_density),
            )