import torch
import state_config as stype

def velocity_advection(state, pos_tensor, pos_middle_tensor, device, timestep=0.1, x_cells=200, y_cells=150):
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
    
        x0 = prev_x.floor().long()
        x1 = x0 + 1
        xfrac = prev_x - x0.float()
    
        y0 = prev_y.floor().long()
        y1 = y0 + 1
        yfrac = prev_y - y0.float()
    
        #handle obstacles
    
        w00 = (1 - xfrac) * (1 - yfrac)
        w10 = xfrac * (1 - yfrac)
        w01 = (1 - xfrac) * yfrac
        w11 = xfrac * yfrac
    
        m00 = (state[stype.CELL_TYPE, y0, x0] > 0).to(state.dtype)
        m10 = (state[stype.CELL_TYPE, y0, x1] > 0).to(state.dtype)
        m01 = (state[stype.CELL_TYPE, y1, x0] > 0).to(state.dtype)
        m11 = (state[stype.CELL_TYPE, y1, x1] > 0).to(state.dtype)
    
        valid_weights = torch.stack(
            (w00 * m00, w10 * m10, w01 * m01, w11 * m11)
        )
        weight_sum = valid_weights.sum(dim=0)
    
        weighted_density = (
            w00 * m00 * state[stype.VELOCITY_X, y0, x0]
            + w10 * m10 * state[stype.VELOCITY_X, y0, x1]
            + w01 * m01 * state[stype.VELOCITY_X, y1, x0]
            + w11 * m11 * state[stype.VELOCITY_X, y1, x1]
        )
    
        vx = torch.where(
            weight_sum > 0,
            weighted_density / weight_sum.clamp_min(1e-8),
            torch.zeros_like(weighted_density),
        )
    
        #interpolate y-vel at previous position
    
        prev_x = pos_tensor[0,1:-1,1:-1] - timestep * vx_avg_tensor
        prev_y = (pos_middle_tensor[1,1:-1,1:-1] - timestep * state[stype.VELOCITY_Y, 1:-1, 1:-1]) - 0.5
    
        prev_x = prev_x.clamp(1.0, x_cells - 2.0)
        prev_y = prev_y.clamp(1.0, y_cells - 2.0)
    
        x0 = prev_x.floor().long()
        x1 = x0 + 1
        xfrac = prev_x - x0.float()
    
        y0 = prev_y.floor().long()
        y1 = y0 + 1
        yfrac = prev_y - y0.float()
    
        #handle obstacles
    
        w00 = (1 - xfrac) * (1 - yfrac)
        w10 = xfrac * (1 - yfrac)
        w01 = (1 - xfrac) * yfrac
        w11 = xfrac * yfrac
    
        m00 = (state[stype.CELL_TYPE, y0, x0] > 0).to(state.dtype)
        m10 = (state[stype.CELL_TYPE, y0, x1] > 0).to(state.dtype)
        m01 = (state[stype.CELL_TYPE, y1, x0] > 0).to(state.dtype)
        m11 = (state[stype.CELL_TYPE, y1, x1] > 0).to(state.dtype)
    
        valid_weights = torch.stack(
            (w00 * m00, w10 * m10, w01 * m01, w11 * m11)
        )
        weight_sum = valid_weights.sum(dim=0)
    
        weighted_density = (
            w00 * m00 * state[stype.VELOCITY_Y, y0, x0]
            + w10 * m10 * state[stype.VELOCITY_Y, y0, x1]
            + w01 * m01 * state[stype.VELOCITY_Y, y1, x0]
            + w11 * m11 * state[stype.VELOCITY_Y, y1, x1]
        )
    
        vy = torch.where(
            weight_sum > 0,
            weighted_density / weight_sum.clamp_min(1e-8),
            torch.zeros_like(weighted_density),
        )

        #update velocities
    
        state[stype.VELOCITY_X, 1:-1, 1:-1] = vx * state[stype.CELL_TYPE, 1:-1, 1:-1]
        state[stype.VELOCITY_Y, 1:-1, 1:-1] = vy * state[stype.CELL_TYPE, 1:-1, 1:-1]
    
