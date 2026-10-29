import pygame
from pygame.locals import *
import torch
import moderngl
import torch.nn.functional as F
import numpy as np
from PIL import Image
from time import perf_counter

import diffusion
import state_config as stype
import drawhelper

# read config file
# for now, hardcode variables

#TODO add different smoke types w/different densities
#want to test flow instabilities


FPS = 120

timestep= 4.0/FPS

x_cells = 200
y_cells = 150

diffusion_steps = 15
diffusion_overrelaxation = 1.4
advection_factor = 1

density = 3

#benchmarking
pressure_solve_ms = 0.0
advect_solve_ms = 0.0
total_frame_ms = 0.0

#configure torch

if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# dims: velx,vely, density, cell type, pressure

VELOCITY_X = stype.VELOCITY_X
VELOCITY_Y = stype.VELOCITY_Y
DENSITY = stype.DENSITY
CELL_TYPE = stype.CELL_TYPE
PRESSURE = stype.PRESSURE

# y  x-->     
# |  
# V  

# cell type - 0 solid, 1 fluid

#needed for advection - dims x,y

#position is normalized:
cell_width = 1.0

pos_tensor = torch.zeros((2, y_cells, x_cells), device=device)
pos_middle_tensor = torch.zeros((2, y_cells, x_cells), device=device)

pos_tensor[0] = torch.arange(x_cells, device=device).unsqueeze(0).repeat(y_cells, 1)
pos_tensor[1] = torch.arange(y_cells, device=device).unsqueeze(1).repeat(1, x_cells)

pos_middle_tensor[0] = torch.arange(x_cells, device=device).unsqueeze(0).repeat(y_cells, 1) + (0.5)
pos_middle_tensor[1] = torch.arange(y_cells, device=device).unsqueeze(1).repeat(1, x_cells) + (0.5)


state = torch.zeros((5, y_cells, x_cells), device=device)

#give each cell zero velocity and a constant density

state[0] = torch.zeros((y_cells, x_cells), device=device)
state[1] = torch.zeros((y_cells, x_cells), device=device)
#state[2] = density * torch.ones((y_cells, x_cells), device=device)

#keep border cells solid (adds padding)
#need extra padding on left and bottom for diffusion calculatiom

state[3, 1:-1, 1:-1] = 1

#grid setup
#  ⏹⏹⏹⏹⏹⏹⏹⏹⏹⏹⏹
#  ⏹⬚⬚⬚⬚⬚⬚⬚⬚⬚⏹
#  ⏹⬚⬚⬚⬚⬚⬚⬚⬚⬚⏹
#  ⏹⬚⬚⬚⬚⬚⬚⬚⬚⬚⏹
#  ⏹⬚⬚⬚⬚⬚⬚⬚⬚⬚⏹
#  ⏹⏹⏹⏹⏹⏹⏹⏹⏹⏹⏹

#cells use staggered x and y velocities

#        cell (i,j)
#
#    ----- x(i,j) ------
#    |        |         |
#    |        V         |
# y(i,j) ->            y(i+1,j) ->
#    |                  |
#    |                  |
#    |                  |
#    ----- x(i,j+1) -----
#             |
#             V



# to add/remove flow and density 
# dims: velx,vely, density,pressure

flow = torch.zeros((4, y_cells, x_cells), device=device)

#smoke inflow from left

dist = 5

bottom = (y_cells//2)-dist
top = (y_cells//2)+dist

flow[0, :, 2:5] = 10  # set velocity in x-direction at the left boundary
flow[2, bottom:top, 1:3] = 1  # set density at the left boundary
flow[3, :, 1:3] = 10  # set pressure in x-direction at the right boundary

#smoke outflow from right

flow[0, :, -5:-2] = 10  # set velocity in x-direction at the right boundary
flow[2, :, -3:-1] = 0  # set density at the right boundary
flow[3, :, -3:-1] = -10  # set pressure in x-direction at the right boundary

#add circle obstacle


gridmy = y_cells // 2
gridmx = x_cells // 3
radius = 10

for x in range(gridmx-radius, gridmx+radius+1):
    for y in range(gridmy-radius, gridmy+radius+1):
        if (x-gridmx)**2 + (y-gridmy)**2 <= radius**2:
            state[CELL_TYPE, y, x] = 0



# DEBUG - assign velocity based on cell coordinates
# state[0] = torch.arange(x_cells, device=device).unsqueeze(0).repeat(y_cells, 1)
# state[1] = torch.arange(y_cells, device=device).unsqueeze(1).repeat(1, x_cells)


# DEBUG 2 - assign random velocity

# state[0] = torch.randn((y_cells, x_cells), device=device)
# state[1] = torch.randn((y_cells, x_cells), device=device)

#set border grid velocity to zero

state[VELOCITY_X, 0, :] = 0
state[VELOCITY_X, -1, :] = 0
state[VELOCITY_X, :, 0] = 0
state[VELOCITY_X, :, 1] = 0
state[VELOCITY_X, :, -1] = 0

state[VELOCITY_Y, 0, :] = 0
state[VELOCITY_Y, 1, :] = 0
state[VELOCITY_Y, -1, :] = 0
state[VELOCITY_Y, :, 0] = 0
state[VELOCITY_Y, :, -1] = 0


#simulation

#diffusion setup
row_indices = torch.arange(1, y_cells - 1, device=device).unsqueeze(1)
column_indices = torch.arange(1, x_cells - 1, device=device).unsqueeze(0)
red_cells = (row_indices + column_indices) % 2 == 0
black_cells = ~red_cells


def update():
    global state, advect_solve_ms, pressure_solve_ms

    add_flow()

    add_force()

    #time overall diffusion step

    pressure_solve_ms = diffusion.diffusion_timed(state, red_cells, black_cells,device, diffusion_steps,diffusion_overrelaxation, density, cell_width, timestep)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()

    velocity_advection()

    smoke_advection()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    advect_solve_ms = (perf_counter() - start) * 1000


#TODO consolidate interpolation into its own function
#     break up backtrace into multiple smaller steps and stop at first solid obstacle
def velocity_advection():
    global state

    #semi-lagrangian advection

    #previous location can be approximated as x - t*u  (assume straight path)
    #find average velocity at this position
    #use that average velocity as new u

    vy_avg_tensor = 0.25 * (state[VELOCITY_Y, 1:-1, 1:-1] + state[VELOCITY_Y, 1:-1, :-2] + state[VELOCITY_Y, 2:, 1:-1] + state[VELOCITY_Y, 2:, :-2])
    vx_avg_tensor = 0.25 * (state[VELOCITY_X, 1:-1, 1:-1] + state[VELOCITY_X, 1:-1, 2:] + state[VELOCITY_X, :-2, 1:-1] + state[VELOCITY_X, :-2, 2:])

    prev_x = (pos_middle_tensor[0, 1:-1, 1:-1] - timestep * state[VELOCITY_X, 1:-1, 1:-1]) - 0.5
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

    m00 = (state[CELL_TYPE, y0, x0] > 0).to(state.dtype)
    m10 = (state[CELL_TYPE, y0, x1] > 0).to(state.dtype)
    m01 = (state[CELL_TYPE, y1, x0] > 0).to(state.dtype)
    m11 = (state[CELL_TYPE, y1, x1] > 0).to(state.dtype)

    valid_weights = torch.stack(
        (w00 * m00, w10 * m10, w01 * m01, w11 * m11)
    )
    weight_sum = valid_weights.sum(dim=0)

    weighted_density = (
        w00 * m00 * state[VELOCITY_X, y0, x0]
        + w10 * m10 * state[VELOCITY_X, y0, x1]
        + w01 * m01 * state[VELOCITY_X, y1, x0]
        + w11 * m11 * state[VELOCITY_X, y1, x1]
    )

    vx = torch.where(
        weight_sum > 0,
        weighted_density / weight_sum.clamp_min(1e-8),
        torch.zeros_like(weighted_density),
    )

    #interpolate y-vel at previous position

    prev_x = pos_tensor[0,1:-1,1:-1] - timestep * vx_avg_tensor
    prev_y = (pos_middle_tensor[1,1:-1,1:-1] - timestep * state[VELOCITY_Y, 1:-1, 1:-1]) - 0.5

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

    m00 = (state[CELL_TYPE, y0, x0] > 0).to(state.dtype)
    m10 = (state[CELL_TYPE, y0, x1] > 0).to(state.dtype)
    m01 = (state[CELL_TYPE, y1, x0] > 0).to(state.dtype)
    m11 = (state[CELL_TYPE, y1, x1] > 0).to(state.dtype)

    valid_weights = torch.stack(
        (w00 * m00, w10 * m10, w01 * m01, w11 * m11)
    )
    weight_sum = valid_weights.sum(dim=0)

    weighted_density = (
        w00 * m00 * state[VELOCITY_Y, y0, x0]
        + w10 * m10 * state[VELOCITY_Y, y0, x1]
        + w01 * m01 * state[VELOCITY_Y, y1, x0]
        + w11 * m11 * state[VELOCITY_Y, y1, x1]
    )

    vy = torch.where(
        weight_sum > 0,
        weighted_density / weight_sum.clamp_min(1e-8),
        torch.zeros_like(weighted_density),
    )

    state[VELOCITY_X, 1:-1, 1:-1] = vx * state[CELL_TYPE, 1:-1, 1:-1]
    state[VELOCITY_Y, 1:-1, 1:-1] = vy * state[CELL_TYPE, 1:-1, 1:-1]

def smoke_advection():
    global state

    vx_avg_tensor = 0.5 * (state[VELOCITY_X, 1:-1, 1:-1] + state[VELOCITY_X, 1:-1, 2:])
    vy_avg_tensor = 0.5 * (state[VELOCITY_Y, 1:-1, 1:-1] + state[VELOCITY_Y, 2:, 1:-1])

    prev_x = pos_tensor[0, 1:-1, 1:-1] - timestep * vx_avg_tensor
    prev_y = pos_tensor[1, 1:-1, 1:-1] - timestep * vy_avg_tensor

    prev_x = prev_x.clamp(1.0, x_cells - 2.0)
    prev_y = prev_y.clamp(1.0, y_cells - 2.0)

    x0 = prev_x.floor().long()
    x1 = x0 + 1
    xfrac = prev_x - x0.float()

    y0 = prev_y.floor().long()
    y1 = y0 + 1
    yfrac = prev_y - y0.float()

    w00 = (1 - xfrac) * (1 - yfrac)
    w10 = xfrac * (1 - yfrac)
    w01 = (1 - xfrac) * yfrac
    w11 = xfrac * yfrac

    m00 = (state[CELL_TYPE, y0, x0] > 0).to(state.dtype)
    m10 = (state[CELL_TYPE, y0, x1] > 0).to(state.dtype)
    m01 = (state[CELL_TYPE, y1, x0] > 0).to(state.dtype)
    m11 = (state[CELL_TYPE, y1, x1] > 0).to(state.dtype)

    valid_weights = torch.stack(
        (w00 * m00, w10 * m10, w01 * m01, w11 * m11)
    )
    weight_sum = valid_weights.sum(dim=0)

    weighted_density = (
        w00 * m00 * state[DENSITY, y0, x0]
        + w10 * m10 * state[DENSITY, y0, x1]
        + w01 * m01 * state[DENSITY, y1, x0]
        + w11 * m11 * state[DENSITY, y1, x1]
    )

    density = torch.where(
        weight_sum > 0,
        weighted_density / weight_sum.clamp_min(1e-8),
        torch.zeros_like(weighted_density),
    )

    state[DENSITY, 1:-1, 1:-1] = density * state[CELL_TYPE, 1:-1, 1:-1]
   
def add_force():

    global states

def add_flow():
    global flow

    #apply flow to state
    #only apply where flow is nonzero

    state[VELOCITY_X] = torch.where(flow[VELOCITY_X] != 0, flow[VELOCITY_X], state[VELOCITY_X])
    state[VELOCITY_Y] = torch.where(flow[VELOCITY_Y] != 0, flow[VELOCITY_Y], state[VELOCITY_Y])
    state[DENSITY] = torch.where(flow[DENSITY] != 0, flow[DENSITY], state[DENSITY])
    state[PRESSURE] = torch.where(flow[3] != 0, flow[3], state[3])

# draw simulation

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Fluid Sim")

clock = pygame.time.Clock()

#---GPU rendering setup---

#context binding
ctx = moderngl.create_context()

#create texture
texture = ctx.texture((x_cells, y_cells), 3, dtype='f1') # 3 channels (RGB), unsigned byte
texture.filter = (moderngl.NEAREST, moderngl.NEAREST) # Chunky upscaling

#setup vertex array object
vao = drawhelper.setup_vao(ctx)

#nice draw functions

running = True
prev_mouse_pos = None
viewmode = 0

while running:

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pass
            if event.key == pygame.K_1:
                viewmode = 0
            if event.key == pygame.K_2:
                viewmode = 1
            if event.key == pygame.K_3:
                viewmode = 2
            if event.key == pygame.K_4:
                viewmode = 3



    mouse_pressed = pygame.mouse.get_pressed()
    if mouse_pressed[0]:
        mx,my = pygame.mouse.get_pos()
        gridmx, gridmy = int(mx * x_cells / SCREEN_WIDTH), int(my * y_cells / SCREEN_HEIGHT)

        if 0 <= gridmx < x_cells-1 and 0 <= gridmy < y_cells-1:
            
            state[DENSITY, gridmy-3:gridmy+4, gridmx-3:gridmx+4] += 1 # Add density in a 4x4 area around mouse

            if prev_mouse_pos:
                pmx, pmy = prev_mouse_pos
                gridpmx, gridpmy = int(pmx * x_cells / SCREEN_WIDTH), int(pmy * y_cells / SCREEN_HEIGHT)

                state[0, gridmy-3:gridmy+4, gridmx-3:gridmx+4] += (mx - pmx) * 2.0 #add velocity when dragging w/mouse
                state[1, gridmy-3:gridmy+4, gridmx-3:gridmx+4] += (my - pmy) * 2.0 #add velocity when dragging w/mouse
        prev_mouse_pos = (mx, my)
    else:
        prev_mouse_pos = None


    update()

    
    if viewmode == 0:
        rgb_tensor = drawhelper.drawdensityfield(state, y_cells, x_cells, device)
    elif viewmode == 1:
        rgb_tensor = drawhelper.drawspeedfield(state, y_cells, x_cells, device)
    elif viewmode == 2:
        rgb_tensor = drawhelper.drawpressurefield(state, y_cells, x_cells, device)
    elif viewmode == 3:
        rgb_tensor = drawhelper.drawdivergencefield(state, y_cells, x_cells, device)
    else:
        rgb_tensor = drawhelper.drawdensityfield(state, y_cells, x_cells, device)

   
    
    #write textude to GPU memory
    texture.write(rgb_tensor.contiguous().cpu().numpy())
    
    #render
    ctx.clear(0, 0, 0)
    texture.use(0)

    #debug_texture.use(0)
    vao.render(moderngl.TRIANGLE_STRIP)
    
    pygame.display.flip()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    total_frame_ms = (perf_counter() - start) * 1000
    
    pygame.display.set_caption(
        f"GPU Fluid Sim | FPS: {clock.get_fps():.1f} | "
        f"Pressure solve: {pressure_solve_ms:.2f} ms | "
        f"Advection solve: {advect_solve_ms:.2f} ms | "
        f"Total: {total_frame_ms:.2f} ms"
    )
    clock.tick(FPS)

pygame.quit()
