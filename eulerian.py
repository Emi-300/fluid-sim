import pygame
from pygame.locals import *
import torch
import moderngl
import torch.nn.functional as F
import numpy as np
from PIL import Image
from time import perf_counter

import diffusion, advection, drawhelper
import state_config as stype

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
            state[stype.CELL_TYPE, y, x] = 0



# DEBUG - assign velocity based on cell coordinates
# state[0] = torch.arange(x_cells, device=device).unsqueeze(0).repeat(y_cells, 1)
# state[1] = torch.arange(y_cells, device=device).unsqueeze(1).repeat(1, x_cells)


# DEBUG 2 - assign random velocity

# state[0] = torch.randn((y_cells, x_cells), device=device)
# state[1] = torch.randn((y_cells, x_cells), device=device)

#set border grid velocity to zero

state[stype.VELOCITY_X, 0, :] = 0
state[stype.VELOCITY_X, -1, :] = 0
state[stype.VELOCITY_X, :, 0] = 0
state[stype.VELOCITY_X, :, 1] = 0
state[stype.VELOCITY_X, :, -1] = 0

state[stype.VELOCITY_Y, 0, :] = 0
state[stype.VELOCITY_Y, 1, :] = 0
state[stype.VELOCITY_Y, -1, :] = 0
state[stype.VELOCITY_Y, :, 0] = 0
state[stype.VELOCITY_Y, :, -1] = 0


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

    advection.velocity_advection(state, pos_tensor, pos_middle_tensor, x_cells, y_cells, timestep)

    advection.smoke_advection(state, pos_tensor, pos_middle_tensor, x_cells, y_cells, timestep)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    advect_solve_ms = (perf_counter() - start) * 1000


#TODO consolidate interpolation into its own function
#     break up backtrace into multiple smaller steps and stop at first solid obstacle

def add_force():

    global states

def add_flow():
    global flow

    #apply flow to state
    #only apply where flow is nonzero

    state[stype.VELOCITY_X] = torch.where(flow[0] != 0, flow[0], state[stype.VELOCITY_X])
    state[stype.VELOCITY_Y] = torch.where(flow[1] != 0, flow[1], state[stype.VELOCITY_Y])
    state[stype.DENSITY] = torch.where(flow[2] != 0, flow[2], state[stype.DENSITY])
    state[stype.PRESSURE] = torch.where(flow[3] != 0, flow[3], state[stype.PRESSURE])

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
            
            state[stype.DENSITY, gridmy-3:gridmy+4, gridmx-3:gridmx+4] += 1 # Add density in a 4x4 area around mouse

            if prev_mouse_pos:
                pmx, pmy = prev_mouse_pos
                gridpmx, gridpmy = int(pmx * x_cells / SCREEN_WIDTH), int(pmy * y_cells / SCREEN_HEIGHT)

                state[stype.VELOCITY_X, gridmy-3:gridmy+4, gridmx-3:gridmx+4] += (mx - pmx) * 2.0 #add velocity when dragging w/mouse
                state[stype.VELOCITY_Y, gridmy-3:gridmy+4, gridmx-3:gridmx+4] += (my - pmy) * 2.0 #add velocity when dragging w/mouse
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
