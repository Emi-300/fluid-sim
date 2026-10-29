import pygame
from pygame.locals import *
import torch
import moderngl
import torch.nn.functional as F
import numpy as np
from PIL import Image
from time import perf_counter



# read config file
# for now, hardcode variables
FPS = 30

timestep=1.0/FPS

x_cells = 200
y_cells = 150

diffusion_steps = 25
diffusion_overrelaxation = 1.8
advection_factor = 1

density = 3

#benchmarking
pressure_solve_ms = 0.0

#configure torch

if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# dims: velx,vely, density, cell type, pressure

VELOCITY_X = 0
VELOCITY_Y = 1
DENSITY = 2
CELL_TYPE = 3
PRESSURE = 4

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
# dims: velx,vely, density

flow = torch.zeros((3, y_cells, x_cells), device=device)


#inflow from left

# flow[0, :, 0] = 1  # set velocity in x-direction at the left boundary
# flow[1, :, 0] = 0  # set velocity in y-direction at the left boundary
# flow[2, :, 0] = 1  # set density at the left boundary


# DEBUG - assign velocity based on cell coordinates
# state[0] = torch.arange(x_cells, device=device).unsqueeze(0).repeat(y_cells, 1)
# state[1] = torch.arange(y_cells, device=device).unsqueeze(1).repeat(1, x_cells)


# DEBUG 2 - assign random velocity

state[0] = torch.randn((y_cells, x_cells), device=device)
state[1] = torch.randn((y_cells, x_cells), device=device)

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
    global state

    #add_flow()

    #add_force()

    diffusion()

    velocity_advection()

    smoke_advection()


def diffusion():

    global state, pressure_solve_ms

    #diffusion gives weird checkerboard pattern -> fix later
    
    divergence_tensor = (state[0, 1:-1, 2:] * state[CELL_TYPE, 1:-1, 1:-1]
                        - state[0, 1:-1, 1:-1] * state[CELL_TYPE, 1:-1, :-2]
                        + state[1, 2:, 1:-1] * state[CELL_TYPE, 1:-1, 1:-1]
                        - state[1, 1:-1, 1:-1] * state[CELL_TYPE, :-2, 1:-1])

    kernel_matrix = torch.tensor(         [[0.0, 1, 0.0],
                                              [1, 0.0, 1],
                                              [0.0, 1, 0.0]], device=device)
        
    kernel = kernel_matrix.view(1, 1, 3, 3)

    neighbors_matrix = F.conv2d(state[CELL_TYPE].unsqueeze(0).unsqueeze(0), kernel).squeeze()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()

    for _ in range(diffusion_steps):

       findpressure(divergence_tensor, neighbors_matrix,red_cells,black_cells)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pressure_solve_ms = (perf_counter() - start) * 1000
    
    updatevelocity()

def findpressure(divergence_tensor, neighbors_matrix, red_cells, black_cells):

    global state

    #gauss-seidel iteration
    #potential optimizations - multigrid solver
    #                        - store edge data within tensor                   

    #pressure equation: average neighbor pressure - (density)(width / timestep) * (divergence)
    #update red cells

    average_pressure_tensor =  (state[PRESSURE, 1:-1, :-2] + state[PRESSURE, 1:-1, 2:] + state[PRESSURE, :-2, 1:-1] + state[PRESSURE, 2:, 1:-1]) * state[CELL_TYPE, 1:-1, 1:-1] # objects should have 0 pressuree

    new_pressure_tensor = (average_pressure_tensor[red_cells] - (density * cell_width / timestep) * divergence_tensor[red_cells]) / neighbors_matrix[red_cells]
    state[PRESSURE, 1:-1, 1:-1][red_cells] += diffusion_overrelaxation * (new_pressure_tensor - state[PRESSURE, 1:-1, 1:-1][red_cells])

    #update black cells

    average_pressure_tensor = (state[PRESSURE, 1:-1, :-2] + state[PRESSURE, 1:-1, 2:] + state[PRESSURE, :-2, 1:-1] + state[PRESSURE, 2:, 1:-1]) * state[CELL_TYPE, 1:-1, 1:-1] # objects should have 0 pressure

    new_pressure_tensor = (average_pressure_tensor[black_cells] - (density * cell_width / timestep) * divergence_tensor[black_cells]) / neighbors_matrix[black_cells]
    state[PRESSURE, 1:-1, 1:-1][black_cells] += diffusion_overrelaxation * (new_pressure_tensor - state[PRESSURE, 1:-1, 1:-1][black_cells])

def updatevelocity():

    #don't update non-fluid cells

    state[VELOCITY_X, 1:-1, 1:-1] -= (timestep / (density * cell_width)) * (state[PRESSURE, 1:-1, 1:-1] - state[PRESSURE, 1:-1, :-2]) * state[CELL_TYPE,1:-1,1:-1] * state[CELL_TYPE,1:-1,:-2] #making sure velocity is not coming from or going to solid cell
    state[VELOCITY_Y, 1:-1, 1:-1] -= (timestep / (density * cell_width)) * (state[PRESSURE, 1:-1, 1:-1] - state[PRESSURE, :-2, 1:-1]) * state[CELL_TYPE,1:-1,1:-1] * state[CELL_TYPE,:-2,1:-1]


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

    #interpolate x-vel at previous position

    x0 = prev_x.floor().long()
    x1 = x0 + 1
    xfrac = prev_x - x0.float()

    y0 = prev_y.floor().long()
    y1 = y0 + 1
    yfrac = prev_y - y0.float()

    vx00 = state[VELOCITY_X, y0, x0]
    vx10 = state[VELOCITY_X, y0, x1]
    vx01 = state[VELOCITY_X, y1, x0]
    vx11 = state[VELOCITY_X, y1, x1]

    vx = (1 - xfrac) * (1 - yfrac) * vx00 + xfrac * (1 - yfrac) * vx10 + (1 - xfrac) * yfrac * vx01 + xfrac * yfrac * vx11

    state[VELOCITY_X, 1:-1, 1:-1] = vx * state[CELL_TYPE, 1:-1, 1:-1]


    #interpolate y-vel at previous position

    prev_x = pos_tensor[0,1:-1,1:-1] - timestep * vx_avg_tensor
    prev_y = (pos_middle_tensor[1,1:-1,1:-1] - timestep * state[VELOCITY_Y, 1:-1, 1:-1]) - 0.5

    x0 = prev_x.floor().long()
    x1 = x0 + 1
    xfrac = prev_x - x0.float()

    y0 = prev_y.floor().long()
    y1 = y0 + 1
    yfrac = prev_y - y0.float()

    vy00 = state[VELOCITY_Y, y0, x0]
    vy10 = state[VELOCITY_Y, y0, x1]
    vy01 = state[VELOCITY_Y, y1, x0]
    vy11 = state[VELOCITY_Y, y1, x1]

    vy = (1 - xfrac) * (1 - yfrac) * vy00 + xfrac * (1 - yfrac) * vy10 + (1 - xfrac) * yfrac * vy01 + xfrac * yfrac * vy11

    state[VELOCITY_Y, 1:-1, 1:-1] = vy * state[CELL_TYPE, 1:-1, 1:-1]
  
def smoke_advection():
    global state

    vx_avg_tensor = 0.5 * (state[VELOCITY_X, 1:-1, 1:-1] + state[VELOCITY_X, 1:-1, 2:])
    vy_avg_tensor = 0.5 * (state[VELOCITY_Y, 1:-1, 1:-1] + state[VELOCITY_Y, 2:, 1:-1])

    prev_x = pos_tensor[0, 1:-1, 1:-1] - timestep * vx_avg_tensor
    prev_y = pos_tensor[1, 1:-1, 1:-1] - timestep * vy_avg_tensor

    x0 = prev_x.floor().long()
    x1 = x0 + 1
    xfrac = prev_x - x0.float()

    y0 = prev_y.floor().long()
    y1 = y0 + 1
    yfrac = prev_y - y0.float()

    density00 = state[DENSITY, y0, x0]
    density10 = state[DENSITY, y0, x1]
    density01 = state[DENSITY, y1, x0]
    density11 = state[DENSITY, y1, x1]

    density = (1 - xfrac) * (1 - yfrac) * density00 + xfrac * (1 - yfrac) * density10 + (1 - xfrac) * yfrac * density01 + xfrac * yfrac * density11

    state[DENSITY, 1:-1, 1:-1] = density * state[CELL_TYPE, 1:-1, 1:-1]

     
def add_force():
    #just gravity for now

    global states

    state[1, 1:-1, 1:-1] += 0.1 * state[CELL_TYPE, :-2, 1:-1]  # apply gravity in the y-direction

def add_flow():
    global flow
    #apply flow to state

    state[0] += flow[0] * timestep
    state[1] += flow[1] * timestep
    state[2] += flow[2] * timestep

# draw simulation

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Fluid Sim")

#framerate
clock = pygame.time.Clock()

#Set max fps - FPS = 60

#---GPU rendering setup---

#context binding
ctx = moderngl.create_context()

#create texture
texture = ctx.texture((x_cells, y_cells), 3, dtype='f1') # 3 channels (RGB), unsigned byte
texture.filter = (moderngl.NEAREST, moderngl.NEAREST) # Smooth upscaling

# Setup simple texture rendering program (Quad shader)
quad_program = ctx.program(
    vertex_shader='''
        #version 330
        in vec2 in_vert;
        in vec2 in_texcoord;
        out vec2 v_texcoord;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
            v_texcoord = in_texcoord;
        }
    ''',
    fragment_shader='''
        #version 330
        uniform sampler2D Texture;
        in vec2 v_texcoord;
        out vec4 f_color;
        void main() {
            f_color = texture(Texture, v_texcoord);
        }
    '''
)



# Render vertices mapping to screen boundaries
vertices = ctx.buffer(np.array([
    -1.0,  1.0,   0.0, 0.0, # Top Left
    -1.0, -1.0,   0.0, 1.0, # Bottom Left
     1.0,  1.0,   1.0, 0.0, # Top Right
     1.0, -1.0,   1.0, 1.0, # Bottom Right
], dtype=np.float32))

#setup vertex array object
vao = ctx.vertex_array(quad_program, [(vertices, '2f 2f', 'in_vert', 'in_texcoord')])

#nice draw functions


def drawvelfield():
    # velocity field 
    display_vel_x = torch.clamp(state[0] * 15, 0, 255).to(torch.uint8) # velocity magnitude, somewhat normalized
    display_vel_y = torch.clamp(state[1] * 15, 0, 255).to(torch.uint8) # velocity magnitude, somewhat normalized
    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 0] = display_vel_x  # Red
    rgb_tensor[..., 1] = display_vel_y  # Green

    return rgb_tensor

def drawspeedfield():
    display_speed = torch.sqrt(state[VELOCITY_X]**2 + state[VELOCITY_Y]**2)
    display_speed = torch.clamp(display_speed * 15, 0, 255).to(torch.uint8) # speed magnitude, somewhat normalized

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 0] = display_speed  # Red
    rgb_tensor[..., 1] = display_speed  # Green
    rgb_tensor[..., 2] = display_speed * 0.5  # Blue

    return rgb_tensor

def drawdensityfield():
    display_density = torch.clamp(state[2] * 255, 0, 255).to(torch.uint8) # density magnitude, somewhat normalized

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 0] = display_density   # Red
    rgb_tensor[..., 1] = display_density   # Green
    rgb_tensor[..., 2] = display_density   # Blue

    return rgb_tensor

def drawpressurefield():

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    rgb_tensor[1:-1, 1:-1, 0] = torch.clamp(-state[4, 1:-1, 1:-1] *10, 0, 255).to(torch.uint8) # Red - negative divergence
    rgb_tensor[1:-1, 1:-1, 2] = torch.clamp(state[4, 1:-1, 1:-1] *10, 0, 255).to(torch.uint8)  # Blue - positive divergence

    return rgb_tensor

def drawuvfield():

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    #x in red, y in green

    rgb_tensor[..., 0] = torch.clamp(state[0] *(255/x_cells), 0, 255).to(torch.uint8)  # Red
    rgb_tensor[..., 1] = torch.clamp(state[1] *(255/y_cells), 0, 255).to(torch.uint8)  # Green

    return rgb_tensor

def drawdivergencefield():
    divergence = (state[0, 1:-1, 2:] - state[0, 1:-1, 1:-1] + state[1, 2:, 1:-1] - state[1, 1:-1, 1:-1] )

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    rgb_tensor[1:-1, 1:-1, 0] = torch.clamp(-divergence*100, 0, 255).to(torch.uint8) # Red - negative divergence
    rgb_tensor[1:-1, 1:-1, 2] = torch.clamp(divergence*100, 0, 255).to(torch.uint8)  # Blue - positive divergence

    return rgb_tensor

def drawsolidcells():

    kernel_matrix = torch.tensor([[0.0, 1.0, 0.0],
                                          [1.0, 0.0, 1.0],
                                          [0.0, 1.0, 0.0]], device=device)
    
    kernel = kernel_matrix.view(1, 1, 3, 3)
    
    #padding is already built in
    #in diffusion function to allow for dynamic movement of obstacles
    
    solid_neighbors = F.conv2d(state[3].unsqueeze(0).unsqueeze(0), kernel).squeeze()

    # Solid cells are represented by a specific color, e.g., white
    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    rgb_tensor[1:-1, 1:-1, 0] = 85 * solid_neighbors // 3 # Red
    rgb_tensor[1:-1, 1:-1, 1] = 85 * solid_neighbors // 2  # Green
    rgb_tensor[1:-1, 1:-1, 2] = 85 * solid_neighbors  # Blue

    return rgb_tensor

running = True
prev_mouse_pos = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                pass



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

    rgb_tensor = drawdensityfield()

    #write textude to GPU memory
    texture.write(rgb_tensor.contiguous().cpu().numpy())
    
    #render
    ctx.clear(0, 0, 0)
    texture.use(0)

    #debug_texture.use(0)
    vao.render(moderngl.TRIANGLE_STRIP)
    
    pygame.display.flip()
    
    pygame.display.set_caption(
        f"GPU Fluid Sim | FPS: {clock.get_fps():.1f} | "
        f"Pressure solve: {pressure_solve_ms:.2f} ms"
    )
    clock.tick(FPS)

pygame.quit()
