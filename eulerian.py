import pygame
from pygame.locals import *
import torch
import moderngl
import torch.nn.functional as F
import numpy as np



# read config file
# for now, hardcode variables
FPS = 60

x_cells = 400
y_cells = 300

diffusion_factor = 0.25
advection_factor = 1
#configure torch

if(torch.cuda.is_available()):
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# density tensor (for compressible fluid) - implement later

#random vel-density field for now

# 3 dims -> velx,vely, density

state = torch.randn((3, y_cells, x_cells), device=device) # (n-dims, H, W) configuration for pytorch functions




# execute simulation (use GPU)





#useful functions
def vel_magnitude(vel):
    return torch.sqrt(vel[..., 0]**2 + vel[..., 1]**2)


def update():
    diffusion()

    advection()



def diffusion():

    global state

    #basic diffusion using a kernel

    diffusion_kernel = torch.tensor([[0.0, diffusion_factor, 0.0],
                                    [diffusion_factor, 0.0, diffusion_factor],
                                    [0.0, diffusion_factor, 0.0]], device=device)

    # Apply diffusion kernel to the density field
    state[2] = F.conv2d(state[2].unsqueeze(0), diffusion_kernel.unsqueeze(0).unsqueeze(0), padding=1).squeeze(0)

def advection():
    global state

    #move density along the velocity field
    

    

# draw simulation

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700

pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Fluid Sim")

#framerate
clock = pygame.time.Clock()
FPS = 60

#---GPU rendering setup---

#context binding
ctx = moderngl.create_context()

#create texture
texture = ctx.texture((x_cells, y_cells), 3, dtype='f1') # 3 channels (RGB), unsigned byte
texture.filter = (moderngl.LINEAR, moderngl.LINEAR) # Smooth upscaling

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

vao = ctx.vertex_array(quad_program, [(vertices, '2f 2f', 'in_vert', 'in_texcoord')])


#nice draw functions


def drawvelfield():
    # velocity field 
    display_vel = torch.clamp(torch.sqrt(state[0] ** 2 + state[1] ** 2) * 15, 0, 255).to(torch.uint8) # velocity magnitude, somewhat normalized

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 1] = display_vel // 2  # Green
    rgb_tensor[..., 2] = display_vel      # Blue

    return rgb_tensor
    

def drawdensityfield():
    display_density = torch.clamp(state[2] * 15, 0, 255).to(torch.uint8) # density magnitude, somewhat normalized

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 0] = display_density      # Red
    rgb_tensor[..., 1] = display_density // 2  # Green
    rgb_tensor[..., 2] = display_density // 4  # Blue

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
            
            state[2, gridmy-3:gridmy+4, gridmx-3:gridmx+4] += 5.0 # Add density in a 4x4 area around mouse

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
    vao.render(moderngl.TRIANGLE_STRIP)
    
    pygame.display.flip()
    clock.tick(FPS)
    
    pygame.display.set_caption(f"GPU Fluid Sim | FPS: {clock.get_fpsCustom():.1f}" if hasattr(clock, 'get_fpsCustom') else f"GPU Fluid Sim | FPS: {clock.get_fps():.1f}")
    clock.tick(FPS)

pygame.quit()
