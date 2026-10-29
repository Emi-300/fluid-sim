import moderngl
import numpy as np
import state_config as stype
import torch

def setup_vao(ctx):
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

    return vao


def drawvelocityfield(state,y_cells,x_cells,device):
    # velocity field 
    display_vel_x = torch.clamp(state[stype.VELOCITY_X] * 15, 0, 255).to(torch.uint8) # velocity magnitude, somewhat normalized
    display_vel_y = torch.clamp(state[stype.VELOCITY_Y] * 15, 0, 255).to(torch.uint8) # velocity magnitude, somewhat normalized
    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 0] = display_vel_x  # Red
    rgb_tensor[..., 1] = display_vel_y  # Green

    return rgb_tensor

def drawspeedfield(state,y_cells,x_cells,device):
    display_speed = torch.sqrt(state[stype.VELOCITY_X]**2 + state[stype.VELOCITY_Y]**2)
    display_speed = torch.clamp(display_speed * 15, 0, 255).to(torch.uint8) # speed magnitude, somewhat normalized

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 0] = display_speed  # Red
    rgb_tensor[..., 1] = display_speed  # Green
    rgb_tensor[..., 2] = display_speed * 0.5  # Blue

    return rgb_tensor

def drawdensityfield(state,y_cells,x_cells,device):
    display_density = torch.clamp(state[stype.DENSITY] * 255, 0, 255).to(torch.uint8) # density magnitude, somewhat normalized

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)
    rgb_tensor[..., 0] = display_density   # Red
    rgb_tensor[..., 1] = display_density   # Green
    rgb_tensor[..., 2] = display_density   # Blue

    return rgb_tensor

def drawpressurefield(state,y_cells,x_cells,device):

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    rgb_tensor[1:-1, 1:-1, 0] = torch.clamp(-state[stype.PRESSURE, 1:-1, 1:-1] *10, 0, 255).to(torch.uint8) # Red - negative divergence
    rgb_tensor[1:-1, 1:-1, 2] = torch.clamp(state[stype.PRESSURE, 1:-1, 1:-1] *10, 0, 255).to(torch.uint8)  # Blue - positive divergence

    return rgb_tensor

def drawuvfield(state,y_cells,x_cells,device):

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    #x in red, y in green

    rgb_tensor[..., 0] = torch.clamp(state[stype.VELOCITY_X] *(255/x_cells), 0, 255).to(torch.uint8)  # Red
    rgb_tensor[..., 1] = torch.clamp(state[stype.VELOCITY_Y] *(255/y_cells), 0, 255).to(torch.uint8)  # Green

    return rgb_tensor

def drawdivergencefield(state,y_cells,x_cells,device):
    divergence = (state[stype.VELOCITY_X, 1:-1, 2:] - state[stype.VELOCITY_X, 1:-1, 1:-1] + state[stype.VELOCITY_Y, 2:, 1:-1] - state[stype.VELOCITY_Y, 1:-1, 1:-1] )

    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    rgb_tensor[1:-1, 1:-1, 0] = torch.clamp(-divergence*100, 0, 255).to(torch.uint8) # Red - negative divergence
    rgb_tensor[1:-1, 1:-1, 2] = torch.clamp(divergence*100, 0, 255).to(torch.uint8)  # Blue - positive divergence

    return rgb_tensor

def drawsolidcells(state,y_cells,x_cells,device):

    kernel_matrix = torch.tensor([[0.0, 1.0, 0.0],
                                          [1.0, 0.0, 1.0],
                                          [0.0, 1.0, 0.0]], device=device)
    
    kernel = kernel_matrix.view(1, 1, 3, 3)
    
    #padding is already built in
    #in diffusion function to allow for dynamic movement of obstacles
    
    fluid_neighbors = state[stype.CELL_TYPE, 1:-1, :-2] + state[stype.CELL_TYPE, 1:-1, 2:] + state[stype.CELL_TYPE, :-2, 1:-1] + state[stype.CELL_TYPE, 2:, 1:-1]

    # Solid cells are represented by a specific color, e.g., white
    rgb_tensor = torch.zeros((y_cells, x_cells, 3), dtype=torch.uint8, device=device)

    rgb_tensor[1:-1, 1:-1, 0] = 85 * fluid_neighbors // 3 # Red
    rgb_tensor[1:-1, 1:-1, 1] = 85 * fluid_neighbors // 2  # Green
    rgb_tensor[1:-1, 1:-1, 2] = 85 * fluid_neighbors  # Blue

    return rgb_tensor
