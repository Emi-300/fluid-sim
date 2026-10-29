We assume incompressible fluid (makes math easier)
 - Reasonable assumption for free gas

Assume inviscous for now (might change later)

Need velocity field v & density p
can have pressure field p
 - collocated vs staggered grid

Grid Setup

$ ~~~~~~~~~~~~~v_{i,j}        \\$
$~~~~~~~~~--\downarrow -- ~~\\$
$~~~~~~~~~| ~~~~~~~~~~~~~~~~ |\\$
$u_{i,j} \rightarrow ~~~~~~~~~~~~~~~~ \rightarrow u_{i+1,j}     \\$
$~~~~~~~~~| ~~~~~~~~~~~~~~~~ |\\$
$~~~~~~~~~--\downarrow -- ~~\\$
$ ~~~~~~~~~~~~~v_{i,j+1}        \\$

$\\ p_{i,j} = 0$

3 steps
 - external forces
 -> For all $i,j : v_{i,j} \leftarrow v_{i,j} + \Delta t \cdot F_{tot,y}$
 -> For all $i,j : u_{i,j} \leftarrow u_{i,j} + \Delta t \cdot F_{tot,x}$
  
 - projection (make fluid incompressible)
   - Need to use divergence: $d \leftarrow u_{i+1,j} - u_{i,j} + v_{i,j+1} - v_{i,j} $  
   - $d = 0$ for incompressible fluids
   - need to push all velocities by the same amount
     - $u_{i,j} \leftarrow u_{i,j} + \frac{d}{4}$ 
     - $u_{i+1,j} \leftarrow u_{i+1,j} - \frac{d}{4}$
     - $v_{i,j} \leftarrow v_{i,j} + \frac{d}{4}$
     - $v_{i,j+1} \leftarrow v_{i,j+1} - \frac{d}{4}$
   - for walls, the velocity touching the wall cannot be modified $ \rightarrow \frac{d}{4}$ must change to $\frac{d}{3}$

   - General case - gauss-seidel method (iterate over n times to get a good result): 
    $ \\ d \leftarrow u_{i+1,j} - u_{i,j} + v_{i,j+1} - v_{i,j} \\$ 
    $s \leftarrow s_{i+1,j} - s_{i-1,j} + s_{i,j+1} - s_{i,j-1} \\$
    $\\$
    $u_{i,j} \leftarrow u_{i,j} + \frac{d s_{i-1,j}}{s} \\$ 
    $u_{i+1,j} \leftarrow u_{i+1,j} -  \frac{d s_{i+1,j}}{s} \\$
    $v_{i,j} \leftarrow v_{i,j} + \frac{d s_{i,j-1}}{s} \\$
    $v_{i,j+1} \leftarrow v_{i,j+1} - \frac{d s_{i,j+1}}{s} \\$

        $p_{i,j} \leftarrow p_{i,j} + \frac{d}{s} \cdot \frac{\rho h}{\Delta t} \\$ - $\rho$ = density, $h$ = grid spacing (gives additional info, not necessary for sim)
        
    - Need to add border cells to make sure we don't access cells outside the grid, or have them copy neighbor values

    This method is simple but slow (needs to iterate multiple times to converge) -> use overrelaxation
    $\\~~~~~~~d \leftarrow o(u_{i+1,j} - u_{i,j} + v_{i,j+1} - v_{i,j})$ where o is a scalar, $1<o<2$
    this decreases convergence time dramatically, while pressure values remain correct



 - advection (move fluid with velocity field)
    - need to move velocity values inside the grid
    - method: semi-lagrangian advection
        - see which "particle" moved to u location
        - then set new velocity $u_{t+\Delta t}$ to the velocity $u_t$ at previous pos
    
    to do this,
        - compute vector $v$ at position $x$ where $u$ is stored
        - then, previous location can be approximated as $x - \Delta t \cdot v$ (assuming a straight path, thus induces viscosity)
        - can use vorticity confinement to reduce this effect

    - do the same for density
        - store density value at center of each cell
        - advect it like velocity components





References used: 
How to write an Eulerian fluid simulator with 200 lines of code - Ten Minute Physics - https://www.youtube.com/watch?v=iKAVRgIrUOU
Coding Adventure: Simulating Smoke - Sebastian Lague - https://www.youtube.com/watch?v=Q78wvrQ9xsU&t=1409s
Fluid Simulation: Siggraph 2007 Course Notes - Robert Bridson - https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf
