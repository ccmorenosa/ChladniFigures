"""Make the data sample for the Chladni Figures simulation."""
import argparse
from multiprocessing import Pool, cpu_count

import numpy as np
from sympy import lambdify, sympify

from sources import sources_functions
from wave_equation_PDE import System

# Create the parser for the console arguments.
parser = argparse.ArgumentParser(
    prog="simulator.py",
    description="Chladni Figures Simulator.",
    formatter_class=argparse.RawTextHelpFormatter
)

# Add parser for the system parameters.
system_parser = parser.add_argument_group("System Parameters")

# Add argument for the damping term in the wave equation.
system_parser.add_argument(
    "-g",  "--gamma", type=float, default=100,
    help="Set the damping factor for the system. Default: 100."
)

# Add argument for the speed of sound in the material.
system_parser.add_argument(
    "-s",  "--speed_sound", type=float, default=246.667,
    help="Set the speed of sound in the plate. Default: 246.667."
)

# Add argument for the source function.
system_parser.add_argument(
    "-so",  "--source_exp", type=str, default=["sine"], nargs="+",
    help="""Set the source functions. It must be a list of the following
options, 'sine', 'square', 'triangular', 'sawtooth' or an expression in terms
of t and w like 'sin(w*t*t + t)'. source_exp and frequency_exp must be the same
size. Default: ['sine']."""
)

# Add parser for the simulation parameters.
simulation_parser = parser.add_argument_group("Simulation Parameters")

# Add argument output video.
simulation_parser.add_argument(
    "-ov",  "--output_video", type=str,
    help="""Set the path to the output video file with the animation of the
simulation."""
)

# Add argument for the theta parameter.
simulation_parser.add_argument(
    "-t",  "--theta", type=float, default=0.5,
    help="Set the theta parameter of the simulation. Default: 0.5."
)

# Parse arguments.
args = parser.parse_args()

# Create list for the simulation threads
threads = []

# Create the pool of processes.
pool = Pool(processes=int(cpu_count()))

for w in range(300, 500, 5):
    # Get source function.
    source_w_funcs = []
    w_funcs = []

    for src_exp in args.source_exp:

        if src_exp in sources_functions:
            source_w_funcs.append(sources_functions[src_exp])

        else:
            source_w_funcs.append(
                lambdify(["t", "w"], sympify(src_exp), "numpy")
            )

    w_funcs.append(
        lambdify(["t"], sympify(f"{w}"), "numpy")
    )

    def source_func(t):
        """Force the system with the given function."""
        src_val = 0
        for source_w_func, w_func in zip(source_w_funcs, w_funcs):
            src_val += source_w_func(t * 2 * np.pi, w_func(t * 2 * np.pi))

        return src_val

    # Create the system object.
    chladni_figures_system = System(
        args.speed_sound, args.gamma, source_func,
        1/w * 100, 1/w / 100, args.theta, w_funcs=w_funcs
    )

    # Add video output if any.
    if args.output_video is not None:
        chladni_figures_system.create_video_output(args.output_video)

    # Set the initial conditions of the system.
    chladni_figures_system.set_initial_conditions()

    # Add solve thread of the differential equations.
    threads.append(pool.apply_async(chladni_figures_system))

# Collect threads.
for thread in threads:
    thread.get(timeout=-1)
