#! /bin/python3
"""Main file of the project."""
import argparse

from wave_equation_PDE.ChladniFigures import System
import numpy as np
from sympy import lambdify, sympify

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
    "-s",  "--speed_sound", type=float, default=5000,
    help="Set the speed of sound in the plate. Default: 5000."
)

# Add argument for the source function.
system_parser.add_argument(
    "-so",  "--source_exp", type=str, default="sin(w * t)",
    help="Set the source function. Default: 'sin(w * t)."
)

# Add argument for the source function.
system_parser.add_argument(
    "-w",  "--frequency_exp", type=str, default="350",
    help="Set the frequency function. Default: '350'"
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

# Add argument for the maximum time of simulation.
simulation_parser.add_argument(
    "-tma",  "--t_max", type=float, default=1.,
    help="Set the maximum time in seconds for the simulation. Default: 1."
)

# Parse arguments.
args = parser.parse_args()

# Get source function.
source_exp = sympify(args.source_exp)
source_w_func = lambdify(["t", "w"], source_exp, "numpy")

# Get frequency function.
w_exp = sympify(args.frequency_exp)
w_func = lambdify(["w"], w_exp, "numpy")


def source_func(t):
    """Force the system with the given function."""
    return source_w_func(t * 2 * np.pi, w_func(t * 2 * np.pi))


# Create the system object.
chladni_figures_system = System(
    args.speed_sound, args.gamma, source_func, args.t_max, args.theta
)

# Add video output if any.
if args.output_video is not None:
    chladni_figures_system.create_video_output(args.output_video)

# Set the initial conditions of the system.
chladni_figures_system.set_initial_conditions()

# Solve the differential equations.
chladni_figures_system.solve()
