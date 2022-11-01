"""Base script to solve the forced wave equation."""
import cv2
import numpy as np
import pyvista
import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import FunctionSpace
from mpi4py import MPI
from petsc4py import PETSc
from ufl import dx


class System(object):
    """
    Define all the system parameters and equation for the Chladni Figures.

    Parameters
    ----------
    c: Float.
        Speed of sound in the material in m/s.

    gamma: Float.
        Damping parameter.

    source: Callable.
        Source function.

    t_max: Float.
        Maximum time of the simulation in s.

    theta: Float.
        Parameter for forward/backward discretization.

    domain_side: Int.
        Size of the side in the domain mesh.

    """

    # Initial time.
    t_0 = 0

    # Time dependent output.
    xdmf = None

    # Video output.
    video = None

    def __init__(
        self, c, gamma, source,
        t_max=1, theta=0.5,
        domain_side=400
    ):
        """Construct the object."""
        # Define source parameters.
        self.source = source

        # Define system parameters.
        self.c = c
        self.c2 = c**2
        self.gamma = gamma

        self.theta_1 = theta
        self.theta_2 = 1 - theta

        # Time evolution parameters.
        self.t_max = t_max
        self.dt = 1 / 1000000
        self.n_steps = self.t_max // self.dt

        # Create the domain.
        self.domain_side = domain_side

        self.domain = mesh.create_unit_square(
            MPI.COMM_WORLD,
            self.domain_side, self.domain_side,
            mesh.CellType.quadrilateral
        )

        # Location of the center of the plate.
        self.mid_pos = [
            1/2 - 1/(domain_side),
            1/2 + 1/(domain_side)
        ]

        # Create the function space.
        self.V = FunctionSpace(self.domain, ("CG", 1))
        self.index_order = self.V.mesh.geometry.input_global_indices

        # Initial conditions.
        self.u_n = fem.Function(self.V)
        self.u_n.name = "u_n"

        self.v_n = fem.Function(self.V)
        self.v_n.name = "v_n"

        # Solution variable.
        self.u_h = fem.Function(self.V)
        self.u_h.name = "u_h"

        self.v_h = fem.Function(self.V)
        self.v_h.name = "v_h"

        # Source function.
        self.f = fem.Function(self.V)  # f(t_n).
        self.f.name = "f"

        self.f_n = fem.Function(self.V)  # f(t_{n-1}).
        self.f_n.name = "f_n"

        # TODO: Implement this boundary condition later.
        # Defining the boundary conditions or all the components.
        x = ufl.SpatialCoordinate(self.domain)
        self.g = 0 * x[1]

        # Test and trial functions.
        self.u = ufl.TrialFunction(self.V)
        self.v = ufl.TrialFunction(self.V)
        self.w = ufl.TestFunction(self.V)

        # Define solvers.
        # Amplitud mesh.
        self.a_u = self.u * self.w * dx
        if self.gamma > 0:
            self.a_u += (
                self.gamma * self.dt * self.theta_1 * self.u * self.w * dx
            )
        self.a_u += (
            self.theta_1**2 * self.dt**2 * self.c2 *
            ufl.dot(ufl.grad(self.u), ufl.grad(self.w)) * dx
        )

        bf_u = fem.form(self.a_u)

        self.A_u = fem.petsc.assemble_matrix(bf_u)
        self.A_u.assemble()

        self.solver_u = PETSc.KSP().create(self.domain.comm)
        self.solver_u.setOperators(self.A_u)
        self.solver_u.setType(PETSc.KSP.Type.PREONLY)
        self.solver_u.getPC().setType(PETSc.PC.Type.LU)

        self.a_v = self.v * self.w * dx
        if self.gamma > 0:
            self.a_v += (
                self.gamma * self.dt * self.theta_1 * self.v * self.w * dx
            )

        bf_v = fem.form(self.a_v)

        self.A_v = fem.petsc.assemble_matrix(bf_v)
        self.A_v.assemble()

        self.solver_v = PETSc.KSP().create(self.domain.comm)
        self.solver_v.setOperators(self.A_v)
        self.solver_v.setType(PETSc.KSP.Type.PREONLY)
        self.solver_v.getPC().setType(PETSc.PC.Type.LU)

    def create_xdmf_output(self, filename):
        """
        Create an output file for the solutions in time.

        Parameters
        ----------
        filename: String.
            Name of the output file.

        """
        # Time dependent output for the solution.
        self.xdmf = io.XDMFFile(self.domain.comm, filename, "w")
        self.xdmf.write_mesh(self.domain)

    def create_video_output(self, filename, fps=20):
        """
        Create a video with the solutions.

        Parameters
        ----------
        filename: String.
            Name of the output file.

        fps: Int.
            Number of frames per second.

        """
        # Video config.
        self.video = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc("m", "p", "4", "v"),
            fps,
            (720, 720)
        )

    def eval_source(self, x, y, t):
        """
        Force the system with the given function.

        Parameters
        ----------
        x: Array.
            x component of the coordinates of the cells in the mess.

        y: Array.
            y component of the coordinates of the cells in the mess.

        t: Float.
            Time.

        """
        f_t = np.zeros(len(x))

        for xy_i in range(len(x)):
            if (
                (self.mid_pos[0] <= x[xy_i] <= self.mid_pos[1]) and
                (self.mid_pos[0] <= y[xy_i] <= self.mid_pos[1])
            ):
                f_t[xy_i] = self.source(t)

        return f_t

    def set_initial_conditions(
        self, u_0=lambda x: 0*x[0], v_0=lambda x: 0*x[0]
    ):
        """
        Set the initial condition of the system.

        Parameters
        ----------
        u_0: Array, Callable.
            Values of the amplitud mesh at t=0s. The array or the return of
            the callable must be of size [domain_side+1, domain_side+1].

        v_0: Array, Callable.
            Values of the velocity mesh at t=0s. The array or the return of
            the callable must be of size [domain_side+1, domain_side+1].

        """
        # Amplitud mesh.
        self.u_n.interpolate(u_0)
        self.u_h.interpolate(u_0)

        if self.xdmf is not None:
            # Write initial condition.
            self.xdmf.write_function(self.u_h, 0)

        # Velocities mesh.
        self.v_n.interpolate(v_0)
        self.v_h.interpolate(v_0)

    def solve(self, show_interval=None):
        """Solve the system."""
        i = 0

        for t in np.arange(self.t_0, self.t_max + self.dt, self.dt):

            # Forcing function.
            self.f.interpolate(
                lambda x: self.eval_source(x[0], x[1], t + self.dt)
            )

            self.f_n.interpolate(
                lambda x: self.eval_source(x[0], x[1], t)
            )

            L_u = self.u_n * self.w * dx
            if self.gamma > 0:
                L_u += (
                    self.gamma * self.dt * self.theta_1 *
                    self.u_n * self.w * dx
                )
            L_u -= (
                self.dt**2 * self.theta_1*self.theta_2 * self.c2 *
                ufl.dot(ufl.grad(self.u_n), ufl.grad(self.w)) * dx
            )
            L_u += self.dt * self.v_n * self.w * dx
            L_u += self.dt**2 * self.theta_1**2 * self.f * self.w * dx
            L_u += (
                self.dt**2 * self.theta_1*self.theta_2 * self.f_n * self.w * dx
            )

            lf_u = fem.form(L_u)

            b_u = fem.petsc.create_vector(lf_u)

            fem.petsc.assemble_vector(b_u, lf_u)

            self.solver_u.solve(b_u, self.u_h.vector)
            self.u_h.x.scatter_forward()

            L_v = self.v_n * self.w * dx
            if self.gamma > 0:
                L_v -= (
                    self.gamma * self.dt * self.theta_2 *
                    self.v_n * self.w * dx
                )
            L_v -= (
                self.dt * self.theta_1 * self.c2 *
                ufl.dot(ufl.grad(self.u_h), ufl.grad(self.w)) * dx
            )
            L_v -= (
                self.dt * self.theta_2 * self.c2 *
                ufl.dot(ufl.grad(self.u_n), ufl.grad(self.w)) * dx
            )
            L_v += self.dt * self.theta_1 * self.f * self.w * dx
            L_v += self.dt * self.theta_2 * self.f_n * self.w * dx

            lf_v = fem.form(L_v)

            b_v = fem.petsc.create_vector(lf_v)

            fem.petsc.assemble_vector(b_v, lf_v)

            self.solver_v.solve(b_v, self.v_h.vector)
            self.v_h.x.scatter_forward()

            if self.xdmf is not None:
                # Write initial condition.
                self.xdmf.write_function(self.u_h, t)

            if self.video is not None:
                image = np.zeros((self.domain_side+1)**2)

                max_u_h = np.max(self.u_h.x.array.real)
                min_u_h = np.min(self.u_h.x.array.real)

                for index, val in zip(self.index_order, self.u_h.x.array.real):
                    image[index] = 255 * (val - min_u_h) / (max_u_h - min_u_h)

                image = cv2.applyColorMap(
                    np.uint8(np.reshape(
                        image, (self.domain_side+1, self.domain_side+1)
                    )),
                    cv2.COLORMAP_JET
                )

                image = cv2.resize(
                    image, (720, 720), interpolation=cv2.INTER_AREA
                )

                self.video.write(image)

            self.u_n.x.array[:] = self.u_h.x.array
            self.v_n.x.array[:] = self.v_h.x.array

            if show_interval is not None and i % show_interval == 0:
                u_grid = pyvista.UnstructuredGrid(
                    *plot.create_vtk_mesh(self.V)
                )
                u_grid.point_data["u"] = self.u_h.x.array.real
                u_grid.set_active_scalars("u")
                # warped = u_grid.warp_by_scalar("u", factor=1.5)
                u_plotter = pyvista.Plotter()
                u_plotter.add_mesh(u_grid)
                # u_plotter.add_mesh(warped)
                u_plotter.view_xy()
                if not pyvista.OFF_SCREEN:
                    u_plotter.show()

            # v_grid = pyvista.UnstructuredGrid(*plot.create_vtk_mesh(V))
            # v_grid.point_data["u"] = v_h.x.array.real
            # v_grid.set_active_scalars("u")
            # warped = v_grid.warp_by_scalar("u", factor=1.5)
            # v_plotter = pyvista.Plotter()
            # v_plotter.add_mesh(warped)
            # # v_plotter.view_xy()
            # # if not pyvista.OFF_SCREEN:
            # #     v_plotter.show()

            print(f"[{i / self.n_steps * 100.:.2f}%]", end="\r")

            i += 1

        print()
        self.video.release()
        cv2.destroyAllWindows()
