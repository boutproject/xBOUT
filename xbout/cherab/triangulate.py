"""
Interface to Cherab

This module performs triangulation of BOUT++ grids, making them
suitable for input to Cherab ray-tracing analysis.

"""

import numpy as np


class TriangularData:
    """
    Represents a set of triangles with data constant on them.
    Creates a Cherab Discrete2DMesh, and can then convert
    that to a 3D (axisymmetric) emitting material.
    """

    def __init__(self, vertices, triangles, data):
        self.vertices = vertices
        self.triangles = triangles
        self.data = data

        from raysect.core.math.function.float import Discrete2DMesh

        self.mesh = Discrete2DMesh(
            self.vertices, self.triangles, self.data, limit=False, default_value=0.0
        )

    def to_emitter(
        self,
        parent=None,
        cylinder_zmin=None,
        cylinder_zmax=None,
        cylinder_rmin=None,
        cylinder_rmax=None,
        step: float = 0.01,
    ):
        """
        Returns a 3D Cherab emitter, by rotating the 2D mesh about the Z axis

        step: Volume integration step length [m]

        """
        from raysect.core import translate
        from raysect.primitive import Cylinder, Subtract
        from raysect.optical.material import VolumeTransform
        from cherab.core.math import AxisymmetricMapper
        from cherab.tools.emitters import RadiationFunction

        if cylinder_zmin is None:
            cylinder_zmin = np.amin(self.vertices[:, 1])
        if cylinder_zmax is None:
            cylinder_zmax = np.amax(self.vertices[:, 1])
        if cylinder_rmin is None:
            cylinder_rmin = np.amin(self.vertices[:, 0])
        if cylinder_rmax is None:
            cylinder_rmax = np.amax(self.vertices[:, 0])

        rad_function_3d = AxisymmetricMapper(self.mesh)

        shift = translate(0, 0, cylinder_zmin)
        emitting_material = VolumeTransform(
            RadiationFunction(rad_function_3d, step=step), shift.inverse()
        )

        # Create an annulus by removing the middle from the cylinder.
        return Subtract(
            Cylinder(cylinder_rmax, cylinder_zmax - cylinder_zmin),
            Cylinder(cylinder_rmin, cylinder_zmax - cylinder_zmin),
            transform=shift,
            parent=parent,
            material=emitting_material,
        )

    def plot_2d(self, ax=None, nr: int = 150, nz: int = 150, colorbar: bool = True):
        """
        Make a 2D plot of the data

        nr, nz - Number of samples in R and Z
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        Rmin, Zmin = np.amin(self.vertices, axis=0)
        Rmax, Zmax = np.amax(self.vertices, axis=0)

        from cherab.core.math import sample2d

        r, z, emiss_sampled = sample2d(self.mesh, (Rmin, Rmax, nr), (Zmin, Zmax, nz))

        image = ax.imshow(
            emiss_sampled.T, origin="lower", extent=(r.min(), r.max(), z.min(), z.max())
        )
        if colorbar:
            fig.colorbar(image)
        ax.set_xlabel("r")
        ax.set_ylabel("z")

        return ax

    def wall_flux(
        self,
        wall,
        pixel_samples: int = 2000,
        wall_detector_offset: float = 0.001,
        toroidal_width: float = 0.01,
        step: float = 0.01,
    ):
        """
        Calculate power onto segments of an axisymmetric wall

        Based on the Cherab manual here:
        https://www.cherab.info/demonstrations/radiation_loads/symmetric_power_load.html#symmetric-power-load

        Parameters
        ----------

        wall : AxisymmetricWall
            Defines a closed loop in R-Z, that defines an axisymmetric wall.

        pixel_samples : The number of samples per wall segment

        wall_detector_offset
            Distance of detector pixels from wall [meters].
            Slightly displaced to avoid numerical overlap (ray trapping)

        toroidal_width
            toroidal width of the detectors [meters]

        step
            Ray path step length [meters]

        """
        from raysect.core import Point3D, Vector3D, translate, rotate_basis
        from raysect.optical import World
        from raysect.optical.observer import PowerPipeline0D
        from raysect.optical.material import AbsorbingSurface
        from raysect.optical.observer.nonimaging.pixel import Pixel
        from cherab.tools.primitives import axisymmetric_mesh_from_polygon

        world = World()

        # Create a plasma emitter
        emitter = self.to_emitter(parent=world, step=step)

        # Create the walls around the plasma
        wall_polygon = wall.to_polygon()  # [npoints, 2] array

        # create a 3D mesh from the 2D polygon outline using symmetry
        wall_mesh = axisymmetric_mesh_from_polygon(wall_polygon)
        wall_mesh.parent = world
        wall_mesh.material = AbsorbingSurface()

        result = []
        for rz1, rz2 in wall:
            p1 = Point3D(rz1[0], 0, rz1[1])
            p2 = Point3D(rz2[0], 0, rz2[1])

            # evaluate y_vector
            y_vector_full = p1.vector_to(p2)

            if y_vector_full.length < 1e-5:
                continue  # Skip; points p1, p2 are the same

            y_vector = y_vector_full.normalise()
            y_width = y_vector_full.length

            # evaluate normal_vector (inward pointing)
            normal_vector = y_vector.cross(Vector3D(0, 1, 0)).normalise()

            # evaluate the central point of the detector
            # Note: Displaced in the normal direction by wall_detector_offset
            detector_center = (
                p1 + y_vector_full * 0.5 + wall_detector_offset * normal_vector
            )

            # Use the power pipeline to record total power arriving at the surface
            power_data = PowerPipeline0D()

            # Note: Displace the pixel location
            pixel_transform = translate(
                detector_center.x, detector_center.y, detector_center.z
            ) * rotate_basis(normal_vector, y_vector)

            # Use pixel_samples argument to increase amount of sampling and reduce noise
            pixel = Pixel(
                [power_data],
                x_width=toroidal_width,
                y_width=y_width,
                name=f"wall-{rz1}-{rz2}",
                spectral_bins=1,
                transform=pixel_transform,
                parent=world,
                pixel_samples=pixel_samples,
            )

            pixel_area = toroidal_width * y_width

            # Start collecting samples
            pixel.observe()

            # Estimate power density
            power_density = power_data.value.mean / pixel_area

            # For checking energy conservation.
            # Revolve this tile around the CYLINDRICAL z-axis to get total power collected by these tiles.
            # Add up all the tile contributions to get total power collected.
            detector_radius = np.sqrt(detector_center.x**2 + detector_center.y**2)
            observed_total_power = power_density * (
                y_width * 2 * np.pi * detector_radius
            )

            result.append(
                {
                    "rz1": rz1,
                    "rz2": rz2,
                    "power_density": power_density,
                    "total_power": observed_total_power,
                }
            )

        return result


class Triangulate:
    """
    Represents a set of triangles for a 2D mesh in R-Z

    """

    def __init__(self, rm, zm):
        """
        rm and zm define quadrilateral cell corners in 2D (R, Z)

        rm : [nx, ny, 4]
        zm : [nx, ny, 4]
        """
        assert zm.shape == rm.shape
        assert len(rm.shape) == 3
        nx, ny, n = rm.shape
        assert n == 4

        # Build a list of vertices and a list of triangles
        vertices = []
        triangles = []

        def vertex_index(R, Z):
            """
            Return the vertex index at given (R,Z) location.
            Note: This is inefficient linear search
            """
            # Check if there is already a vertex at this location
            for i, v in enumerate(vertices):
                vr, vz = v
                d2 = (vr - R) ** 2 + (vz - Z) ** 2
                if d2 < 1e-10:
                    return i
            # Not found so add a new vertex
            vertices.append((R, Z))
            return len(vertices) - 1

        for ix in range(nx):
            for jy in range(ny):
                # Adding cell (ix, jy)
                # Get the vertex indices of the 4 corners
                vertex_inds = [
                    vertex_index(rm[ix, jy, n], zm[ix, jy, n]) for n in range(4)
                ]
                # Choose corners so triangles have the same sign
                triangles.append(vertex_inds[0:3])  # Corners 1,2,3
                triangles.append(vertex_inds[:0:-1])  # Corners 4,3,2

        self.vertices = np.array(vertices)
        self.triangles = np.array(triangles)
        self.cell_number = np.arange(nx * ny).reshape((nx, ny))

    def __repr__(self):
        return "<xbout Cherab triangulation>"

    def plot_triangles(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        rs = self.vertices[self.triangles, 0]
        zs = self.vertices[self.triangles, 1]

        # Close the triangles
        rs = np.concatenate((rs, rs[:, 0:1]), axis=1)
        zs = np.concatenate((zs, zs[:, 0:1]), axis=1)

        ax.plot(rs.T, zs.T, "k")

        return ax

    def with_data(self, da):
        """
        Returns a new object containing vertices, triangles, and data

        Parameters
        ----------

        da : xarray.DataArray
            Expected to have 'cherab_grid' attribute
            and 'cell_number' coordinate.
            Should only have 'x' and 'theta' dimensions.

        Returns
        -------

        A TriangularData object
        """

        if "cherab_grid" not in da.attrs:
            raise ValueError("DataArray missing cherab_grid attribute")

        if "cell_number" not in da.coords:
            raise ValueError("DataArray missing cell_number coordinate")

        da = da.squeeze()  # Drop dimensions of size 1

        # Check that extra dimensions (e.g time) have been dropped
        # so that the data has the same dimensions as cell_number
        if da.sizes != da.coords["cell_number"].sizes:
            raise ValueError(
                f"Data and cell_number coordinate have "
                f"different sizes ({da.sizes} and "
                f"{da.coords['cell_number'].sizes})"
            )

        if 2 * da.size == self.triangles.shape[0]:
            # Data has not been sliced, so the size matches
            # the number of triangles

            # Note: Two triangles per quad, so repeat data twice
            return TriangularData(
                self.vertices, self.triangles, da.data.flatten().repeat(2)
            )

        # Data size and number of triangles don't match.
        # Use cell_number to work out which triangles to keep

        cells = da.coords["cell_number"].data.flatten()
        triangles = np.concatenate(
            (self.triangles[cells * 2, :], self.triangles[cells * 2 + 1, :])
        )

        data = np.tile(da.data.flatten(), 2)

        return TriangularData(self.vertices, triangles, data)
