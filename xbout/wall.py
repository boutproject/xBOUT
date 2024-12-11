"""
Routines to read and represent wall geometries
"""

import numpy as np

class AxisymmetricWall:
    def __init__(self, Rs, Zs):
        """
        Defines a 2D (R,Z) axisymmetric wall

        Parameters
        ----------

        Rs : list or 1D array
            Major radius coordinates [meters]
        Zs : list or 1D array
            Vertical coordinates [meters]

        """
        if len(Rs) != len(Zs):
            raise ValueError("Rs and Zs arrays have different lengths")

        # Ensure that the members are numpy arrays
        self.Rs = np.array(Rs)
        self.Zs = np.array(Zs)

    def __iter__(self):
        """
        Iterate over wall elements

        Each iteration returns a pair of (R,Z) pairs:
            ((R1, Z1), (R2, Z2))

        These pairs define wall segment.
        """
        return iter(zip(zip(self.Rs, self.Zs),
                        zip(np.roll(self.Rs, -1), np.roll(self.Zs, -1))))

    def to_polygon(self):
        """
        Returns a 2D Numpy array [npoints, 2]
        Index 0 is major radius (R) in meters
        Index 1 is height (Z) in meters
        """
        return np.stack((self.Rs, self.Zs), axis=-1)

    def plot(self, linestyle='k-', ax = None):
        """
        Plot the wall on given axis. If no axis
        is given then a new figure is created.

        Returns
        -------

        The matplotlib axis containing the plot
        """

        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.Rs, self.Zs, linestyle)
        return ax

def read_geqdsk(filehandle):
    """
    Read wall geometry from a GEQDSK file.

    Note: Requires the freeqdsk package
    """

    if isinstance(filehandle, str):
        with open(filehandle, "r") as f:
            return read_geqdsk(f)

    from freeqdsk import geqdsk

    data = geqdsk.read(filehandle)
    # rlim and zlim should be 1D arrays of wall coordinates

    if not (hasattr(data, "rlim") and hasattr(data, "zlim")):
        raise ValueError(f"Wall coordinates not found in GEQDSK file")

    return AxisymmetricWall(data["rlim"], data["zlim"])


def read_csv(filehandle, delimiter=','):
    """
    Parameters
    ----------

    filehandle: File handle
        Must contain two columns, for R and Z coordinates [meters]
    
    delimier : character
        A single character that separates fields

    Notes:
    - Uses the python `csv` module 
    """
    import csv
    reader = csv.reader(filehandle, delimiter=delimiter)
    Rs = []
    Zs = []
    for row in reader:
        if len(row) == 0:
            continue # Skip empty rows
        if len(row) != 2:
            raise ValueError(f"CSV row should contain two columns: {row}")
        Rs.append(float(row[0]))
        Zs.append(float(row[1]))

    return AxisymmetricWall(Rs, Zs)
        
