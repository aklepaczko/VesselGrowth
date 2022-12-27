import numpy as np
import sympy


class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self._coordinates = np.array([x, y, z])

    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

    @coordinates.setter
    def coordinates(self, val):
        self.x = val[0]
        self.y = val[1]
        self.z = val[2]


class Segment3D:
    def __init__(self, start: Point3D, end: Point3D):
        self.start = start
        self.end = end
        self._length = np.linalg.norm(end.coordinates - start.coordinates)

    @property
    def length(self):
        return np.linalg.norm(self.end.coordinates - self.start.coordinates)

    @length.setter
    def length(self, value):
        pass

    @property
    def midpoint(self):
        center = (self.end.coordinates - self.start.coordinates) / 2
        return Point3D(center[0], center[1], center[2])

    def distance(self, p: Point3D):
        # vec_0 = self.end.coordinates - self.start.coordinates
        # vec_1 = self.start.coordinates - p.coordinates
        #
        # cross = np.cross(vec_0, vec_1)
        #
        # return np.linalg.norm(cross) / np.linalg.norm(vec_0)
        ps = sympy.Point3D(p.coordinates)
        seg = sympy.Segment3D(sympy.Point(self.end.coordinates), sympy.Point(self.start.coordinates))
        d = seg.distance(ps)
        return float(d)
