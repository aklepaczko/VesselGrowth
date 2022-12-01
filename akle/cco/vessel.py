from __future__ import annotations
from typing import Optional

import numpy as np
from sympy import Point3D, Segment3D

from akle.cco import constants


def pressure_drop_on_segment(flow, length, radius):
    return 8 * flow * constants.BLOOD_VISCOSITY_PASCAL_SEC * length / (np.pi * (radius ** 4))


def radius_from_pressure_drop(flow, length, pressure_drop):
    nominator = 8 * flow * constants.BLOOD_VISCOSITY_PASCAL_SEC * length
    denominator = np.pi * pressure_drop
    return (nominator / denominator) ** 0.25


class Vessel:

    def __init__(self,
                 inlet: Point3D,
                 outlet: Point3D,
                 flow: float,
                 pressure_in: float,
                 pressure_out: float,
                 parent: Optional[Vessel] = None):
        self._inlet = inlet
        self._outlet = outlet
        self.length = Segment3D(inlet, outlet).length
        self.flow = flow
        self.pressure_in = pressure_in
        self.pressure_out = pressure_out
        self.parent = parent
        self.son = None
        self.daughter = None
        self.is_parent = False
        self.has_parent = False
        self.radius = radius_from_pressure_drop(flow=self.flow,
                                                length=self.length,
                                                pressure_drop=pressure_in - pressure_out)

    @property
    def inlet(self):
        return self._inlet

    @inlet.setter
    def inlet(self, val):
        self._inlet = val
        self.length = Segment3D(self._inlet, self._outlet).length

    @property
    def outlet(self):
        return self._outlet

    @outlet.setter
    def outlet(self, val):
        self._outlet = val
        self.length = Segment3D(self._inlet, self._outlet).length

    def clear_parent(self):
        self.parent = None
        self.has_parent = False

    def delete_vessel(self, with_subtree: bool):
        self.clear_parent()
        if self.is_parent and with_subtree:
            self.son.delete_vessel(with_subtree)
            self.daughter.delete_vessel(with_subtree)

    def get_volume(self):
        return np.pi * (self.radius ** 2) * self.length

    def set_children(self, son: Vessel, daughter: Vessel):
        self.is_parent = True
        self.son = son
        self.daughter = daughter
        son.parent = self
        daughter.parent = self

    def accumulate_flow(self):
        if self.is_parent:
            self.son.accumulate_flow()
            self.daughter.accumulate_flow()
            self.flow = self.son.flow + self.daughter.flow

    @staticmethod
    def radius_from_bifurcation_law(parent: Vessel, son: Vessel, daughter: Vessel, gamma: float):
        f0 = parent.flow
        f1 = son.flow
        f2 = daughter.flow
        r1 = son.radius
        r2 = daughter.radius
        aux_sum = f0 / f1 * (r1 ** gamma) + f0 / f2 * (r2 ** gamma)
        r0 = aux_sum ** (1 / gamma)
        return r0
