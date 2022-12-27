import numpy as np
import scipy.optimize as optim

from akle.cco import constants
from akle.cco.geometry import Point3D, Segment3D
from akle.cco.vessel import pressure_drop_on_segment, radius_from_pressure_drop, Vessel


class Bifurcation:
    def __init__(self, bifurcating_vessel: Vessel, new_terminal_point: Point3D):
        f1 = bifurcating_vessel.flow
        f2 = constants.TERMINAL_FLOW_MM3_PER_SEC
        f0 = f1 + f2

        x0 = bifurcating_vessel.inlet.x
        x1 = bifurcating_vessel.outlet.x
        x2 = new_terminal_point.x
        x_init = (f0 * x0 + f1 * x1 + f2 * x2) / (2 * f0)

        y0 = bifurcating_vessel.inlet.y
        y1 = bifurcating_vessel.outlet.y
        y2 = new_terminal_point.y
        y_init = (f0 * y0 + f1 * y1 + f2 * y2) / (2 * f0)

        z0 = bifurcating_vessel.inlet.z
        z1 = bifurcating_vessel.outlet.z
        z2 = new_terminal_point.z
        z_init = (f0 * z0 + f1 * z1 + f2 * z2) / (2 * f0)

        temp_bifurcation_point = Point3D(x_init, y_init, z_init)

        l1 = Segment3D(bifurcating_vessel.outlet, temp_bifurcation_point).length

        pb_in = bifurcating_vessel.pressure_out
        bifurcation_pressure = pb_in + pressure_drop_on_segment(f1, l1, bifurcating_vessel.radius)

        if bifurcating_vessel.has_parent:
            self.parent = Vessel(inlet=bifurcating_vessel.inlet,
                                 outlet=temp_bifurcation_point,
                                 flow=f0,
                                 pressure_in=bifurcating_vessel.parent.pressure_out,
                                 pressure_out=bifurcation_pressure,
                                 parent=bifurcating_vessel.parent)
        else:
            self.parent = Vessel(inlet=bifurcating_vessel.inlet,
                                 outlet=temp_bifurcation_point,
                                 flow=f0,
                                 pressure_in=bifurcating_vessel.pressure_in,
                                 pressure_out=bifurcation_pressure)

        self.son = Vessel(inlet=temp_bifurcation_point,
                          outlet=bifurcating_vessel.outlet,
                          flow=f1,
                          pressure_in=self.parent.pressure_out,
                          pressure_out=bifurcating_vessel.pressure_out,
                          parent=self.parent)

        self.daughter = Vessel(inlet=temp_bifurcation_point,
                               outlet=new_terminal_point,
                               flow=f2,
                               pressure_in=self.parent.pressure_out,
                               pressure_out=constants.PRESSURE_OUTLETS_PASCAL,
                               parent=self.parent)

        self.parent.set_children(self.son, self.daughter)

        l2 = Segment3D(self.daughter.inlet, self.daughter.outlet).length
        r2 = radius_from_pressure_drop(f2, l2, bifurcation_pressure - constants.PRESSURE_OUTLETS_PASCAL)
        r0 = Vessel.radius_from_bifurcation_law(parent=self.parent,
                                                son=self.son,
                                                daughter=self.daughter,
                                                gamma=constants.BIFURCATION_LAW_POWER)
        self.daughter.radius = r2
        self.parent.radius = r0

        self.init_volume = bifurcating_vessel.get_volume()

    def __del__(self):
        self.parent.delete_vessel(True)

    def bifurcation_volume(self) -> float:
        return self.parent.get_volume() + self.son.get_volume() + self.daughter.get_volume()

    def optimize_bifurcation(self, num_iterations: int):

        def _objective(x):
            _l0 = np.linalg.norm(self.parent.inlet.coordinates - x)
            _l1 = np.linalg.norm(self.son.outlet.coordinates - x)
            _l2 = np.linalg.norm(self.daughter.outlet.coordinates - x)

            _bifurcation_pressure = self.son.pressure_out + pressure_drop_on_segment(f1, _l1, r1)

            _r2 = radius_from_pressure_drop(f2, _l2, _bifurcation_pressure - constants.PRESSURE_OUTLETS_PASCAL)

            _r0 = Vessel.radius_from_bifurcation_law(parent=self.parent,
                                                     son=self.son,
                                                     daughter=self.daughter,
                                                     gamma=constants.BIFURCATION_LAW_POWER)

            volume = _r0 ** 2 * _l0 + r1 ** 2 * _l1 + _r2 ** 2 * _l2
            return volume

        def _length_constraint(x):
            _l0 = np.linalg.norm(self.parent.inlet.coordinates - x)
            _l1 = np.linalg.norm(self.son.outlet.coordinates - x)
            _l2 = np.linalg.norm(self.daughter.outlet.coordinates - x)

            _r0 = self.parent.radius
            _r1 = self.son.radius
            _r2 = self.daughter.radius

            return [_l0 - 2 * _r0, _l1 - 2 * _r1, _l2 - 2 * _r2]

        f1 = self.son.flow
        f2 = self.daughter.flow
        r1 = self.son.radius

        opts = {'maxiter': num_iterations,
                'disp': True,
                'verbose': True}

        non_linear = optim.NonlinearConstraint(_length_constraint, 0, np.inf)

        x0 = self.son.inlet.coordinates
        res = optim.minimize(_objective,
                             x0,
                             method='trust-constr',
                             constraints=[non_linear],
                             options=opts)

        self.parent.outlet = Point3D(*res.x)
        self.son.inlet = Point3D(*res.x)
        self.daughter.inlet = Point3D(*res.x)

        l1 = self.son.length
        l2 = self.daughter.length

        bifurcation_pressure = self.son.pressure_out + pressure_drop_on_segment(f1, l1, r1)
        self.parent.pressure_out = bifurcation_pressure
        self.son.pressure_in = bifurcation_pressure
        self.daughter.pressure_in = bifurcation_pressure

        r2 = radius_from_pressure_drop(f2, l2, bifurcation_pressure - constants.PRESSURE_OUTLETS_PASCAL)
        self.daughter.radius = r2

        r0 = Vessel.radius_from_bifurcation_law(parent=self.parent,
                                                son=self.son,
                                                daughter=self.daughter,
                                                gamma=constants.BIFURCATION_LAW_POWER)
        self.parent.radius = r0
