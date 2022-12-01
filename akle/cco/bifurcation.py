import numpy as np
from tqdm import tqdm

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
        current_volume = self.bifurcation_volume()
        volume_increase = current_volume - self.init_volume
        min_increase = volume_increase
        best_point = self.parent.outlet

        x0 = self.parent.inlet.x
        x1 = self.son.outlet.x
        x2 = self.daughter.outlet.x
        xs = np.array([x0, x1, x2])

        y0 = self.parent.inlet.y
        y1 = self.son.outlet.y
        y2 = self.daughter.outlet.y
        ys = np.array([y0, y1, y2])

        z0 = self.parent.inlet.z
        z1 = self.son.outlet.z
        z2 = self.daughter.outlet.z
        zs = np.array([z0, z1, z2])

        r0 = self.parent.radius
        r1 = self.son.radius
        r2 = self.daughter.radius

        l0 = self.parent.length
        l1 = self.son.length
        l2 = self.daughter.length

        f1 = self.son.flow
        f2 = self.daughter.flow

        for _ in tqdm(range(num_iterations)):
            weights = np.array([r0 ** 2 / l0, r1 ** 2 / l1, r2 ** 2 / l2])
            denominator = np.sum(weights)

            x_new = np.sum(xs * weights) / denominator
            y_new = np.sum(ys * weights) / denominator
            z_new = np.sum(zs * weights) / denominator

            new_terminal = Point3D(x_new, y_new, z_new)

            self.parent.outlet = new_terminal
            self.son.inlet = new_terminal
            self.daughter.inlet = new_terminal

            l0 = self.parent.length
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

            current_volume = self.bifurcation_volume()
            volume_increase = current_volume - self.init_volume

            if volume_increase < min_increase:
                min_increase = volume_increase
                best_point = self.parent.outlet

        self.parent.outlet = best_point
        self.son.inlet = best_point
        self.daughter.inlet = best_point

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
