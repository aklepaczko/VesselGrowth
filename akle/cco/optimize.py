import numpy as np
from sympy import Point3D, Segment3D

from akle.cco import constants
from akle.cco.vessel import radius_from_pressure_drop, Vessel


def _pressure_drop(flow, length, radius):
    return 8 * flow * constants.BLOOD_VISCOSITY_PASCAL_SEC * length / (np.pi * (radius ** 4))


def get_nearest_vessel_to_point(terminal: Point3D, vessels: list[Vessel]) -> int:
    index = 0
    min_distance = 1e10
    for vessel in vessels:
        vessel_segment = Segment3D(vessel.inlet, vessel.outlet)
        distance = vessel_segment.distance(terminal)
        if distance < min_distance:
            min_distance = distance
            index = vessel.vessel_id
    return index


def scale_radii_and_update_pressures_down_subtree(top_vessel: Vessel, scaling_factor: float):
    p_out = constants.PRESSURE_OUTLETS_PASCAL
    if top_vessel.is_parent:
        scale_radii_and_update_pressures_down_subtree(top_vessel.son, scaling_factor)
        scale_radii_and_update_pressures_down_subtree(top_vessel.daughter, scaling_factor)
        p_out = top_vessel.son.pressure_in
        top_vessel.pressure_out = p_out

    r0 = top_vessel.radius * scaling_factor
    top_vessel.radius = r0
    f0 = top_vessel.flow
    l0 = top_vessel.length
    p_in = p_out + _pressure_drop(f0, l0, r0)
    top_vessel.pressure_in = p_in


def optimize_subtree(top_vessel: Vessel, vessels: list[Vessel]):

    if not top_vessel.is_parent:
        top_vessel.radius = constants.MINIMUM_RADIUS_MM
        top_vessel.pressure_out = constants.PRESSURE_OUTLETS_PASCAL
    else:
        if (not top_vessel.son.is_parent) and (not top_vessel.daughter.is_parent):
            r1 = top_vessel.son.radius
            r2 = top_vessel.daughter.radius
            f1 = top_vessel.son.flow
            f2 = top_vessel.daughter.flow
            p1_out = top_vessel.son.pressure_out
            p2_out = top_vessel.daughter.pressure_out
            l1 = top_vessel.son.length
            l2 = top_vessel.daughter.length

            pb1 = p1_out + _pressure_drop(f1, l1, r1)
            pb2 = p2_out + _pressure_drop(f2, l2, r2)

            pb_out = pb1
            if pb1 > pb2:
                r2 = radius_from_pressure_drop(f2, l2, pb_out - p2_out)
            elif pb1 < pb2:
                pb_out = pb2
                r1 = radius_from_pressure_drop(f1, l1, pb_out - p1_out)
            top_vessel.son.radius = r1
            top_vessel.daughter.radius = r2
            top_vessel.son.pressure_in = pb_out
            top_vessel.daughter.pressure_in = pb_out

            top_vessel.pressure_out = pb_out
            r0 = Vessel.radius_from_bifurcation_law(top_vessel,
                                                    top_vessel.son,
                                                    top_vessel.daughter,
                                                    constants.BIFURCATION_LAW_POWER)
            f0 = top_vessel.flow
            l0 = top_vessel.length
            pb_in = pb_out + _pressure_drop(f0, l0, r0)
            top_vessel.pressure_in = pb_in
            top_vessel.radius = r0
        else:
            if top_vessel.son.is_parent:
                optimize_subtree(top_vessel.son, vessels)
            if top_vessel.daughter.is_parent:
                optimize_subtree(top_vessel.daughter, vessels)

            pb1 = top_vessel.son.pressure_in
            pb2 = top_vessel.daughter.pressure_in

            pb_out = pb1
            if pb1 > pb2:
                factor = ((pb2 - constants.PRESSURE_OUTLETS_PASCAL) /
                          (pb_out - constants.PRESSURE_OUTLETS_PASCAL)) ** 0.25
                scale_radii_and_update_pressures_down_subtree(top_vessel.daughter, factor)
            elif pb1 < pb2:
                pb_out = pb2
                factor = ((pb1 - constants.PRESSURE_OUTLETS_PASCAL) /
                          (pb_out - constants.PRESSURE_OUTLETS_PASCAL)) ** 0.25
                scale_radii_and_update_pressures_down_subtree(top_vessel.son, factor)

            r0 = Vessel.radius_from_bifurcation_law(top_vessel,
                                                    top_vessel.son,
                                                    top_vessel.daughter,
                                                    constants.BIFURCATION_LAW_POWER)
            top_vessel.radius = r0
            top_vessel.pressure_out = pb_out

            f0 = top_vessel.flow
            l0 = top_vessel.length
            pb_in = pb_out + _pressure_drop(f0, l0, r0)
            top_vessel.pressure_in = pb_in
