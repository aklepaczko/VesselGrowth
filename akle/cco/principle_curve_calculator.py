import numpy as np
from rpy2 import robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri


class PrincipalCurveCalculator:
    def __init__(self):
        utils = rpackages.importr('utils')
        utils.chooseCRANmirror(ind=1)
        if not rpackages.isinstalled('princurve'):
            utils.install_packages('princurve')
        rpackages.importr('princurve', robject_translations={'lines.principal.curve': 'lines_principal_curve2',
                                                             'plot.principal.curve': 'plot_principal_curve2',
                                                             'points.principal.curve': 'points_principal_curve2'})
        numpy2ri.activate()

    @staticmethod
    def fit_curve(points_xyz: np.ndarray) -> np.ndarray:
        curve = robjects.r.principal_curve(points_xyz, thresh=1000, maxit=3)
        return np.asarray(curve.rx2('s').rx(curve.rx2('ord'), True)).copy()
