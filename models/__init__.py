from .models import RegressionModel, ClassificationModel
from .residual_cgp import ResCGPNet, ResCGPNet8, ResCGPNet11, ResCGPNet17
from .plain_cgp import PlainCGPNet, PlainCGPNet6, PlainCGPNet8, PlainCGPNet11, PlainCGPNet17

__all__ = [
    "RegressionModel",
    "ClassificationModel",
    "ResCGPNet",
    "ResCGPNet8",
    "ResCGPNet11",
    "ResCGPNet17",

    "PlainCGPNet",
    "PlainCGPNet6",
    "PlainCGPNet8",
    "PlainCGPNet11",
    "PlainCGPNet17",
]