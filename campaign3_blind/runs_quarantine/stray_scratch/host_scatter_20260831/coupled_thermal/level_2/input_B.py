"""
Kratos Multiphysics input for coupled thermal problem - Subdomain B
Level 2, mesh 14x16
Thermal conductivity k = 200.0
"""

from KratosMultiphysics import *
from KratosMultiphysics.HeatTransferApplication import *

model = Model()
mp = model.CreateModelPart("Main")

thermal_conductivity = 200.0

solver_settings = Parameters("""{
    "model_part_name": "Main",
    "custom_solver_settings": {
        "linear_solver": {
            "solver_type": "umfpack",
            "tolerance": 1e-10,
            "max_iteration": 1000
        },
        "convergence_criterion": {
            "type": "residual",
            "convergence_tolerance": 1e-10,
            "maximum_number_of_iterations": 1000
        }
    }
}""")

heat_solver = HeatConductionSolver(solver_settings)
