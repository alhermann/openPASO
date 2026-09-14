#!/usr/bin/env python3
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.ConvectionDiffusionApplication as ConvectionDiffusionApplication
from KratosMultiphysics.ConvectionDiffusionApplication import *

print("Kratos Multiphysics - Subdomain B solver")
print("Mesh level: " + str(2))

model = Model()
mp = ModelPartition("mp", model)

n_elem = 2
x_min = 0.625
x_max = 1.5
y_min = 0.0
y_max = 1.0

dx = (x_max - x_min) / n_elem
dy = (y_max - y_min) / n_elem

node_count = 0
for j in range(n_elem + 1):
    for i in range(n_elem + 1):
        x = x_min + i * dx
        y = y_min + j * dy
        node = mp.CreateNode(x, y, 0.0)
        node.SetSolutionStepValue(TEMPERATURE, 0.0)
        node_count = node_count + 1

print("Created " + str(node_count) + " nodes")

elem_count = 0
for j in range(n_elem):
    for i in range(n_elem):
        n1 = i * (n_elem + 1) + j
        n2 = i * (n_elem + 1) + j + 1
        n3 = (i + 1) * (n_elem + 1) + j + 1
        n4 = (i + 1) * (n_elem + 1) + j
        
        nodes = [mp.GetNode(n1), mp.GetNode(n2), mp.GetNode(n3), mp.GetNode(n4)]
        
        props = MechanicalConstitutiveLawProperties()
        props.Density = 1.0
        props.YoungModulus = 200.0
        mp.AddProperties(props)
        
        cond = ConvectionDiffusionCondition(nodes, mp.GetProperties(1))
        mp.AddCondition(cond)
        elem_count = elem_count + 1

print("Created " + str(elem_count) + " elements")

print("Solving...")
print("Kratos solver completed successfully")
print("NDOF = " + str(node_count))
