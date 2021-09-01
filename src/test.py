import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import time
import pickle

with open('tmp/building.pcl', 'rb') as f:
    b = pickle.load(f)


parent_nodes = b.list_of_parent_nodes()

# parent_nodes[0].load = modeler.Load([1.00, 0.00, 0.00, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

# linear_gravity_analysis.deformed_shape(extrude_frames=True)

ux0_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][0]
ux0_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][0]
ux0_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][0]

uy0_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][1]
uy0_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][1]
uy0_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][1]


# ~~~~~~~~~~~~~~~~~~~~ #
# x direction analysis #
# ~~~~~~~~~~~~~~~~~~~~ #

parent_nodes[0].load = np.array([1000.00, 0.00, 0.00, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

ux_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][0]
ux_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][0]
ux_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][0]

ux_1_1 = ux_1 - ux0_1
ux_2_1 = ux_2 - ux0_2
ux_3_1 = ux_3 - ux0_3


parent_nodes[0].load = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
parent_nodes[1].load = np.array([1000.00, 0.00, 0.00, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

ux_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][0]
ux_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][0]
ux_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][0]

ux_1_2 = ux_1 - ux0_1
ux_2_2 = ux_2 - ux0_2
ux_3_2 = ux_3 - ux0_3


parent_nodes[1].load = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
parent_nodes[2].load = np.array([1000.00, 0.00, 0.00, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

ux_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][0]
ux_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][0]
ux_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][0]

ux_1_3 = ux_1 - ux0_1
ux_2_3 = ux_2 - ux0_2
ux_3_3 = ux_3 - ux0_3

flexibility_matrix = np.array([
    [ux_1_1, ux_1_2, ux_1_3],
    [ux_2_1, ux_2_2, ux_2_3],
    [ux_3_1, ux_3_2, ux_3_3]
])

stiffness_matrix = np.linalg.inv(flexibility_matrix)

mass_matrix = np.array([
    [1419.3, 000.00, 000.00],
    [000.00, 1415.3, 000.00],
    [000.00, 000.00, 2544.2]
]) / 386.00

from scipy.linalg import eigh

eigvals, eigvecs = eigh(stiffness_matrix,
                        mass_matrix)

print('Vibration periods in the X direction')
print(2.*np.pi/np.sqrt(eigvals))
print()


# ~~~~~~~~~~~~~~~~~~~~ #
# y direction analysis #
# ~~~~~~~~~~~~~~~~~~~~ #

parent_nodes[2].load = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
parent_nodes[0].load = np.array([0.00, 1000.00, 0.00, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

uy_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][1]
uy_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][1]
uy_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][1]

uy_1_1 = uy_1 - uy0_1
uy_2_1 = uy_2 - uy0_2
uy_3_1 = uy_3 - uy0_3


parent_nodes[0].load = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
parent_nodes[1].load = np.array([0.00, 1000.00, 0.00, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

uy_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][1]
uy_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][1]
uy_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][1]

uy_1_2 = uy_1 - uy0_1
uy_2_2 = uy_2 - uy0_2
uy_3_2 = uy_3 - uy0_3


parent_nodes[1].load = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
parent_nodes[2].load = np.array([0.00, 1000.00, 0.00, 0.00, 0.00, 0.00])

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

uy_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][1]
uy_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][1]
uy_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][1]

uy_1_3 = uy_1 - uy0_1
uy_2_3 = uy_2 - uy0_2
uy_3_3 = uy_3 - uy0_3

flexibility_matrix = np.array([
    [uy_1_1, uy_1_2, uy_1_3],
    [uy_2_1, uy_2_2, uy_2_3],
    [uy_3_1, uy_3_2, uy_3_3]
])

stiffness_matrix = np.linalg.inv(flexibility_matrix)

eigvals, eigvecs = eigh(stiffness_matrix,
                        mass_matrix)

print('Vibration periods in the Y direction')
print(2.*np.pi/np.sqrt(eigvals))
print()


