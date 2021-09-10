import sys
sys.path.append("../OpenSeesPy_Building_Modeler")

import numpy as np
import modeler
import solver
import time
import pickle
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# obtain displacements through static analysis #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

with open('tmp/building.pcl', 'rb') as f:
    b = pickle.load(f)


parent_nodes = b.list_of_parent_nodes()

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

# linear_gravity_analysis.deformed_shape(extrude_frames=True)

ux0_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][0]
ux0_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][0]
ux0_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][0]


parent_nodes[0].load = np.array([1419.3, 0.00, 0.00, 0.00, 0.00, 0.00]) * 1e3 * 0.10
parent_nodes[1].load = np.array([1415.3, 0.00, 0.00, 0.00, 0.00, 0.00]) * 1e3 * 0.10
parent_nodes[2].load = np.array([2544.2, 0.00, 0.00, 0.00, 0.00, 0.00]) * 1e3 * 0.10

linear_gravity_analysis = solver.LinearGravityAnalysis(b)
linear_gravity_analysis.run()

ux_1 = linear_gravity_analysis.node_displacements[parent_nodes[0].uniq_id][0][0]
ux_2 = linear_gravity_analysis.node_displacements[parent_nodes[1].uniq_id][0][0]
ux_3 = linear_gravity_analysis.node_displacements[parent_nodes[2].uniq_id][0][0]

dux_1 = ux_1 - ux0_1
dux_2 = ux_2 - ux0_2
dux_3 = ux_3 - ux0_3

print('Static displacements')
print(dux_1, dux_2, dux_3)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# obtain displacements through dynamic analysis #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

#
# generate nlth gm files
#

dt = 0.005
num_points = int(10.00 / dt)
ag_vals = np.full(num_points, 0.10)
gm_directory = 'ground_motions/test_case/parsed'
np.savetxt(gm_directory+'/'+'dynamic_test.txt', ag_vals)


#
# run nlth analysis
#

with open('tmp/building.pcl', 'rb') as f:
    b = pickle.load(f)

parent_nodes = b.list_of_parent_nodes()
    
nlth = solver.NLTHAnalysis(b)
# nlth.plot_ground_motion(gm_directory + '/' + 'dynamic_test.txt', 0.005)

nlth.run(10.00, 0.15,
         gm_directory + '/' + 'dynamic_test.txt',
         None,
         None,
         0.005)


n_steps = len(nlth.node_displacements[parent_nodes[0].uniq_id])
t_values = np.linspace(0.00, 5.00, n_steps)
u1 = []
for i in range(n_steps):
    u1.append(nlth.node_displacements[parent_nodes[0].uniq_id][i][0])
u1 = np.array(u1)
u2 = []
for i in range(n_steps):
    u2.append(nlth.node_displacements[parent_nodes[1].uniq_id][i][0])
u2 = np.array(u2)
u3 = []
for i in range(n_steps):
    u3.append(nlth.node_displacements[parent_nodes[2].uniq_id][i][0])
u3 = np.array(u3)

print(-u1[-1],  -u2[-1], -u3[-1])

#
# show the response in a figure
#

plt.figure(figsize=(6, 6))
plt.plot(t_values, -u1, label='lvl 1')
plt.plot(t_values, -u2, label='lvl 2')
plt.plot(t_values, -u3, label='lvl 3')
plt.grid()
plt.legend()
plt.xlabel('Time [$s$]')
plt.ylabel('Displacement [$in$]')
plt.show()
# plt.savefig('tmp/dynamic_test.pdf')
plt.close()
