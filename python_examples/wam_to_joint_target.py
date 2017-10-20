import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--interactive", action="store_true")
args = parser.parse_args()

import numpy as np
import openravepy
import trajoptpy
import json
import time

env = openravepy.Environment()
env.StopSimulation()
#env.Load("/home/clemens/openrave/src/data/wam_table.env.xml")
env.Load("robots/barrett-wam-sensors.zae")
env.Load("../data/wam_table.xml")

trajoptpy.SetInteractive(args.interactive) # pause every iteration, until you press 'p'. Press escape to disable further plotting
robot = env.GetRobots()[0]
manip = robot.GetManipulator('arm')

tool_link = robot.GetLink("wam7")
local_dir = np.array([0.,0.,-1.])

arm_inds = manip.GetArmIndices()
arm_joints = [robot.GetJointFromDOFIndex(ind) for ind in arm_inds]

n_steps = 10

def f(x):
    robot.SetDOFValues(x, arm_inds, False)
    return tool_link.GetTransform()[:2,:3].dot(local_dir)
def dfdx(x):
    robot.SetDOFValues(x, arm_inds, False)
    world_dir = tool_link.GetTransform()[:3,:3].dot(local_dir)
    return np.array([np.cross(joint.GetAxis(), world_dir)[:2] for joint in arm_joints]).T.copy()

# right of table
joint_start = [-0.7426929473876954,
 0.9810707553927964,
 9.993918917428071e-08,
 0.940855696469757,
 -2.5474904852273994e-08,
 1.1647957091619077,
 0.6405251067082487]


joint_start = [0, 0, 0, 0, 0, 0, 0]
# center of table
joint_start = [0.0,
0.6129659054107124,
2.3516749954164678e-14,
1.8880799777139607,
-4.529709940470639e-14,
0.6626403331756591,
-6.899305807069695e-14
]

robot.SetDOFValues(joint_start, robot.GetManipulator('arm').GetArmIndices())

joint_target = [0.5606344327915153,
 1.5434823489215235,
 0.2512632460376474,
 0.8100638201543314,
 0.7383471727371216,
 -1.0713351608267458,
 0.0]

# left of table
joint_target = [0.7426929473876954,
 0.9810707553927964,
 9.993918917428071e-08,
 0.940855696469757,
 -2.5474904852273994e-08,
 1.1647957091619077,
 2.1405251067082487]

# close to robot
joint_target = [0.6634513146923048,
 0.3952704370021819,
 -2.7154692075134847e-07,
 2.766953458330111,
 1.3545046195773125e-08,
 -0.08810024360243851,
 0.6697384715080266]

joint_target = [1.590324536736133,
 0.781675696372986,
 -1.4456497699103223,
 2.7669535005048798,
 -4.799655442984406,
 0.825472593307495,
 -1.2556250095367436]

request = {
  "basic_info" : {
    "n_steps" : n_steps,
    "manip" : "arm", # see below for valid values
    "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
  },
  "costs" : [
  {
    "type" : "joint_vel", # joint-space velocity cost
    "params": {"coeffs" : [0.1]} # a list of length one is automatically expanded to a list of length n_dofs
    # also valid: [1.9, 2, 3, 4, 5, 5, 4, 3, 2, 1]
  },
  {
    "type" : "pose",
    "params" : {
          "pos_coeffs" : [0, 0, 1],
          "rot_coeffs" : [1, 1, 1],
          "xyz" : [0.27290561, 0.49326863, -0.00960189],
          "wxyz" : [0, 0, 1, 0],
          "link" : "Finger0-0",
          "timestep" : n_steps/2
    },
  }
#  {
#    "type" : "collision",
#    "params" : {
#      "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
#      "dist_pen" : [0.025] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
#    },    
#  }
  ],
  "constraints" : [
  {
    "type" : "joint", # joint-space target
    "params" : {"vals" : joint_target } # length of vals = # dofs of manip
  }
  ],
  "init_info" : {
      "type" : "straight_line", # straight line in joint space.
      "endpoint" : joint_target
  }
}
s = json.dumps(request) # convert dictionary into json-formatted string
prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem


# BEGIN add_costs
if False:
    for t in xrange(1,n_steps):    
        #if args.diffmethod == "numerical":
        #    prob.AddErrorCost(f, [(t,j) for j in xrange(7)], "ABS", "up%i"%t)
        #elif args.diffmethod == "analytic":
        if True:
            prob.AddErrorCost(f, dfdx, [(t,j) for j in xrange(7)], "ABS", "up%i"%t)

t_start = time.time()
result = trajoptpy.OptimizeProblem(prob) # do optimization
t_elapsed = time.time() - t_start
print result
print "optimization took %.3f seconds"%t_elapsed

from trajoptpy.check_traj import traj_is_safe
prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
assert traj_is_safe(result.GetTraj(), robot) # Check that trajectory is collision free
