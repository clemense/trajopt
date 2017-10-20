import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--interactive", action="store_true")
parser.add_argument("--position_only", action="store_true")
args = parser.parse_args()

import openravepy
import trajoptpy
import json
import numpy as np
import trajoptpy.kin_utils as ku

env = openravepy.Environment()
env.StopSimulation()
env.Load("robots/barrett-wam-sensors.zae")
env.Load("../data/wam_table.xml")

trajoptpy.SetInteractive(args.interactive) # pause every iteration, until you press 'p'. Press escape to disable further plotting

robot = env.GetRobots()[0]
joint_start = [-1.832, -0.332, -1.011, -1.437, -1.1  , -2.106,  3.074]
joint_start = [0.0,
 0.6129659054107124,
 2.3516749954164678e-14,
 1.8880799777139607,
 -4.529709940470639e-14,
 0.6626403331756591,
 -6.899305807069695e-14
]
robot.SetDOFValues(joint_start, robot.GetManipulator('arm').GetArmIndices())

#quat_target = [1,0,0,0] # wxyz
#xyz_target = [6.51073449e-01,  -1.87673551e-01, 4.91061915e-01]
#hmat_target = openravepy.matrixFromPose( np.r_[quat_target, xyz_target] )

hmat_target = np.array([[ -9.99755947e-01,  -1.24566045e-13,  -2.20917653e-02,
          5.63729902e-01],
       [ -1.24276611e-13,   1.00000000e+00,  -1.44743435e-14,
         -5.96041415e-10],
       [  2.20917653e-02,  -1.17253212e-14,  -9.99755947e-01,
         -9.60187849e-03],
       [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
          1.00000000e+00]])

hmat_target = np.array([[-0.99985185,  0.0134869 , -0.01069479,  0.27290561],
       [ 0.01369188,  0.99971939, -0.01933049,  0.49326863],
       [ 0.01043108, -0.01947405, -0.99975595, -0.00960189],
       [ 0.        ,  0.        ,  0.        ,  1.        ]])

quat_target = openravepy.quatFromRotationMatrix(hmat_target)
xyz_target = hmat_target[:3, 3]

# BEGIN ik
manip = robot.GetManipulator("arm")
init_joint_target = ku.ik_for_link(hmat_target, manip, "wam7",
    filter_options = openravepy.IkFilterOptions.CheckEnvCollisions)
# END ik


request = {
  "basic_info" : {
    "n_steps" : 10,
    "manip" : "arm", # see below for valid values
    "start_fixed" : True # i.e., DOF values at first timestep are fixed based on current robot state
  },
  "costs" : [
  {
    "type" : "joint_vel", # joint-space velocity cost
    "params": {"coeffs" : [1]} # a list of length one is automatically expanded to a list of length n_dofs
  },
  {
    "type" : "collision",
    "name" :"cont_coll", # shorten name so printed table will be prettier
    "params" : {
      "continuous" : True,
      "coeffs" : [20], # penalty coefficients. list of length one is automatically expanded to a list of length n_timesteps
      "dist_pen" : [0.025] # robot-obstacle distance that penalty kicks in. expands to length n_timesteps
    }
  }
  ],
  "constraints" : [
  # BEGIN pose_constraint
  {
    "type" : "pose", 
    "params" : {"xyz" : xyz_target, 
                "wxyz" : quat_target, 
                "link": "wam7",
                "timestep" : 9
                }
                 
  }
  # END pose_constraint
  ],
  # BEGIN init
  "init_info" : {
      "type" : "straight_line", # straight line in joint space.
      "endpoint" : init_joint_target.tolist() # need to convert numpy array to list
  }
  # END init
}

if args.position_only: request["constraints"][0]["params"]["rot_coeffs"] = [0,0,0]

s = json.dumps(request) # convert dictionary into json-formatted string
prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
result = trajoptpy.OptimizeProblem(prob) # do optimization
print result

from trajoptpy.check_traj import traj_is_safe
prob.SetRobotActiveDOFs() # set robot DOFs to DOFs in optimization problem
assert traj_is_safe(result.GetTraj(), robot) # Check that trajectory is collision free

# Now we'll check to see that the final constraint was satisfied
robot.SetActiveDOFValues(result.GetTraj()[-1])
posevec = openravepy.poseFromMatrix(robot.GetLink("wam7").GetTransform())
quat, xyz = posevec[0:4], posevec[4:7]

quat *= np.sign(quat.dot(quat_target))
if args.position_only:
    assert (quat - quat_target).max() > 1e-3
else:
    assert (quat - quat_target).max() < 1e-3

