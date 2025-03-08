#!/usr/bin/env python3

from typing import List
import numpy as np

from drake_trajopt.all import DrakeState, TrajectoryOptimizer

def main():
    # Define the scenario as a YAML string
    scenario_data = """
directives:
  # Add Spot robot with arm
  - add_model:
      name: spot
      file: package://drake_trajopt_models/spot/spot_with_arm_and_floating_base_actuators.urdf

  # Add toolbox
  - add_model:
      name: toolbox
      file: package://drake_trajopt_models/toolbox/toolbox.urdf

  # Weld toolbox to the world
  - add_weld:
      parent: world
      child: toolbox::body
      X_PC:
        translation: [2.0, 0.0, 0.4]  # position the toolbox in front of Spot
    """

    # Create DrakeState with the scenario data
    drake_state = DrakeState(
        scenario_data=scenario_data,
        gripper_link_name="spot::arm_link_wr1",
        finger_link_names=["spot::arm_link_fngr"],
        run_meshcat=True
    )

    q_spot_nominal = np.array([
        0.0,  # x
        0.0,  # y
        0.0,  # yaw position
        2.26974487e-04,  # arm_sh0
        -2.30147457e+00,  # arm_sh1 
        1.62356675e+00,  # arm_el0
        7.04407692e-03,  # arm_el1
        1.55041754e+00,  # arm_wr0
        -4.28676605e-03,  # arm_wr1
        -1.543833,  # arm_f1x
    ])
    q_toolbox = -0.2 * np.ones((7,))  # open toolbox
    drake_state.SetRobotPositions(np.hstack([q_spot_nominal, q_toolbox]))
    drake_state.Publish()

    # Define a straight line path for Spot to move towards the toolbox
    # The path will be a straight line in the x-y plane
    path: List[np.ndarray] = []
    for x in np.linspace(0., 4., 100):
        q_spot = q_spot_nominal.copy()
        q_spot[0] = x
        path.append(np.hstack([q_spot, q_toolbox]))

    trajopt = TrajectoryOptimizer(drake_state)
    dv_indices = np.arange(10)  # only the first 10 positions are the decision variables
    trajopt.Smooth(path, dv_indices)


if __name__ == "__main__":
    main()
