#!/usr/bin/env python3

import numpy as np
from pydrake.all import RigidTransform, StartMeshcat
from drake_trajopt.drake_state import DrakeState

def main():
    # Define the scenario as a YAML string
    scenario_data = """
directives:
  # Add Spot robot with arm
  - add_model:
      name: spot
      file: package://drake_trajopt_models/spot/spot_with_arm_and_floating_base_actuators.urdf
      default_joint_positions:
          arm_sh0: [2.26974487e-04]
          arm_sh1: [-2.30147457e+00]
          arm_el0: [1.62356675e+00]
          arm_el1: [7.04407692e-03]
          arm_wr0: [1.55041754e+00]
          arm_wr1: [-4.28676605e-03]
          arm_f1x: [-1.543833]

  # Add toolbox
  - add_model:
      name: toolbox
      file: package://drake_trajopt_models/toolbox/toolbox.urdf
    """

    # Create DrakeState with the scenario data
    drake_state = DrakeState(
        scenario_data=scenario_data,
        gripper_link_name="spot::arm_link_wr1",
        finger_link_names=["spot::arm_link_fngr"],
        run_meshcat=True
    )

    # Position Spot in front of the table
    # Get the current positions
    plant = drake_state.plant
    plant_context = drake_state.plant_context
    current_positions = plant.GetPositions(plant_context)
    
    # Set the base position (assuming the first 3 values are x, y, z)
    # and the 4th value is the rotation around z
    current_positions[0] = 0.5  # x position
    current_positions[1] = 0.0  # y position
    current_positions[3] = 0.0  # rotation around z (facing the table)
    
    # Update the robot state with the new positions
    drake_state.SetRobotPositions(current_positions)
    drake_state.Publish()
    
    print("Scenario loaded. Press Ctrl+C to exit.")
    
    try:
        input("Press Enter to exit...")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
