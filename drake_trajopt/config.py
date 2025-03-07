from addict import Dict

# Trajectory optimization config
trajopt = Dict()
trajopt.limits.velocity = 1
trajopt.limits.acceleration = 1
trajopt.kot.max_control_points = 20
trajopt.kot.num_collision_checks = 15  # options - int or "default"
trajopt.kot.collision_margin = 0.02  # 2cm
trajopt.kot.collision_tolerance = 1e-3  # 1mm
trajopt.kot.viz_sleep_time = 0.1  # seconds
trajopt.kot.num_viz_traj_samples = 30
trajopt.toppra.num_grid_points = 100
