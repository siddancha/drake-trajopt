"""
Drake's KinematicTrajectoryOptimization + TOPP-RA for smoothing path.

Author: Siddharth Ancha

Code references:
 • https://github.com/RussTedrake/manipulation/blob/master/book/trajectories/kinematic_trajectory_optimization.ipynb
 • https://github.com/cohnt/smoothing-demo-for-lis/blob/main/smoothing.py
"""

import numpy as np
import time
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING

from .timer import Timer
from .utils import Logger
import drake_trajopt.config as cfg

from pydrake.all import (
    BsplineBasis,
    BsplineTrajectory,
    KinematicTrajectoryOptimization,
    MathematicalProgramResult,
    MinimumDistanceLowerBoundConstraint,
    PathParameterizedTrajectory,
    PiecewisePolynomial,
    Rgba,
    QueryObject,
    SceneGraphInspector,
    SignedDistancePair,
    SnoptSolver,
    Solve,
    Toppra,
    Trajectory,
    Variable,
)
from manipulation.meshcat_utils import PublishPositionTrajectory

if TYPE_CHECKING:
    from .drake_state import DrakeState


class TrajectoryOptimizer:
    def __init__(self, drake_state: 'DrakeState'):
        self.drake_state = drake_state

    @property
    def plant(self):
        return self.drake_state.plant

    @property
    def scene_graph(self):
        return self.drake_state.scene_graph

    @property
    def plant_context(self):
        return self.drake_state.plant_context

    @property
    def sg_context(self):
        return self.drake_state.sg_context

    @property
    def root_context(self):
        return self.drake_state.root_context

    # Color constants
    RED = Rgba(1.0, 0.0, 0.0, 1.0)
    YELLOW = Rgba(1.0, 1.0, 0.0, 1.0)
    BLUE = Rgba(0.0, 0.0, 1.0, 1.0)
    WHITE = Rgba(1.0, 1.0, 1.0, 0.5)
    BLACK = Rgba(0.0, 0.0, 0.0, 1.0)

    def RehearseTrajectory(
            self,
            path: List[np.ndarray],
            times: List[float]) -> np.ndarray:
        """
        Args:
            path (List[np.ndarray]): Path to rehearse.
            times (List[float]): Times corresponding to each configuration.
        """
        assert times[0] == 0

        for i in range(1, len(path)):
            # Sleep till next timestep.
            duration = times[i] - times[i - 1]
            time.sleep(duration)

            # Set the positions
            self.plant.SetPositions(self.plant_context, path[i])

            # Publish to Meshcat.
            self.drake_state.Publish()

        return path[-1]

    # =========================================================================
    # region Public methods
    # =========================================================================

    def Smooth(
            self,
            path_guess: List[np.ndarray],
            dv_indices: List[int],
        ) -> Tuple[List[np.ndarray], List[float]]:
        """
        Args:
            path_guess (List[np.ndarray, dtype=float, shape=(P,)]): path guess
            dv_indices (List[int]): indices of decision variables to optimize

        Returns:
            (List[np.ndarray, dtype=float, shape=(P,)]): smoothed path
            (List[float]): times corresponding to each configuration
        """
        # Save actual positions to reset them later.
        saved_positions = self.plant.GetPositions(self.plant_context)

        Logger().INFO("Performing trajopt smoothing ...")

        timer = Timer(num_skip=0)

        # Make initial Bspline from RRT path.
        traj = self.MakeBsplineTrajectoryFromInitPath(path_guess)

        # Plot RRT trajectory.
        self.PlotEndEffectorTrajInMeshcat(traj, "/init_path", self.WHITE)

        # Perform kinematic trajectory optimization.
        traj, result = self.TriLevelKinematicTrajopt(traj, dv_indices, timer)

        if not result.is_success():
            Logger.ERROR(
                f"SNOPT failed. Returning the best trajectory found so far.",
                raise_exception=False,
            )
            # Give a special color to the trajectory when SNOPT fails
            self.PlotEndEffectorTrajInMeshcat(traj, "/trajopt_smoothing_path", self.BLACK)

        # Perform TOPP-RA time scaling.
        traj = self.Toppra(traj, timer)

        assert traj.start_time() == 0
        Logger().INFO("\n" + timer.get_stats_string())

        t_result = np.linspace(traj.start_time(), traj.end_time(), num=100)
        path_result = [traj.value(t)[:, 0] for t in t_result]

        self.RehearseTrajectory(path_result, t_result)

        # Reset the positions
        # self.plant.SetPositions(self.plant_context, saved_positions)
        # self.drake_state.Publish()

        return path_result, t_result


    def UniLevelKinematicTrajopt(self,
        init_guess: BsplineTrajectory,
        dv_indices: List[int],
        timer: Timer,
    ) -> Tuple[BsplineTrajectory, MathematicalProgramResult]:    

        # Compute collision bound in start and goal positions.
        bound = self.ComputeCollisionBoundInStartAndGoalPositions(init_guess, timer)

        # Setup trajopt problem without collision constraints.
        trajopt = self.MakeKinematicTrajoptWithoutCollisionConstraints(
            init_guess = init_guess,
            dv_indices = dv_indices,
            traj_viz_color = self.YELLOW if bound <= 0 else self.RED,
        )

        if bound <= 0:
            # Do not apply collision constraints.
            Logger.WARN("Start or goal positions too close to collision. "
                        "Not applying collision constraints")
        else:
            self.ApplyCollisionConstraints(trajopt, bound)

        return self.SolveKinematicTrajopt(trajopt, timer)


    def BiLevelKinematicTrajopt(
            self,
            init_guess: BsplineTrajectory,
            dv_indices: List[int],
            timer: Timer,
    ) -> Tuple[BsplineTrajectory, MathematicalProgramResult]:

        # Setup trajopt problem without collision constraints.
        trajopt = self.MakeKinematicTrajoptWithoutCollisionConstraints(
            init_guess = init_guess,
            dv_indices = dv_indices,
            traj_viz_color = self.YELLOW,
        )

        # Solve trajopt without collision constraints.
        traj, result = self.SolveKinematicTrajopt(trajopt, timer)
        # TODO (Sid): Do something if result.is_success() is False?

        # Compute collision bound in start and goal positions.
        bound = self.ComputeCollisionBoundInStartAndGoalPositions(traj, timer)

        if bound <= 0:
            # Do not apply collision constraints.
            Logger.WARN("Start or goal positions too close to collision. "
                        "Not applying collision constraints")
            return traj, result

        # Now, bound > 0, therefore collision constraints *should* be applied.

        trajopt = self.MakeKinematicTrajoptWithoutCollisionConstraints(
            init_guess = traj,
            dv_indices = dv_indices,
            traj_viz_color = self.RED,
        )
        self.ApplyCollisionConstraints(trajopt, bound)
        
        return self.SolveKinematicTrajopt(trajopt, timer)


    def TriLevelKinematicTrajopt(
            self,
            init_guess: BsplineTrajectory,
            dv_indices: List[int],
            timer: Timer,
    ) -> Tuple[BsplineTrajectory, MathematicalProgramResult]:

        # Compute collision bound in start and goal positions.
        bound = self.ComputeCollisionBoundInStartAndGoalPositions(init_guess, timer)

        # Setup trajopt problem without collision constraints.
        trajopt = self.MakeKinematicTrajoptWithoutCollisionConstraints(
            init_guess = init_guess,
            dv_indices = dv_indices,
            traj_viz_color = self.YELLOW if bound <= 0 else self.RED,
        )

        if bound <= 0:
            # Do not apply collision constraints.
            Logger.WARN("Start or goal positions too close to collision. "
                        "Not applying collision constraints")
            return self.SolveKinematicTrajopt(trajopt, timer)

        # Now bound > 0, therefore collision constraints *should* be applied.
        self.ApplyCollisionConstraints(trajopt, bound)

        # Solve uni-level trajopt with collision constraints.
        traj, result = self.SolveKinematicTrajopt(trajopt, timer)
        if result.is_success():
            # Uni-level trajopt with collision constraints succeeded. Return here.
            return traj, result

        # Uni-level trajopt with collision constraints failed. Now we will try
        # trajopt again in a bi-level fashion.

        # Setup trajopt problem without collision constraints.
        trajopt = self.MakeKinematicTrajoptWithoutCollisionConstraints(
            init_guess = init_guess,
            dv_indices = dv_indices,
            traj_viz_color = self.YELLOW,
        )

        # Solve trajopt without collision constraints.       
        traj, result = self.SolveKinematicTrajopt(trajopt, timer)
        # TODO (Sid): Do something if result.is_success() is False?

        # Now we will use this solution to initialize a trajopt problem with
        # collision constraints.
        trajopt = self.MakeKinematicTrajoptWithoutCollisionConstraints(
            init_guess = traj,
            dv_indices = dv_indices,
            traj_viz_color = self.BLUE,
        )
        self.ApplyCollisionConstraints(trajopt, bound)

         # Solve trajopt with collision constraints.       
        return self.SolveKinematicTrajopt(trajopt, timer)

    # endregion
    # =========================================================================


    def GetClearance(self, positions: np.ndarray, max_distance: float) -> float:
        # Save actual positions to reset them later.
        saved_positions = self.plant.GetPositions(self.plant_context)

        # Temporarily set these positions to compute clearance.
        self.plant.SetPositions(self.plant_context, positions)

        # Inspector from scene graph context
        query_object: QueryObject = self.scene_graph.get_query_output_port().Eval(self.sg_context)
        inspector: SceneGraphInspector = query_object.inspector()

        # Inspector from scene graph model
        # inspector = scene_graph.model_inspector()

        clearance = float('inf')
        nearest_pair_names = None

        # Tommy's code using ComputeSignedDistancePairClosestPoints()
        # pairs = inspector.GetCollisionCandidates()
        # for pair in pairs:
        #     geomId_A, geomId_B = pair
        #     name_A = inspector.GetName(geomId_A)
        #     name_B = inspector.GetName(geomId_B)
        #     sd_pair: SignedDistancePair = query_object.ComputeSignedDistancePairClosestPoints(geomId_A, geomId_B)
        #     clearance = min(clearance, sd_pair.distance)

        sd_pairs: List[SignedDistancePair] = \
            query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=max_distance)
        for sd_pair in sd_pairs:
            if sd_pair.distance < clearance:
                clearance = sd_pair.distance
                
                # Get frame names of the nearest pair
                geomId_A, geomId_B = sd_pair.id_A, sd_pair.id_B
                frameId_A = inspector.GetFrameId(geomId_A)
                frameId_B = inspector.GetFrameId(geomId_B)
                name_A = f"{inspector.GetName(frameId_A)}/{inspector.GetName(geomId_A)}"
                name_B = f"{inspector.GetName(frameId_B)}/{inspector.GetName(geomId_B)}"
                nearest_pair_names = (name_A, name_B)

        # Reset the positions
        self.plant.SetPositions(self.plant_context, saved_positions)

        return clearance, nearest_pair_names

    @staticmethod
    def SampleVizValuesFromBsplineTraj(traj: BsplineTrajectory) -> np.ndarray:
        """
        Returns:
            (np.ndarray, dtype=float, shape=(P, N)): evenly spaced samples
                from the trajectory.
        """
        num_samples = cfg.trajopt.kot.num_viz_traj_samples
        return traj.vector_values(np.linspace(0, 1, num_samples),)  # (P, N)


    def PlotEndEffectorTrajInMeshcat(
            self,
            traj: BsplineTrajectory | np.ndarray,
            meshcat_path: str,
            traj_viz_color: Rgba,
        ) -> np.ndarray:
        """
        Args:
            traj BsplineTrajectory | (np.ndarray, dtype=float, shape=(P, C)):
                Bspline trajectory, or evenly sampled values from the trajectory.
        """
        traj_values = (
            traj if isinstance(traj, np.ndarray)
            else self.SampleVizValuesFromBsplineTraj(traj)
        )

        # Loop that computes forward kinematics of the end-effector for
        # each configuration in the trajectory.
        ee_traj = []
        for i in range(traj_values.shape[1]):
            positions = traj_values[:, i]  # (P,)
            self.plant.SetPositions(self.plant_context, positions)
            X_WG = self.drake_state.X_WG()
            point_3d = X_WG.translation()  # (3,)
            ee_traj.append(point_3d)
        ee_traj = np.stack(ee_traj, axis=1)  # (3, N)

        self.drake_state.meshcat.SetLine(
            meshcat_path,
            ee_traj,
            line_width=4,
            rgba=traj_viz_color,
        )

        return traj_values


    def PlotEndEffectorTrajInMeshcatCallback(
            self,
            prev_traj_values: Optional[np.ndarray],
            control_points: np.ndarray,
            traj_viz_color: Rgba,
        ) -> np.ndarray:
        """
        Args:
            control_points (np.ndarray, dtype=float, shape=(P * C)): control points.
        """
        control_points = control_points.reshape((self.drake_state.P, -1))  # (P, C)
        basis = BsplineBasis(order=4, num_basis_functions=control_points.shape[1])
        traj = BsplineTrajectory(basis, control_points)
        traj_values = self.SampleVizValuesFromBsplineTraj(traj)  # (P, N)

        # Sleep if the trajectory has changed significantly, so that the user
        # is able to notice the previous trajectory.
        # optimization.
        TOLERANCE = 1e-3  # 1mm
        if prev_traj_values is not None:
            diff = traj_values - prev_traj_values  # (P, N)
            distance: np.ndarray = np.linalg.norm(diff, axis=0)  # (N,)
            if distance.max() > TOLERANCE:
                time.sleep(cfg.trajopt.kot.viz_sleep_time)

        return self.PlotEndEffectorTrajInMeshcat(
            traj_values, "/trajopt_smoothing_path", traj_viz_color,
        )


    def MakeBsplineTrajectoryFromInitPath(self, path_guess: np.ndarray) \
        -> BsplineTrajectory:
        """
        Args:
            q_guess (List[np.ndarray, dtype=float, shape=(C,)]): path guess
            dv_indices (List[int]): indices of decision variables to optimize
        """
        # Subsample control points if there are too many
        if len(path_guess) > cfg.trajopt.kot.max_control_points:
            step = len(path_guess) // cfg.trajopt.kot.max_control_points
            path_guess = path_guess[::step]
        num_control_pts = len(path_guess)
        Logger.INFO(f"[KTO] Number of control points: {num_control_pts}")

        basis = BsplineBasis(order=4, num_basis_functions=num_control_pts)
        path_guess = np.stack(path_guess, axis=1)  # (P, C)
        return BsplineTrajectory(basis, path_guess)


    def ComputeCollisionBoundInStartAndGoalPositions(
            self,
            traj: BsplineTrajectory,
            timer: Timer,
    ) -> float:
        bound = cfg.trajopt.kot.collision_margin
        influence_distance_offset = 0.1

        q_start = traj.InitialValue()  # (P,)
        q_goal  = traj.FinalValue()    # (P,)

        # Compute clearance for the start position
        with timer.time_as("Start clearance"):
            clearance, nearest_pair_names = self.GetClearance(q_start, max_distance=influence_distance_offset)
            Logger.INFO("")
            Logger.INFO(f"[KTO] Start min distance: {clearance:.4f}")
            Logger.INFO(f"[KTO] between pair {nearest_pair_names[0]} and {nearest_pair_names[1]}")
            Logger.INFO("")
            clearance_bound = max(clearance - cfg.trajopt.kot.collision_tolerance, 0)
            bound = min(bound, clearance_bound)

        # Compute clearance for the goal position
        with timer.time_as("Goal clearance"):
            clearance, nearest_pair_names = self.GetClearance(q_goal, max_distance=influence_distance_offset)
            Logger.INFO("")
            Logger.INFO(f"[KTO] Goal min distance: {clearance:.4f}")
            Logger.INFO(f"[KTO] between pair {nearest_pair_names[0]} and {nearest_pair_names[1]}")
            Logger.INFO("")
            clearance_bound = max(clearance - cfg.trajopt.kot.collision_tolerance, 0)
            bound = min(bound, clearance_bound)

        return bound


    def MakeKinematicTrajoptWithoutCollisionConstraints(
            self,
            init_guess: BsplineTrajectory,
            dv_indices: List[int],
            traj_viz_color: Rgba,
        ) -> KinematicTrajectoryOptimization:
        """
        Args:
            num_control_pts (int): number of control points.
            q_start (np.ndarray, dtype=float, shape=(P,)): start configuration.
            q_goal (np.ndarray, dtype=float, shape=(P,)): goal configuration.
            timer: Timer object to measure time.
            dv_indices (List[int]): indices of decision variables to optimize.
        """
        num_q = self.plant.num_positions()
        num_control_pts = init_guess.num_control_points()

        q_start = init_guess.InitialValue()  # (P,)
        q_goal  = init_guess.FinalValue()    # (P,)

        trajopt = KinematicTrajectoryOptimization(init_guess)
        prog = trajopt.get_mutable_prog()
        trajopt.prog().SetSolverOption(SnoptSolver.id(), 'Print file', "/tmp/snopt.out")

        # Add constraint to freeze certain positions in the entire trajectory
        vars = prog.decision_variables()[:-1]  # (P * C)
        vars = vars.reshape([num_control_pts, num_q]).T  # (P, C)
        frozen_indices = [p for p in range(num_q) if p not in dv_indices]
        bcs = []
        for p in frozen_indices:
            for c in range(num_control_pts):
                var, target = vars[p, c], q_start[p]
                bc = prog.AddConstraint(var, target, target)
                bcs.append(bc)
        assert prog.CheckSatisfiedAtInitialGuess(bcs)

        # Duration cost and constraints
        trajopt.AddDurationCost(1.0)
        bc = trajopt.AddDurationConstraint(0.5, 10)
        assert prog.CheckSatisfiedAtInitialGuess(bc)

        # Path length cost
        # TODO (Sid): When collision constraints start working,
        # switch to energy cost.
        # trajopt.AddPathLengthCost(1.0)
        trajopt.AddPathEnergyCost(1.0)
        
        # Position bounds
        position_min = self.plant.GetPositionLowerLimits()
        position_max = self.plant.GetPositionUpperLimits()
        bcs = trajopt.AddPositionBounds(position_min, position_max)
        assert prog.CheckSatisfiedAtInitialGuess(bcs)

        # Velocity bounds
        velocity_min = np.maximum(self.plant.GetVelocityLowerLimits(), -cfg.trajopt.limits.velocity)
        velocity_max = np.minimum(self.plant.GetVelocityUpperLimits(), +cfg.trajopt.limits.velocity)
        trajopt.AddVelocityBounds(velocity_min, velocity_max)

        # End point position constraints
        bc = trajopt.AddPathPositionConstraint(q_start, q_start, 0)
        assert prog.CheckSatisfiedAtInitialGuess(bc)
        bc = trajopt.AddPathPositionConstraint(q_goal , q_goal , 1)
        assert prog.CheckSatisfiedAtInitialGuess(bc)

        # End point velocity constraints
        zeros = np.zeros((num_q, 1))  # (P,)
        trajopt.AddPathVelocityConstraint(zeros, zeros, 0)
        trajopt.AddPathVelocityConstraint(zeros, zeros, 1)

        # Add callback to visualize end-effector trajectory during optimization
        # process.
        viz_traj_values = None
        def VizCallback(control_points: np.ndarray[Variable]):
            """
            Args:
                control_points (np.ndarray, dtype=float, shape=(P, C)): control points.
            """
            nonlocal viz_traj_values
            viz_traj_values = self.PlotEndEffectorTrajInMeshcatCallback(
                viz_traj_values, control_points, traj_viz_color,
            )
        prog.AddVisualizationCallback(VizCallback, trajopt.control_points().ravel())

        return trajopt


    def ApplyCollisionConstraints(
        self,
        trajopt: KinematicTrajectoryOptimization,
        bound: float,
    ):
        influence_distance_offset = 0.1
        prog = trajopt.get_mutable_prog()
    
        Logger.INFO(f"[KTO] Collision distance lower bound: {bound:.4f}")
        collision_constraint = MinimumDistanceLowerBoundConstraint(
            self.plant,
            bound,
            self.plant_context,
            None,
            influence_distance_offset,
        )
        num_collision_checks = cfg.trajopt.kot.num_collision_checks
        if num_collision_checks == 'default':
            num_collision_checks = trajopt.num_control_points()
        num_collision_checks = max(num_collision_checks, 2)
        Logger.INFO(f"[KTO] Num collision checks: {num_collision_checks}")
        for s in (0, 1):
            # Collision constraints should be satisfied at the start and end.
            bc = trajopt.AddPathPositionConstraint(collision_constraint, s)
            if not prog.CheckSatisfiedAtInitialGuess(bc):
                Logger.ERROR(
                    f"[KTO] Collision constraint not satisfied by the "
                    f"initial path at the start or goal position "
                    f"corresponding to s={s}. However, they were already "
                    f"checked before.",
                    raise_exception=True,
                )
        for s in np.linspace(0, 1, num=num_collision_checks)[1:-1]:
            bc = trajopt.AddPathPositionConstraint(collision_constraint, s)
            if not prog.CheckSatisfiedAtInitialGuess(bc):
                Logger.WARN(
                    f"[KTO] Collision constraint not satisfied by the "
                    f"initial path at the intermediate position "
                    f"corresponding to s={s}. Note that this might cause "
                    f"SNOPT to fail.",
                )


    def SolveKinematicTrajopt(
            self,
            trajopt: KinematicTrajectoryOptimization,
            timer: Timer,
        ) -> Tuple[BsplineTrajectory, MathematicalProgramResult]:
        """
        NOTE: This assumes that the optimization problem has been setup, and
        a it has been initialized to an initial guess.
        """
        prog = trajopt.get_mutable_prog()

        with timer.time_as("Kinematic Trajopt"):
            result = Solve(prog)

        traj: BsplineTrajectory = trajopt.ReconstructTrajectory(result)
        return traj, result


    def Toppra(self, traj: Trajectory, timer: Timer) -> PathParameterizedTrajectory:
        toppra = Toppra(
            traj,
            self.plant,
            np.linspace(
                traj.start_time(), traj.end_time(),
                cfg.trajopt.toppra.num_grid_points,
            ),
        )
        
        # Velocity bounds
        velocity_min = np.maximum(self.plant.GetVelocityLowerLimits(), -cfg.trajopt.limits.velocity)
        velocity_max = np.minimum(self.plant.GetVelocityUpperLimits(), +cfg.trajopt.limits.velocity)
        toppra.AddJointVelocityLimit(velocity_min, velocity_max)

        # Acceleration bounds
        acceleration_min = np.maximum(self.plant.GetAccelerationLowerLimits(), -cfg.trajopt.limits.acceleration)
        acceleration_max = np.minimum(self.plant.GetAccelerationUpperLimits(), +cfg.trajopt.limits.acceleration)
        toppra.AddJointAccelerationLimit(acceleration_min, acceleration_max)

        # Computes a 1D piecewise polynomial of the trajectory of time
        with timer.time_as("TOPP-RA"):
            time_traj: Optional[PiecewisePolynomial] = toppra.SolvePathParameterization()

        if time_traj is None:
            raise RuntimeError("TOPP-RA failed to solve path parameterization")

        return PathParameterizedTrajectory(traj, time_traj)
