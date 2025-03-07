from copy import copy
import dataclasses as dc
import os
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

from pydrake.all import (
    CollisionFilterDeclaration,
    FrameId,
    GeometryId,
    GeometryInstance,
    GeometrySet,
    Mesh,
    Meshcat,
    MeshcatVisualizer,
    QueryObject,
    Rgba,
    RigidTransform,
    StartMeshcat,
    ProximityProperties,
    IllustrationProperties,
    SceneGraphInspector,
)

from manipulation.station import LoadScenario, MakeHardwareStation

from qr_spot.sim.brain import RoboverseConfToDrakePositions

import numpy as np

from .trajopt import TrajectoryOptimizer

@dc.dataclass
class GeomData:
    geom_instance: GeometryInstance
    geom_id: Optional[GeometryId] = None  # will have a geom_id if it is registered

    @property
    def is_registered(self) -> bool:
        return self.geom_id is not None


@dc.dataclass
class DrakeLink:
    link_id: str
    geoms: List[GeomData]


@dc.dataclass
class DrakeShape:
    shape_name: str
    links_dict: Dict[str, DrakeLink]


class DrakeState:
    def __init__(
            self,
            scenario_file: str | Path,
            gripper_link_name: str,
            finger_link_names: List[str] | None = None,
            run_meshcat: bool = True,
        ):
        """
        Args:
            scenario_file (str | Path): Path to the scenario file to load
                into the hardware station.
            run_meshcat (bool): Whether to run Meshcat.
        """
        self.meshcat: Optional[Meshcat] = StartMeshcat() if run_meshcat else None

        scenario = LoadScenario(str(scenario_file))
        self.station = MakeHardwareStation(scenario, self.meshcat)
        self.plant = self.station.plant()
        self.scene_graph = self.station.scene_graph()

        self.illustration_visualizer: MeshcatVisualizer = self.station.GetSubsystemByName("meshcat_visualizer(illustration)")
        self.proximity_visualizer: MeshcatVisualizer = self.station.GetSubsystemByName("meshcat_visualizer(proximity)")

        # These contexts are only created once and forever reused.
        self.root_context = self.station.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.root_context)
        self.sg_context = self.scene_graph.GetMyContextFromRoot(self.root_context)

        self.illustration_visualizer_context = self.illustration_visualizer.GetMyContextFromRoot(self.root_context)
        self.proximity_visualizer_context = self.proximity_visualizer.GetMyContextFromRoot(self.root_context)

        self.trajopt = TrajectoryOptimizer(self)

        # self._world_shapes: Dict[str, DrakeShape] = collections.OrderedDict()
        # self._grasped_shape: Optional[DrakeShape] = None

        self.world_frame_id = self.GetFrameId("world")
        self.gripper_frame_id = self.GetFrameId(gripper_link_name)
        finger_frame_ids = [self.GetFrameId(e) for e in finger_link_names]

        self.cf_manager = self.scene_graph.collision_filter_manager(self.sg_context)

        # Declaration to exclude collisions between manipuland added to the
        # gripper (wrist) frame and the finger frame.
        self.declaration_on_attach = (
            CollisionFilterDeclaration().ExcludeBetween(
                GeometrySet(self.gripper_frame_id),
                GeometrySet(finger_frame_ids),
            )
        )

        # Declaration to allow collisions between manipuland added to the world
        # frame and both the gripper (wrist) frame and the finger frame.
        self.declaration_on_release = (
            CollisionFilterDeclaration().AllowBetween(
                GeometrySet(self.world_frame_id),
                GeometrySet([self.gripper_frame_id] + finger_frame_ids),
            )
        )

        self.Publish()

    def GetFrameId (self, frame_name) -> FrameId:
        query_object: QueryObject = self.scene_graph.get_query_output_port().Eval(self.sg_context)
        inspector: SceneGraphInspector = query_object.inspector()
        for frame_id in inspector.GetAllFrameIds():
            if  inspector.GetName(frame_id) == frame_name:
                return frame_id

        raise Exception(f"Frame {frame_name} not found in SceneGraph")

    def X_WG (self) -> RigidTransform:
        query_object: QueryObject = self.scene_graph.get_query_output_port().Eval(self.sg_context)
        X_WG = query_object.GetPoseInWorld(self.gripper_frame_id)
        
        # Copying the rigid transform is important here because X_WG
        # automatically gets updated when the plant positions are set!
        return RigidTransform(X_WG)

    def SetRobotPositions(self, positions: np.ndarray):
        self.plant.SetPositions(self.plant_context, positions)

    def Publish(self):
        self.illustration_visualizer.ForcedPublish(self.illustration_visualizer_context)
        self.proximity_visualizer.ForcedPublish(self.proximity_visualizer_context)

    def AddDrakeLinkToSceneGraph(self, dlink: DrakeLink, parent_frame_id: FrameId):
        for geom_data in dlink.geoms:
            # Register each processed geometry to scene graph
            geom_instance = geom_data.geom_instance

            # We need to copy the GeometryInstance for every call to
            # SceneGraph.RegisterGeometry(), because RegisterGeometry() takes a
            # unique_ptr of the GeometryInstance, so we can't reuse the
            # GeometryInstance.
            geom_id = self.scene_graph.RegisterGeometry(
                self.sg_context,
                self.plant.get_source_id(),
                parent_frame_id,
                copy(geom_instance),
            )

            # Update geom_id
            geom_data.geom_id = geom_id

    def AddDrakeShapeToSceneGraph(self, dshape: DrakeShape, parent_frame_id: FrameId):
        for dlink in dshape.links_dict.values():
            self.AddDrakeLinkToSceneGraph(dlink, parent_frame_id)

    def ConvertDrakeShapeWorldToEndEffector(self, dshape: DrakeShape):
        for dlink in dshape.links_dict.values():
            for geom_data in dlink.geoms:
                geom_instance = geom_data.geom_instance

                # Update pose due to new parent frame: from world to end effector
                X_WO = geom_instance.pose()
                X_WG = self.X_WG()
                X_GO = X_WG.inverse() @ X_WO
                geom_instance.set_pose(X_GO)

    def ConvertDrakeShapeEndEffectorToWorld(self, dshape: DrakeShape):
        for dlink in dshape.links_dict.values():
            for geom_data in dlink.geoms:
                geom_instance = geom_data.geom_instance

                # Update pose due to new parent frame: from end effector to world
                X_GO = geom_instance.pose()
                X_WG = self.X_WG()
                X_WO = X_WG @ X_GO
                geom_instance.set_pose(X_WO)

    def RemoveDrakeLinkFromSceneGraph(self, dlink: DrakeLink):
        for geom_data in dlink.geoms:
            self.scene_graph.RemoveGeometry(
                self.sg_context,
                self.plant.get_source_id(),
                geom_data.geom_id,
            )
            geom_data.geom_id = None

    def RemoveDrakeShapeFromSceneGraph(self, dshape: DrakeShape) -> bool:
        updated = False
        for dlink in dshape.links_dict.values():
            self.RemoveDrakeLinkFromSceneGraph(dlink)
            updated = True
        return updated


    def UpdateDrakeLinkTransformInSceneGraph(
            self,
            dlink: DrakeLink,
            X_PO: RigidTransform,
        ) -> bool:
        for geom_data in dlink.geoms:
            # Skip if the transform is exactly the same.
            if geom_data.geom_instance.pose().IsExactlyEqualTo(X_PO):
                return False

            # TODO (Sid): Figure out if we only modify the transform without
            # changing the shape.
            self.scene_graph.ChangeShape(
                self.plant.get_source_id(),
                geom_data.geom_id,
                geom_data.geom_instance.shape(),
                X_PO,
            )

            return True

    # def SetGraspedShape(self, attached: Optional[Tuple[Body, Trans]]):
    #     if attached is not None:
    #         body, trans = attached
    #         attached_shape_name = body.shape.name
    #         if self._grasped_shape is not None:
    #             # Case 1: Asked to grasp something + something already grasped
    #             # Assert that we're not being asked to suddenly grasp a different shape.
    #             # If not, do not update the shape; do nothing.
    #             assert attached_shape_name == self._grasped_shape.shape_name
    #             return
    #         else:
    #             # Case 2: Asked to grasp something + nothing grasped
    #             # Grasp the shape

    #             # Check that attached shape exists in the world
    #             assert attached_shape_name in self._world_shapes, \
    #                 f"Shape {attached_shape_name} not found in self._world_shapes"

    #             # Remove shape from world
    #             dshape = self._world_shapes.pop(attached_shape_name)
    #             self.RemoveDrakeShapeFromSceneGraph(dshape)
    #             # TODO (Sid): For some reason, Publish() needs to be called
    #             # right after RemoveGeometry() to update Meshcat. Is this a bug
    #             # in Drake?
    #             self.Publish()
    #             self.ConvertDrakeShapeWorldToEndEffector(dshape)
    #             self.AddDrakeShapeToSceneGraph(dshape, self.gripper_frame_id)
    #             self.cf_manager.Apply(self.declaration_on_attach)
    #             self._grasped_shape = dshape
    #             self.Publish()
    #             return
    #     else:
    #         if self._grasped_shape is None:
    #             # Case 3: Asked to grasp nothing + nothing grasped
    #             # Do nothing.
    #             return
    #         else:
    #             # Case 4: Asked to grasp nothing + something already grasped
    #             # Release the shape
    #             dshape = self._grasped_shape
    #             self.RemoveDrakeShapeFromSceneGraph(dshape)
    #             # TODO (Sid): For some reason, Publish() needs to be called
    #             # right after RemoveGeometry() to update Meshcat. Is this a bug
    #             # in Drake?
    #             self.Publish()
    #             self.ConvertDrakeShapeEndEffectorToWorld(dshape)
    #             self.AddDrakeShapeToSceneGraph(dshape, self.world_frame_id)
    #             self.cf_manager.Apply(self.declaration_on_release)
    #             self._world_shapes[dshape.shape_name] = dshape
    #             self._grasped_shape = None
    #             self.Publish()
    #             return

    # @staticmethod
    # def DrakifyLink(
    #     link: model_module.Link,
    #     X_PO: RigidTransform,
    #     shape_name: str) -> DrakeLink:

    #     link_id = str(id(link))

    #     meshes = []
    #     colors = []

    #     imeshes = link.visual_mesh or link.collision_meshes
    #     if not isinstance(imeshes, list):
    #         imeshes = [imeshes]
    #     for mesh in imeshes:
    #         if use_random_colors_objects:
    #             color = np.random.rand(4)
    #         else:
    #             color = mesh.visual.face_colors[0]
    #         meshes.append(mesh)
    #         colors.append(color)

    #     dlink = DrakeLink(link_id, [])
    #     for i, (mesh, color) in enumerate(zip(meshes, colors)):
    #         mesh_id = f"{link_id}_{i}"
    #         file_name = f"{data_directory}{mesh_id}.obj"

    #         mesh.export(file_name)
    #         mesh = Mesh(file_name)

    #         rgba = ((c if c <= 1. else c / 255.) for c in color)

    #         mesh_name = f"{shape_name}_{link_id}"
    #         geom_instance = GeometryInstance(X_PO, mesh, mesh_name)
    #         assert geom_instance.pose().IsExactlyEqualTo(X_PO)
    #         proximity_properties = ProximityProperties()
    #         illustration_properties = IllustrationProperties()
    #         for properties in [illustration_properties, proximity_properties]:
    #             properties.AddProperty("phong", "diffuse", Rgba(*rgba))
    #         geom_instance.set_proximity_properties(proximity_properties)
    #         geom_instance.set_illustration_properties(illustration_properties)

    #         dlink.geoms.append(GeomData(geom_instance))
        
    #     return dlink

    # def GetHpnLinksFromShape(self, shape: Shape) -> List[model_module.Link]:
    #     if isinstance(shape, model_module.Link):
    #         links = [shape]
    #     elif isinstance(shape, model_module.CascadedLink):
    #         links = shape.link_list
    #     else:
    #         raise TypeError('shape must be Link or CascadedLink')
    #     return links
    
    # def MergeHpnShapeIntoDrakeShape(
    #         self,
    #         shape: Shape,
    #         X_PO: RigidTransform,
    #         dshape: DrakeShape,
    #     ) -> bool:

    #     links = self.GetHpnLinksFromShape(shape)
    #     new_links_dict = {str(id(link)): link for link in links}

    #     updated = False

    #     # Case 1: Remove links that are not in the new shape
    #     for link_id in dshape.links_dict.keys() - new_links_dict.keys():
    #         dlink = dshape.links_dict.pop(link_id)
    #         self.RemoveDrakeLinkFromSceneGraph(dlink)
    #         updated = True

    #     for link_id, link in new_links_dict.items():
    #         if link_id in dshape.links_dict.keys():
    #             # Case 2: Link is a match, we only need to update the transform
    #             dlink = dshape.links_dict[link_id]
    #             updated |= self.UpdateDrakeLinkTransformInSceneGraph(dlink, X_PO)
    #         else:
    #             # Case 3: Link is not a match, we need to add the link
    #             dlink = self.DrakifyLink(link, X_PO, dshape.shape_name)
    #             self.AddDrakeLinkToSceneGraph(dlink, self.world_frame_id)
    #             dshape.links_dict[link_id] = dlink
    #             updated = True
        
    #     return updated

    # =========================================================================
    # region Public API
    # =========================================================================

    def UpdateRobotState(self, positions: np.ndarray):
        self.SetRobotPositions(positions)
        self.Publish()
        return


    # def add_shape(self, shape: Shape, transform: Trans, publish: bool=True) -> bool:
    #     """
    #     Returns:
    #         updated (bool): True if anything was updated in the viewer.
    #     """
    #     # Do not add shape if it is grasped
    #     if (self._grasped_shape is not None) and (shape.name == self._grasped_shape.shape_name):
    #         return False

    #     links = self.GetHpnLinksFromShape(shape)
    #     X_PO = RigidTransform(transform.matrix)

    #     # Sid: Unsure why TLPK were doing this; but I'm doing this too so that
    #     # removing it doesn't break anything.
    #     for link in links:
    #         link.update(force=True)

    #     if shape.name in self._world_shapes:
    #         dshape = self._world_shapes[shape.name]
    #     else:
    #         # Add shape to the world
    #         dshape = DrakeShape(shape.name, {})
    #         self._world_shapes[shape.name] = dshape

    #     updated = self.MergeHpnShapeIntoDrakeShape(shape, X_PO, dshape)
    #     if updated and publish: self.Publish()
    #     return updated


    # def remove_shape(self, shape_name: str, publish: bool=True) -> bool:
    #     """
    #     Returns:
    #         updated (bool): True if anything was updated in the viewer.
    #     """
    #     # Do not remove shape if it is grasped
    #     if (self._grasped_shape is not None) and (shape_name == self._grasped_shape.shape_name):
    #         return False

    #     if shape_name in self._world_shapes:
    #         dshape = self._world_shapes.pop(shape_name)
    #         updated = self.RemoveDrakeShapeFromSceneGraph(dshape)
    #         if updated and publish: self.Publish()
    #         return updated
    #     else:
    #         return False


    # def UpdateObjects(self, shapes: List[Tuple[Shape, Trans]], publish: bool=True) -> bool:
    #     """
    #     Add a batch of shapes and remove any that are not in the batch
    #     shapes is a list of (shape, trans)
    #     """
    #     updated = False

    #     # Add all shapes in this batch
    #     for (shape, trans) in shapes:
    #         updated |= self.add_shape(shape, trans, publish=False)

    #     # Remove any existing shapes from world that were not in this batch
    #     new_names = {shape.name for (shape, trans) in shapes}
    #     for shape_name in self._world_shapes.keys() - new_names:
    #         updated |= self.remove_shape(shape_name, publish=False)

    #     if updated and publish: self.Publish()
    #     return updated
    
    # def UpdateFromPhys(self, phys: Physical):    
    #     view_shapes = []
    #     for bname in phys._bodies:
    #         if bname in phys.permanent_bnames and bname in phys.drawn:
    #             continue
    #         trans = phys._body_trans.get(bname, None)
    #         if trans is None: continue
    #         shape = phys._bodies[bname].shape
    #         view_shapes.append((shape, trans))
        
    #     conf = phys.get_conf()
    #     for (s, t) in phys.attached_shape_trans(conf):
    #         view_shapes.append((s, t))

    #     # for shadow in phys._shadows.values():
    #     #     view_shapes.append((shadow.shape, TRI))

    #     # Update robot state
    #     conf = phys.get_conf()
    #     self.UpdateRobotState(conf)

    #     # This removes any shapes not in current display
    #     # viewer.new_shapes(view_shapes)
    #     self.UpdateObjects(view_shapes)


    # endregion Public API
    # =========================================================================

