import os
import time
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple, List

import numpy as np
import pybullet as p
import pybullet_data
import pybullet_utils.bullet_client as bc

import panda_gym.assets


class PyBullet:
    """Convenient class to use PyBullet physics engine.

    Args:
        render (bool, optional): Enable rendering. Defaults to False.
        n_substeps (int, optional): Number of sim substep when step() is called. Defaults to 20.
        background_color (np.ndarray, optional): The background color as (red, green, blue).
            Defaults to np.array([223, 54, 45]).
    """

    def __init__(
            self, render: bool = False, n_substeps: int = 20,
            background_color: np.ndarray = np.array([223.0, 54.0, 45.0])
    ) -> None:
        self.background_color = background_color.astype(np.float64) / 255
        options = "--background_color_red={} \
                    --background_color_green={} \
                    --background_color_blue={}".format(
            *self.background_color
        )
        self.connection_mode = p.GUI if render else p.DIRECT
        self.physics_client = bc.BulletClient(connection_mode=self.connection_mode, options=options)
        self.physics_client.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        if not render:
            self.physics_client.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

        self.n_substeps = n_substeps
        self.timestep = 1.0 / 500
        self.physics_client.setTimeStep(self.timestep)
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0, 0, -9.81)
        self._bodies_idx = {}

    @property
    def dt(self):
        """Timestep."""
        return self.timestep * self.n_substeps

    def step(self) -> None:
        """Step the simulation."""
        for _ in range(self.n_substeps):
            self.physics_client.stepSimulation()

    def close(self) -> None:
        """Close the simulation."""
        self.physics_client.disconnect()

    def render(
            self,
            mode: str = "human",
            width: int = 720,
            height: int = 480,
            target_position: np.ndarray = np.zeros(3),
            distance: float = 1.4,
            yaw: float = 45,
            pitch: float = -30,
            roll: float = 0,
    ) -> Optional[np.ndarray]:
        """Render.

        If mode is "human", make the rendering real-time. All other arguments are
        unused. If mode is "rgb_array", return an RGB array of the scene.

        Args:
            mode (str): "human" of "rgb_array". If "human", this method waits for the time necessary to have
                a realistic temporal rendering and all other args are ignored. Else, return an RGB array.
            width (int, optional): Image width. Defaults to 720.
            height (int, optional): Image height. Defaults to 480.
            target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
                Defaults to [0., 0., 0.].
            distance (float, optional): Distance of the camera. Defaults to 1.4.
            yaw (float, optional): Yaw of the camera. Defaults to 45.
            pitch (float, optional): Pitch of the camera. Defaults to -30.
            roll (int, optional): Rool of the camera. Defaults to 0.

        Returns:
            RGB np.ndarray or None: An RGB array if mode is 'rgb_array', else None.
        """
        if mode == "human":
            self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_SINGLE_STEP_RENDERING)
            time.sleep(self.dt)  # wait to seems like real speed
        if mode == "rgb_array":
            if self.connection_mode == p.DIRECT:
                warnings.warn(
                    "The use of the render method is not recommended when the environment "
                    "has not been created with render=True. The rendering will probably be weird. "
                    "Prefer making the environment with option `render=True`. For example: "
                    "`env = gym.make('PandaReach-v2', render=True)`.",
                    UserWarning,
                )
            view_matrix = self.physics_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=target_position,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2,
            )
            proj_matrix = self.physics_client.computeProjectionMatrixFOV(
                fov=60, aspect=float(width) / height, nearVal=0.1, farVal=100.0
            )
            (_, _, px, depth, _) = self.physics_client.getCameraImage(
                width=width,
                height=height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )

            return px

    def get_base_position(self, body: str) -> np.ndarray:
        """Get the position of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[0]
        return np.array(position)

    def get_base_orientation(self, body: str) -> np.ndarray:
        """Get the orientation of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The orientation, as quaternion (x, y, z, w).
        """
        orientation = self.physics_client.getBasePositionAndOrientation(self._bodies_idx[body])[1]
        return np.array(orientation)

    def get_base_rotation(self, body: str, type: str = "euler") -> np.ndarray:
        """Get the rotation of the body.

        Args:
            body (str): Body unique name.
            type (str): Type of angle, either "euler" or "quaternion"

        Returns:
            np.ndarray: The rotation.
        """
        quaternion = self.get_base_orientation(body)
        if type == "euler":
            rotation = self.physics_client.getEulerFromQuaternion(quaternion)
            return np.array(rotation)
        elif type == "quaternion":
            return np.array(quaternion)
        else:
            raise ValueError("""type must be "euler" or "quaternion".""")

    def get_base_velocity(self, body: str) -> np.ndarray:
        """Get the velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[0]
        return np.array(velocity)

    def get_base_angular_velocity(self, body: str) -> np.ndarray:
        """Get the angular velocity of the body.

        Args:
            body (str): Body unique name.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getBaseVelocity(self._bodies_idx[body])[1]
        return np.array(angular_velocity)

    def get_link_state(self, body: str, link: int) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            FROM QUICKSTART GUIDE:
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = link_state
        link_trn    =   linkWorldPosition               (Cartesian position of center of mass)
        link_rot    =   linkWorldOrientation            (Cartesian orientation of center of mass, in
                                                        quaternion [x,y,z,w])
        com_trn     =   localInertialFramePosition      (local position offset of inertial frame (center of mass)
                                                        expressed in the URDF link frame)
        com_rot     =   localInertialFrameOrientation   (local orientation (quaternion [x,y,z,w]) offset of the inertial
                                                        frame expressed in URDF link frame.)
        frame_pos   =   worldLinkFramePosition          (world position of the URDF link frame)
        frame_rot   =   worldLinkFrameOrientation       (world orientation of the URDF link frame)
        link_vt     =   worldLinkLinearVelocity         (Cartesian world velocity. Only returned if computeLinkVelocity
                                                        non-zero)
        link_vr     =   worldLinkAngularVelocity        (Cartesian world velocity. Only returned if computeLinkVelocity
                                                        non-zero)

            FROM PYBULLET DOC:
            Provides extra information such as the Cartesian world coordinates center of mass (COM) of the link,
            relative to the world reference frame.
        position_linkcom_world, world_rotation_linkcom,
        position_linkcom_frame, frame_rotation_linkcom,
        position_frame_world, world_rotation_frame,
        linearVelocity_linkcom_world, angularVelocity_linkcom_world
        = getLinkState(
                objectUniqueId,
                linkIndex,
                computeLinkVelocity=0,
                computeForwardKinematics=0,
                physicsClientId=0
            )
        """
        link_state = self.physics_client.getLinkState(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            computeLinkVelocity=1,
            computeForwardKinematics=1
        )
        link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = link_state
        return np.array(link_trn), np.array(link_rot), np.array(com_trn), np.array(com_rot), np.array(
            frame_pos), np.array(frame_rot), np.array(link_vt), np.array(link_vr)

    def get_link_position(self, body: str, link: int) -> np.ndarray:
        """Get the position of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The position, as (x, y, z).
        """
        position = self.physics_client.getLinkState(self._bodies_idx[body], link)[0]
        return np.array(position)

    def get_link_orientation(self, body: str, link: int) -> np.ndarray:
        """Get the orientation of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The rotation, as (rx, ry, rz).
        """
        orientation = self.physics_client.getLinkState(self._bodies_idx[body], link)[1]
        return np.array(orientation)

    def get_link_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The velocity, as (vx, vy, vz).
        """
        velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[6]
        return np.array(velocity)

    def get_link_angular_velocity(self, body: str, link: int) -> np.ndarray:
        """Get the angular velocity of the link of the body.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.

        Returns:
            np.ndarray: The angular velocity, as (wx, wy, wz).
        """
        angular_velocity = self.physics_client.getLinkState(self._bodies_idx[body], link, computeLinkVelocity=True)[7]
        return np.array(angular_velocity)

    def get_joint_angle(self, body: str, joint: int) -> float:
        """Get the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The angle.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[0]

    def get_joint_velocity(self, body: str, joint: int) -> float:
        """Get the velocity of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body

        Returns:
            float: The velocity.
        """
        return self.physics_client.getJointState(self._bodies_idx[body], joint)[1]

    def get_joint_states(self, body: str, joints: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the joint states (position, velocity and torques) of the joint of the body.
        Args:
            body (str): Body unique name.
            joints (list of int): Joint indices in the body

        Returns:
            float: The velocity.
        """

        joint_states = self.physics_client.getJointStates(self._bodies_idx[body], joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return np.array(joint_positions), np.array(joint_velocities), np.array(joint_torques)

    def get_joint_info(self, body: str, joints: List[int]) -> List[Tuple]:
        """

        Args:
            body (str): Body unique name.
            joints (list of int): Joint indices in the body

        Returns:
            jointIndex:         the same joint index as the input parameter
            jointName:          the name of the joint, as specified in the URDF (or SDF etc) file
            jointType:          type of the joint, this also implies the number of position and velocity variables.
            qIndex:             the first position index in the positional state variables for this body
            uIndex:             the first velocity index in the velocity state variables for this body
            and more...
        """
        joint_infos = [self.physics_client.getJointInfo(self._bodies_idx[body], i) for i in joints]
        return joint_infos

    def get_motor_joint_states(self, body: str, joints: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        This function is very closely adapted from the PyBullet examples. Can be found under
        https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/jacobian.py
        Inside the Bullet repository: bullet3/examples/pybullet/examples/jacobian.py
        Args:
            body (str): Body unique name.
            joints (list of int): Joint indices in the body

        Returns: motor joint_positions, joint_velocities, joint_torques

        """
        """
        getJointStates output:
        jointPosition:              The position value of this joint.
        jointVelocity:              The velocity value of this joint.
        jointReactionForces:        These are the joint reaction forces, if a torque sensor is enabled for this joint it
                                    is [Fx, Fy, Fz, Mx, My, Mz]. Without torque sensor, it is [0,0,0,0,0,0].
        appliedJointMotorTorque:    This is the motor torque applied during the last stepSimulation
        """
        joint_states_all = self.physics_client.getJointStates(self._bodies_idx[body], joints)
        joint_infos = self.get_joint_info(body, joints)
        joint_states = [j for j, i in zip(joint_states_all, joint_infos) if i[3] > -1]
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        return np.array(joint_positions), np.array(joint_velocities), np.array(joint_torques)

    def set_base_pose(self, body: str, position: np.ndarray, orientation: np.ndarray) -> None:
        """Set the position of the body.

        Args:
            body (str): Body unique name.
            position (np.ndarray): The position, as (x, y, z).
            orientation (np.ndarray): The target orientation as quaternion (x, y, z, w).
        """
        if len(orientation) == 3:
            orientation = self.physics_client.getQuaternionFromEuler(orientation)
        self.physics_client.resetBasePositionAndOrientation(
            bodyUniqueId=self._bodies_idx[body], posObj=position, ornObj=orientation
        )

    def set_joint_angles(self, body: str, joints: np.ndarray, angles: np.ndarray) -> None:
        """Set the angles of the joints of the body.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            angles (np.ndarray): List of target angles, as a list of floats.
        """
        for joint, angle in zip(joints, angles):
            self.set_joint_angle(body=body, joint=joint, angle=angle)

    def set_joint_angle(self, body: str, joint: int, angle: float) -> None:
        """Set the angle of the joint of the body.

        Args:
            body (str): Body unique name.
            joint (int): Joint index in the body.
            angle (float): Target angle.
        """
        self.physics_client.resetJointState(bodyUniqueId=self._bodies_idx[body], jointIndex=joint, targetValue=angle)

    def control_joints(self, body: str, joints: np.ndarray, target_angles: np.ndarray, forces: np.ndarray) -> None:
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            target_angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.POSITION_CONTROL,
            targetPositions=target_angles,
            forces=forces,
        )

    def velocity_control_joints(self, body: str, joints: np.ndarray, target_velocity: np.ndarray,
                                forces: np.ndarray) -> None:
        """Control the joints motor.

        Args:
            body (str): Body unique name.
            joints (np.ndarray): List of joint indices, as a list of ints.
            target_angles (np.ndarray): List of target angles, as a list of floats.
            forces (np.ndarray): Forces to apply, as a list of floats.
        """
        self.physics_client.setJointMotorControlArray(
            self._bodies_idx[body],
            jointIndices=joints,
            controlMode=self.physics_client.VELOCITY_CONTROL,
            targetVelocities=target_velocity,
            forces=forces,
        )

    def inverse_kinematics(self, body: str, link: int, position: np.ndarray, orientation: np.ndarray) -> np.ndarray:
        """Compute the inverse kinematics and return the new joint state.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            position (np.ndarray): Desired position of the end-effector, as (x, y, z).
            orientation (np.ndarray): Desired orientation of the end-effector as quaternion (x, y, z, w).

        Returns:
            np.ndarray: The new joint state.
        """
        # # lower limits for null space
        # ll = [-.967, -2, -2.96, -2.29, -2.96, -2.09, -3.05]
        # # upper limits for null space
        # ul = [0.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
        # # joint ranges for null space
        # jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        # # restposes for null space
        # rp = [0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00]
        # # joint damping coefficents
        # jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        joint_state = self.physics_client.calculateInverseKinematics(
            bodyIndex=self._bodies_idx[body],
            endEffectorLinkIndex=link,
            targetPosition=position,
            targetOrientation=orientation,
            # lowerLimits=ll,
            # upperLimits=ul,
            # jointRanges=jr,
            # restPoses=rp
        )
        return np.array(joint_state)

    def calculate_jacobian(
            self,
            body: str,
            link: int,
            local_position: list,
            obj_positions: list,
            obj_velocities: list,
            obj_accelerations: list
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
            calculateJacobian requires:
        bodyUniqueId int                body unique id, as returned by loadURDF etc.
        linkIndex int                   link index for the jacobian.
        localPosition list of float     the point on the specified link to compute the jacobian for, in
                                        link local coordinates around its center of mass.
        objPositions list of float      joint positions (angles)
        objVelocities list of float     joint velocities
        objAccelerations list of float  desired joint accelerations

        Args:
            body:
            link:
            local_position:
            obj_positions:
            obj_velocities:
            obj_accelerations:

        Returns:

        """
        J_trans, J_rot = self.physics_client.calculateJacobian(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            localPosition=local_position,
            objPositions=obj_positions,
            objVelocities=obj_velocities,
            objAccelerations=obj_accelerations
        )
        return np.array(J_trans), np.array(J_rot)

    def place_visualizer(self, target_position: np.ndarray, distance: float, yaw: float, pitch: float) -> None:
        """Orient the camera used for rendering.

        Args:
            target (np.ndarray): Target position, as (x, y, z).
            distance (float): Distance from the target position.
            yaw (float): Yaw.
            pitch (float): Pitch.
        """
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target_position,
        )

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        """Disable rendering within this context."""
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 0)
        yield
        self.physics_client.configureDebugVisualizer(self.physics_client.COV_ENABLE_RENDERING, 1)

    def loadURDF(self, body_name: str, **kwargs: Any) -> None:
        """Load URDF file.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
        """
        self._bodies_idx[body_name] = self.physics_client.loadURDF(**kwargs)

    def create_box(
            self,
            body_name: str,
            half_extents: np.ndarray,
            mass: float,
            position: np.ndarray,
            rgba_color: Optional[np.ndarray] = np.ones(4),
            specular_color: np.ndarray = np.zeros(3),
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            texture: Optional[str] = None,
    ) -> None:
        """Create a box.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            half_extents (np.ndarray): Half size of the box in meters, as (x, y, z).
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            texture (str or None, optional): Texture file name. Defaults to None.
        """
        visual_kwargs = {
            "halfExtents": half_extents,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"halfExtents": half_extents}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_BOX,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )
        if texture is not None:
            texture_path = os.path.join(panda_gym.assets.get_data_path(), texture)
            texture_uid = self.physics_client.loadTexture(texture_path)
            self.physics_client.changeVisualShape(self._bodies_idx[body_name], -1, textureUniqueId=texture_uid)

    def create_cylinder(
            self,
            body_name: str,
            radius: float,
            height: float,
            mass: float,
            position: np.ndarray,
            rgba_color: Optional[np.ndarray] = np.zeros(4),
            specular_color: np.ndarray = np.zeros(3),
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a cylinder.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            height (float): The height in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "length": height,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius, "height": height}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_CYLINDER,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def create_sphere(
            self,
            body_name: str,
            radius: float,
            mass: float,
            position: np.ndarray,
            rgba_color: Optional[np.ndarray] = np.zeros(4),
            specular_color: np.ndarray = np.zeros(3),
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a sphere.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            radius (float): The radius in meter.
            mass (float): The mass in kg.
            position (np.ndarray): The position, as (x, y, z).
            rgba_color (np.ndarray, optional): Body color, as (r, g, b, a). Defaults as [0, 0, 0, 0]
            specular_color (np.ndarray, optional): Specular color, as (r, g, b). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        visual_kwargs = {
            "radius": radius,
            "specularColor": specular_color,
            "rgbaColor": rgba_color,
        }
        collision_kwargs = {"radius": radius}
        self._create_geometry(
            body_name,
            geom_type=self.physics_client.GEOM_SPHERE,
            mass=mass,
            position=position,
            ghost=ghost,
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
            visual_kwargs=visual_kwargs,
            collision_kwargs=collision_kwargs,
        )

    def _create_geometry(
            self,
            body_name: str,
            geom_type: int,
            mass: float = 0.0,
            position: np.ndarray = np.zeros(3),
            ghost: bool = False,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
            visual_kwargs: Dict[str, Any] = {},
            collision_kwargs: Dict[str, Any] = {},
    ) -> None:
        """Create a geometry.

        Args:
            body_name (str): The name of the body. Must be unique in the sim.
            geom_type (int): The geometry type. See self.physics_client.GEOM_<shape>.
            mass (float, optional): The mass in kg. Defaults to 0.
            position (np.ndarray, optional): The position, as (x, y, z). Defaults to [0, 0, 0].
            ghost (bool, optional): Whether the body can collide. Defaults to False.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
            visual_kwargs (dict, optional): Visual kwargs. Defaults to {}.
            collision_kwargs (dict, optional): Collision kwargs. Defaults to {}.
        """
        baseVisualShapeIndex = self.physics_client.createVisualShape(geom_type, **visual_kwargs)
        if not ghost:
            baseCollisionShapeIndex = self.physics_client.createCollisionShape(geom_type, **collision_kwargs)
        else:
            baseCollisionShapeIndex = -1
        self._bodies_idx[body_name] = self.physics_client.createMultiBody(
            baseVisualShapeIndex=baseVisualShapeIndex,
            baseCollisionShapeIndex=baseCollisionShapeIndex,
            baseMass=mass,
            basePosition=position,
        )

        if lateral_friction is not None:
            self.set_lateral_friction(body=body_name, link=-1, lateral_friction=lateral_friction)
        if spinning_friction is not None:
            self.set_spinning_friction(body=body_name, link=-1, spinning_friction=spinning_friction)

    def create_plane(self, z_offset: float) -> None:
        """Create a plane. (Actually, it is a thin box.)

        Args:
            z_offset (float): Offset of the plane.
        """
        self.create_box(
            body_name="plane",
            half_extents=np.array([3.0, 3.0, 0.01]),
            mass=0.0,
            position=np.array([0.0, 0.0, z_offset - 0.01]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.15, 0.15, 0.15, 1.0]),
        )

    def create_table(
            self,
            length: float,
            width: float,
            height: float,
            x_offset: float = 0.0,
            lateral_friction: Optional[float] = None,
            spinning_friction: Optional[float] = None,
    ) -> None:
        """Create a fixed table. Top is z=0, centered in y.

        Args:
            length (float): The length of the table (x direction).
            width (float): The width of the table (y direction)
            height (float): The height of the table.
            x_offset (float, optional): The offet in the x direction.
            lateral_friction (float or None, optional): Lateral friction. If None, use the default pybullet
                value. Defaults to None.
            spinning_friction (float or None, optional): Spinning friction. If None, use the default pybullet
                value. Defaults to None.
        """
        self.create_box(
            body_name="table",
            half_extents=np.array([length, width, height]) / 2,
            mass=0.0,
            position=np.array([x_offset, 0.0, -height / 2]),
            specular_color=np.zeros(3),
            rgba_color=np.array([0.95, 0.95, 0.95, 1]),
            lateral_friction=lateral_friction,
            spinning_friction=spinning_friction,
        )

    def set_lateral_friction(self, body: str, link: int, lateral_friction: float) -> None:
        """Set the lateral friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            lateral_friction (float): Lateral friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            lateralFriction=lateral_friction,
        )

    def set_spinning_friction(self, body: str, link: int, spinning_friction: float) -> None:
        """Set the spinning friction of a link.

        Args:
            body (str): Body unique name.
            link (int): Link index in the body.
            spinning_friction (float): Spinning friction.
        """
        self.physics_client.changeDynamics(
            bodyUniqueId=self._bodies_idx[body],
            linkIndex=link,
            spinningFriction=spinning_friction,
        )

    # def get_jacobian(self):
    #     J = self.physics_client.calculateJacobian(
    #         self._bodies_idx[body_name],
    #         self.ee_link,
    #         self.get_ee_position(),
    #         [self.get_joint_angle(idx) for idx in range(7)],
    #         [self.get_joint_velocity(idx) for idx in range(7)],
    #         [0]*7
    #     )
    #     return J
