import numpy as np
from gym import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class Panda(PyBulletRobot):
    """Panda robot in PyBullet.

    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
            self,
            sim: PyBullet,
            block_gripper: bool = False,
            base_position: np.ndarray = np.array([0.0, 0.0, 0.0]),
            control_type: str = "ee",
            velocity_control: bool = False,
            redundancy_resolution: bool = False,
    ) -> None:
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.velocity_control = velocity_control
        self.use_redundancy_resolution = redundancy_resolution
        n_action = 3 if self.control_type == "ee" else 7  # control (x, y z) if "ee", else, control the 7 joints
        n_action += 0 if self.block_gripper else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="panda",
            file_name="franka_panda/panda.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 2, 3, 4, 5, 6, 9, 10]),
            joint_forces=np.array([87.0, 87.0, 87.0, 87.0, 12.0, 120.0, 120.0, 170.0, 170.0]),
        )

        self.fingers_indices = np.array([9, 10])
        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])
        self.ee_link = 11
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[0], lateral_friction=1.0)
        self.sim.set_lateral_friction(self.body_name, self.fingers_indices[1], lateral_friction=1.0)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[0], spinning_friction=0.001)
        self.sim.set_spinning_friction(self.body_name, self.fingers_indices[1], spinning_friction=0.001)

    def set_action(self, action: np.ndarray) -> None:
        action = action.copy()  # ensure action don't change
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.velocity_control:
            if self.control_type == "ee":
                raise NotImplementedError("Velocity control is not supported in EE control atm.")
            else:
                arm_joint_ctrl = action[:7]
                if self.use_redundancy_resolution:
                    raise NotImplementedError("Redundancy resolution not yet implemented")
                # target_velocities = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

            if self.block_gripper:
                fingers_ctrl = -0.01
            else:
                fingers_ctrl = action[-1] * 0.2  # limit maximum change in position

            target_velocities = np.concatenate((arm_joint_ctrl, [fingers_ctrl, fingers_ctrl]))
            self.velocity_control_joints(target_velocity=target_velocities)
        else:
            if self.control_type == "ee":
                ee_displacement = action[:3]
                target_arm_angles = self.ee_displacement_to_target_arm_angles(ee_displacement)
            else:
                arm_joint_ctrl = action[:7]
                target_arm_angles = self.arm_joint_ctrl_to_target_arm_angles(arm_joint_ctrl)

            if self.block_gripper:
                target_fingers_width = 0
            else:
                fingers_ctrl = action[-1] * 0.2  # limit maximum change in position
                fingers_width = self.get_fingers_width()
                target_fingers_width = fingers_width + fingers_ctrl

            target_angles = np.concatenate((target_arm_angles, [target_fingers_width / 2, target_fingers_width / 2]))
            self.control_joints(target_angles=target_angles)

    def ee_displacement_to_target_arm_angles(self, ee_displacement: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the end-effector displacement.

        Args:
            ee_displacement (np.ndarray): End-effector displacement, as (dx, dy, dy).

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        ee_displacement = ee_displacement[:3] * 0.05  # limit maximum change in position
        # get the current position and the target position
        ee_position = self.get_ee_position()
        target_ee_position = ee_position + ee_displacement
        # Clip the height target. For some reason, it has a great impact on learning
        target_ee_position[2] = np.max((0, target_ee_position[2]))
        # compute the new joint angles
        target_arm_angles = self.inverse_kinematics(
            link=self.ee_link, position=target_ee_position, orientation=np.array([1.0, 0.0, 0.0, 0.0])
        )
        target_arm_angles = target_arm_angles[:7]  # remove fingers angles
        return target_arm_angles

    def arm_joint_ctrl_to_target_arm_angles(self, arm_joint_ctrl: np.ndarray) -> np.ndarray:
        """Compute the target arm angles from the arm joint control.

        Args:
            arm_joint_ctrl (np.ndarray): Control of the 7 joints.

        Returns:
            np.ndarray: Target arm angles, as the angles of the 7 arm joints.
        """
        arm_joint_ctrl = arm_joint_ctrl * 0.05  # limit maximum change in position
        # get the current position and the target position
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(7)])
        target_arm_angles = current_arm_joint_angles + arm_joint_ctrl
        return target_arm_angles

    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening
        if not self.block_gripper:
            fingers_width = self.get_fingers_width()
            obs = np.concatenate((ee_position, ee_velocity, [fingers_width]))
        else:
            obs = np.concatenate((ee_position, ee_velocity))
        return obs

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_fingers_width(self) -> float:
        """Get the distance between the fingers."""
        finger1 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[0])
        finger2 = self.sim.get_joint_angle(self.body_name, self.fingers_indices[1])
        return finger1 + finger2

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)

    # def get_jacobian(self):
    #     """
    #     This function is very closely adapted from the PyBullet examples. Can be found under
    #     https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/jacobian.py
    #     Inside the Bullet repository: bullet3/examples/pybullet/examples/jacobian.py
    #
    #     Returns:
    #
    #     """
    #     num_joints = 7
    #     assert isinstance(self.sim, PyBullet), f"Simulation must be type PyBullet but got f{type(self.sim)} instead."
    #
    #     def getMotorJointStates(robot):
    #         """
    #         getJointStates output:
    #         jointPosition:              The position value of this joint.
    #         jointVelocity:              The velocity value of this joint.
    #         jointReactionForces:        These are the joint reaction forces, if a torque sensor is enabled for this joint it
    #                                     is [Fx, Fy, Fz, Mx, My, Mz]. Without torque sensor, it is [0,0,0,0,0,0].
    #         appliedJointMotorTorque:    This is the motor torque applied during the last stepSimulation
    #         """
    #         joint_states_all = self.sim.physics_client.getJointStates(robot, range(num_joints))
    #         """
    #         getJointInfo output:
    #         jointIndex:         the same joint index as the input parameter
    #         jointName:          the name of the joint, as specified in the URDF (or SDF etc) file
    #         jointType:          type of the joint, this also implies the number of position and velocity variables.
    #         qIndex:             the first position index in the positional state variables for this body
    #         uIndex:             the first velocity index in the velocity state variables for this body
    #         """
    #         joint_infos = [self.sim.physics_client.getJointInfo(robot, i) for i in range(num_joints)]
    #         joint_states = [j for j, i in zip(joint_states_all, joint_infos) if i[3] > -1]
    #         joint_positions = [state[0] for state in joint_states]
    #         joint_velocities = [state[1] for state in joint_states]
    #         joint_torques = [state[3] for state in joint_states]
    #         return joint_positions, joint_velocities, joint_torques
    #
    #     def multiplyJacobian3D(robot, jacobian, vector):
    #         result = [0.0, 0.0, 0.0]
    #         i = 0
    #         joint_infos = [p.getJointInfo(robot, i) for i in range(len(vector))]
    #         for c in range(len(vector)):
    #             info = joint_infos[c]
    #             if info[3] > -1:
    #                 for r in range(3):
    #                     result[r] += jacobian[r][i] * vector[c]
    #                 i += 1
    #         return result
    #
    #     # Get the joint and link state directly from Bullet.
    #     pos, vel, torq = getJointStates(kukaId)
    #     mpos, mvel, mtorq = getMotorJointStates(kukaId)
    #
    #     zero_vec = [0.0] * len(mpos)
    #     jac_t, jac_r = p.calculateJacobian(
    #         bodyUniqueId=kukaId,
    #         linkIndex=kukaEndEffectorIndex,
    #         localPosition=com_trn,
    #         objPositions=mpos,
    #         objVelocities=zero_vec,
    #         objAccelerations=zero_vec
    #     )
    #
    #     # Motor state
    #     mpos, mvel, mtorq = getMotorJointStates(self.sim._bodies_idx[self.body_name])
    #
    #     """
    #         FROM PYBULLET DOC:
    #     position_linkcom_world, world_rotation_linkcom,
    #     position_linkcom_frame, frame_rotation_linkcom,
    #     position_frame_world, world_rotation_frame,
    #     linearVelocity_linkcom_world, angularVelocity_linkcom_world
    #       = getLinkState(objectUniqueId, linkIndex, computeLinkVelocity=0,
    #                      computeForwardKinematics=0, physicsClientId=0)
    #     Provides extra information such as the Cartesian world coordinates center of mass (COM) of the link, relative to the world reference frame.
    #     """
    #     linkstate = self.sim.physics_client.getLinkState(
    #         bodyUniqueId=self.sim._bodies_idx[self.body_name],
    #         linkIndex=self.ee_link,
    #         computeLinkVelocity=1,
    #         computeForwardKinematics=1
    #     )
    #     link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = linkstate
    #     """
    #         FROM QUICKSTART GUIDE:
    #     link_trn    =   linkWorldPosition               (Cartesian position of center of mass)
    #     link_rot    =   linkWorldOrientation            (Cartesian orientation of center of mass, in
    #                                                     quaternion [x,y,z,w])
    #     com_trn     =   localInertialFramePosition      (local position offset of inertial frame (center of mass)
    #                                                     expressed in the URDF link frame)
    #     com_rot     =   localInertialFrameOrientation   (local orientation (quaternion [x,y,z,w]) offset of the inertial
    #                                                     frame expressed in URDF link frame.)
    #     frame_pos   =   worldLinkFramePosition          (world position of the URDF link frame)
    #     frame_rot   =   worldLinkFrameOrientation       (world orientation of the URDF link frame)
    #     link_vt     =   worldLinkLinearVelocity         (Cartesian world velocity. Only returned if computeLinkVelocity
    #                                                     non-zero)
    #     link_vr     =   worldLinkAngularVelocity        (Cartesian world velocity. Only returned if computeLinkVelocity
    #                                                     non-zero)
    #
    #     """
    #
    #     """
    #     calculateJacobian requires:
    #
    #     bodyUniqueId int                body unique id, as returned by loadURDF etc.
    #     linkIndex int                   link index for the jacobian.
    #     localPosition list of float     the point on the specified link to compute the jacobian for, in
    #                                     link local coordinates around its center of mass.
    #     objPositions list of float      joint positions (angles)
    #     objVelocities list of float     joint velocities
    #     objAccelerations list of float  desired joint accelerations
    #     """
    #     # Get the Jacobians for the CoM of the end-effector link.
    #     # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
    #     # The localPosition is always defined in terms of the link frame coordinates.
    #     J = self.sim.physics_client.calculateJacobian(
    #         bodyUniqueId=self.sim._bodies_idx[self.body_name],
    #         linkIndex=self.ee_link,
    #         localPosition=com_trn,
    #         objPositions=mpos,
    #         objVelocities=zero_vec,
    #         objAccelerations=zero_vec
    #     )
    #     J_trans, J_rot = J
    #     return J
