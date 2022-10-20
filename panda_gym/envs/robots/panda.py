from typing import Tuple
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
            redundancy_resolution_scaling: float = 1.0
    ) -> None:
        self.block_gripper = block_gripper
        self.control_type = control_type
        self.velocity_control = velocity_control
        self.use_redundancy_resolution = redundancy_resolution
        self.redundancy_resolution_scaling = redundancy_resolution_scaling
        self.num_joints_arm = 7
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
                arm_joint_ctrl = action[:self.num_joints_arm]
                if self.use_redundancy_resolution:
                    qdot_nullspace = self.resolve_redundancy(
                        q_reference=self.neutral_joint_values[:self.num_joints_arm],
                        scaling_factor=self.redundancy_resolution_scaling
                    )
                    arm_joint_ctrl = arm_joint_ctrl + qdot_nullspace[:self.num_joints_arm]

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

    def get_jacobian(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is very closely adapted from the PyBullet examples. Can be found under
        https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/jacobian.py
        Inside the Bullet repository: bullet3/examples/pybullet/examples/jacobian.py
        """
        """
        calculateJacobian requires:

        bodyUniqueId int                body unique id, as returned by loadURDF etc.
        linkIndex int                   link index for the jacobian.
        localPosition list of float     the point on the specified link to compute the jacobian for, in
                                        link local coordinates around its center of mass.
        objPositions list of float      joint positions (angles)
        objVelocities list of float     joint velocities
        objAccelerations list of float  desired joint accelerations
        """
        # Robot's moto states:
        mpos, _, _ = self.sim.get_motor_joint_states(body=self.body_name, joints=list(self.joint_indices))
        # State of EE link:
        _, _, com_trn, _, _, _, _, _ = self.sim.get_link_state(body=self.body_name, link=self.ee_link)
        zero_vec = np.zeros(len(mpos))

        # Get the Jacobians for the CoM of the end-effector link.
        # Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn.
        # The localPosition is always defined in terms of the link frame coordinates.
        J_trans, J_rot = self.sim.calculate_jacobian(
            body=self.body_name,
            link=self.ee_link,
            local_position=list(com_trn),
            obj_positions=list(mpos),
            obj_velocities=list(zero_vec),
            obj_accelerations=list(zero_vec)
        )
        return J_trans, J_rot

    def resolve_redundancy(self, q_reference: np.ndarray, scaling_factor: float = 1.0) -> np.ndarray:
        """
        Resolve redundancy be generating a nullspace vector (joint velocity in Jacobians nullspace) to move
        the robot towards a reference config without moving the EE POSITION (orientation DOES change).
        Args:
            q_reference: 7 dimensional vector of desired joint configuration

        Returns:
            qdot_nullspace: joint velocity vector in nullspace. This vector can be added to
        """

        J_trans, J_rot = self.get_jacobian()
        J_trans = J_trans[:, :self.num_joints_arm]

        # Calculate qdot_nullspace = -scaling * (I-J^+*J)b
        J_plus = np.linalg.pinv(J_trans)
        nullspace_projector = np.identity(self.num_joints_arm) - np.matmul(J_plus, J_trans)

        # current joint config q:
        q = np.array([self.get_joint_angle(i) for i in self.joint_indices[:self.num_joints_arm]])
        assert q.shape == q_reference.shape, \
            f"Panda arm config and reference joint config shapes do not match: {q.shape} != {q_reference.shape}"
        # error vector between current joint config and desired config
        e = q - q_reference

        # project error vector into nullspace
        q_dot_nullspace = - scaling_factor * np.matmul(nullspace_projector, e)
        return q_dot_nullspace
