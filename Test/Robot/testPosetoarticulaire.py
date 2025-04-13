import numpy as np
from scipy.spatial.transform import Rotation as R

# UR5 DH parameters (in meters)
d1 = 0.089159
a2 = -0.425
a3 = -0.39225
d4 = 0.10915
d5 = 0.09465
d6 = 0.0823

#Compute the inverse kinematics for a given end-effector pose (x, y, z) and orientation (alpha, beta, gamma)
def compute_inverse_kinematics(x, y, z, alpha, beta, gamma):
    # Step 1: Compute the homogeneous transformation (position + orientation)
    r = R.from_euler('xyz', [alpha, beta, gamma])
    R06 = r.as_matrix()
    
    # Position of the tool (TCP)
    p06 = np.array([x, y, z])

    # Compute the wrist center position (remove the last joint offset)
    pw = p06 - d6 * R06[:, 2]  # d6 along z-axis of the tool

    # q1: base rotation angle
    q1 = np.arctan2(pw[1], pw[0])

    # Planar distances for solving q2 and q3
    r1 = np.sqrt(pw[0]**2 + pw[1]**2)
    r2 = pw[2] - d1
    r3 = np.sqrt(r1**2 + r2**2)

    # Law of cosines to solve for q3
    cos_q3 = (r3**2 - a2**2 - a3**2) / (2 * a2 * a3)
    q3 = np.arccos(np.clip(cos_q3, -1.0, 1.0))  # Clamp to avoid domain error

    # q2 based on triangle geometry
    q2 = np.arctan2(r2, r1) - np.arctan2(a3 * np.sin(q3), a2 + a3 * np.cos(q3))

    # Helper functions for rotation matrices
    def Rz(theta): return np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta),  np.cos(theta), 0],
                                     [0, 0, 1]])
    def Ry(theta): return np.array([[ np.cos(theta), 0, np.sin(theta)],
                                     [0, 1, 0],
                                     [-np.sin(theta), 0, np.cos(theta)]])

    # Compute R03 using forward kinematics up to joint 3
    R03 = Rz(q1) @ Ry(q2) @ Ry(q3)

    # Compute R36 = R03.T * R06 (remaining rotation from joint 4 to 6)
    R36 = R03.T @ R06

    # Solve for q4, q5, q6 from R36
    q5 = np.arccos(np.clip(R36[2, 2], -1.0, 1.0))
    q4 = np.arctan2(R36[1, 2], R36[0, 2])
    q6 = np.arctan2(R36[2, 1], -R36[2, 0])

    return np.array([q1, q2, q3, q4, q5, q6])

# Example pose
x, y, z = 0.3, 0.2, 0.5
alpha, beta, gamma = 0, np.pi/2, 0  # orientation in RPY (radians)

joint_angles = compute_inverse_kinematics(x, y, z, alpha, beta, gamma)
print("Joint angles (radians):", np.round(joint_angles, 4))
