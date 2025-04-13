import random
import numpy as np

#poseGridUpCube
#poseCubeUp

# Exemple d'une fonction de cinématique inverse (à adapter selon votre robot)
def inverse_kinematics_func(target_pose):

    # Exemple d'angles de joints calculés de manière fictive
    joint_angles = [0.5, -0.3, 1.2, -1.5, 0.7, 0.4]  # Exemple de coordonnées articulaires
    
    # En réalité, il faudrait utiliser un algorithme de cinématique inverse spécifique au robot.
    return joint_angles

# Fonction pour trouver un nombre aléatoire et récupérer la position d'un cube
def find_and_compute_inverse_kinematics(robot_position, cubes_positions, camera_distance, offset):
    # Find a random cube position from the list
    random_index = random.randint(0, len(cubes_positions) - 1)
    cube_position = cubes_positions.pop(random_index)  # Delete the position in the list

    # Create a pose for the cube in the robot's coordinate system
    pose_cube_robot = np.array(robot_position) + np.array(camera_distance) + np.array(cube_position) + np.array([0,0,offset])

    # 3. Calculer la cinématique inverse pour obtenir la position articulaire du robot
    # Appel à une fonction de cinématique inverse (fournir la fonction de cinématique inverse)
    joint_angles = inverse_kinematics_func(pose_cube_robot)

    # Retourner la position articulaire calculée
    return joint_angles, pose_cube_robot, cube_position

# Exemple d'utilisation
robot_position = [0.5, 0.2, 0.8]  # Position actuelle du robot (x, y, z)
cubes_positions = [[0.3, 0.6, 0.7], [1.0, 0.8, 0.5], [0.4, 1.2, 0.3]]  # Liste des positions des cubes
camera_distance = [0.1, 0.1, 0.1]   # Distance entre la caméra et le gripper
offset = 0.2  # Offset for the cube position 

joint_angles, pose_cube_robot, cube_position = find_and_compute_inverse_kinematics(robot_position, cubes_positions, camera_distance, inverse_kinematics_func)

# Affichage des résultats
print("Position du cube sélectionné :", cube_position)
print("Pose du robot + cube :", pose_cube_robot)
print("Angles articulaires calculés :", joint_angles)
