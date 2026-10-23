"""Assemble the original Galaxea R1 arms with two wrist-mounted ISyHands."""
import copy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'assets/galaxea/R1/urdf/r1_v2_1_0.urdf'
HAND = ROOT / 'assets/isyhand/urdf/isyhand_description/ISyHand_v6.urdf'
COMMIT = 'b81c32520b1d1a3b3299df0c038e042001b46ae3'


def origin_matrix(origin):
    result = np.eye(4)
    if origin is not None:
        result[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        result[:3, :3] = Rotation.from_euler(
            'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return result


def poses(robot, angles):
    """URDF link transforms, including fixed mount transforms."""
    joints = list(robot.findall('joint'))
    children = {j.find('child').get('link') for j in joints}
    roots = {link.get('name') for link in robot.findall('link')} - children
    assert len(roots) == 1, roots
    result = {roots.pop(): np.eye(4)}
    while joints:
        ready = [j for j in joints if j.find('parent').get('link') in result]
        assert ready, 'Disconnected URDF'
        for joint in ready:
            transform = origin_matrix(joint.find('origin'))
            angle = angles.get(joint.get('name'), 0.)
            if joint.get('type') in ('revolute', 'continuous'):
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                transform[:3, :3] @= Rotation.from_rotvec(axis * angle).as_matrix()
            result[joint.find('child').get('link')] = result[joint.find('parent').get('link')] @ transform
            joints.remove(joint)
    return result


def assemble(path, hand_scale=1.):
    tree = ET.parse(SOURCE)
    robot = tree.getroot()
    robot.set('name', 'r1_isyhand')
    # Replace grippers and remove wrist cameras that protrude beside the new hands.
    for element in list(robot):
        if any(part in element.get('name', '') for part in ('gripper', 'realsense')):
            robot.remove(element)
    for joint in robot.findall('joint'):
        if '_arm_joint' not in joint.get('name'):
            joint.set('type', 'fixed')
    for mesh in robot.findall('.//mesh'):
        mesh.set('filename', str(SOURCE.parent.parent / 'meshes' / Path(mesh.get('filename')).name))
    for side in ('left', 'right'):
        attach_hand(robot, side, hand_scale, path.parent / 'left_hand_meshes')
    for mesh in robot.findall('.//mesh'):
        assert Path(mesh.get('filename')).is_file(), mesh.attrib
    assert sum(j.get('type') != 'fixed' for j in robot.findall('joint')) == 48
    tree.write(path, encoding='utf-8', xml_declaration=True)
    return robot


def mirror_hand(hand, mesh_dir):
    """Reflect geometry and joint frames across local XZ, keeping joint values."""
    import trimesh

    reflection = np.diag([1., -1., 1.])
    mesh_dir.mkdir(parents=True, exist_ok=True)
    for origin in hand.findall('.//origin'):
        transform = origin_matrix(origin)
        origin.set('xyz', ' '.join(map(str, reflection @ transform[:3, 3])))
        origin.set('rpy', ' '.join(map(str, Rotation.from_matrix(
            reflection @ transform[:3, :3] @ reflection).as_euler('xyz'))))
    for joint in hand.findall('joint'):
        axis = joint.find('axis')
        if axis is not None:
            vector = reflection @ np.fromstring(axis.get('xyz'), sep=' ')
            # Rotation axes are axial vectors: det(S) S a, unlike translations.
            if joint.get('type') in ('revolute', 'continuous'):
                vector = -vector
            axis.set('xyz', ' '.join(map(str, vector)))
    for inertia in hand.findall('.//inertia'):
        for component in ('ixy', 'iyz'):
            inertia.set(component, str(-float(inertia.get(component, '0'))))
    converted = {}
    for mesh in hand.findall('.//mesh'):
        source = Path(mesh.get('filename'))
        if source not in converted:
            geometry = trimesh.load(source, force='mesh', process=False)
            transform = np.eye(4)
            transform[:3, :3] = reflection
            # Trimesh also reverses triangle winding for this reflection.
            geometry.apply_transform(transform)
            target = mesh_dir / (source.stem + '.obj')
            geometry.export(target)
            converted[source] = target.resolve()
        mesh.set('filename', str(converted[source]))


def attach_hand(robot, side, hand_scale, mesh_dir):
    hand = ET.parse(HAND).getroot()
    # Scale link offsets and geometry together for a coherent kinematic preview.
    # Source mass/inertia are retained: this asset is not calibrated for dynamics.
    for origin in hand.findall('.//origin'):
        xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') * hand_scale
        origin.set('xyz', ' '.join(map(str, xyz)))
    for mesh in hand.findall('.//mesh'):
        scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ') * hand_scale
        mesh.set('scale', ' '.join(map(str, scale)))
    prefix = side + '_hand_'
    for mesh in hand.findall('.//mesh'):
        mesh.set('filename', str((HAND.parent.parent / mesh.get('filename')).resolve()))
    if side == 'left':
        mirror_hand(hand, mesh_dir)
    for node in hand.iter():
        if node.get('name'):
            node.set('name', prefix + node.get('name'))
        if node.tag in ('parent', 'child'):
            node.set('link', prefix + node.get('link'))
    for element in hand:
        robot.append(copy.deepcopy(element))
    mount = ET.SubElement(robot, 'joint', name=side + '_hand_mount', type='fixed')
    ET.SubElement(mount, 'parent', link=side + '_arm_link6')
    ET.SubElement(mount, 'child', link=prefix + 'base_link')
    lateral = .021 if side == 'left' else -.021
    ET.SubElement(mount, 'origin', xyz=f'{.03425 + .025 * hand_scale - .001} {lateral * hand_scale} {-.03375 * hand_scale}', rpy='0 0 0')
    # Attach to the actual arm_link6 mesh face (x=34.25 mm), not the removed gripper frame.
    # Center the palm mesh across the wrist axis; its source origin lies at a corner.
    flange = robot.find(f"link[@name='{side}_arm_link6']")
    visual = ET.SubElement(flange, 'visual')
    ET.SubElement(visual, 'origin', xyz=f'{.03425 + .0125 * hand_scale - .0005} 0 0', rpy='0 1.5707963267948966 0')
    geometry = ET.SubElement(visual, 'geometry')
    ET.SubElement(geometry, 'cylinder', radius=str(.025 * hand_scale), length=str(.025 * hand_scale + .001))


def solve_pose(robot, base_height):
    result = {}
    checks = {}
    desired_rotation = Rotation.from_euler('x', np.pi).as_matrix()
    for side, y in [('left', .27), ('right', -.27)]:
        names = [f'{side}_arm_joint{i}' for i in range(1, 7)]
        limits = [robot.find(f"joint[@name='{name}']/limit") for name in names]
        lower = np.array([float(limit.get('lower')) for limit in limits])
        upper = np.array([float(limit.get('upper')) for limit in limits])
        target = np.array([.43, y, 1.03 - base_height])

        def residual(q):
            wrist = poses(robot, dict(zip(names, q)))[side + '_hand_base_link']
            rotation_error = Rotation.from_matrix(desired_rotation.T @ wrist[:3, :3]).as_rotvec()
            return np.r_[wrist[:3, 3] - target, .3 * rotation_error]

        rng = np.random.default_rng(42)
        solutions = []
        for _ in range(12):
            initial = rng.uniform(lower, upper)
            solution = least_squares(residual, initial, bounds=(lower, upper),
                                     max_nfev=160, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            if np.linalg.norm(solution.fun) < 1e-5:
                solutions.append(solution)
        assert solutions, side
        # Prefer the wrist branch with less forearm roll among accurate solutions.
        best = min(solutions, key=lambda solution: abs(solution.x[3]))
        assert np.linalg.norm(best.fun) < 1e-5, (side, best.fun)
        result.update(zip(names, best.x.tolist()))
        checks[side] = {'target_world_m': (target + [0, 0, base_height]).tolist(),
                        'residual': best.fun.tolist(), 'joint_angles_rad': best.x.tolist()}
    return result, checks


def floor_height(robot):
    import trimesh
    transforms = poses(robot, {})
    low = 0.
    for link in robot.findall('link'):
        if not any(word in link.get('name') for word in ('wheel', 'base_link')) or 'hand' in link.get('name'):
            continue
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                continue
            vertices = trimesh.load(mesh.get('filename'), force='mesh', process=False).vertices
            transform = transforms[link.get('name')] @ origin_matrix(visual.find('origin'))
            low = min(low, (vertices @ transform[:3, :3].T + transform[:3, 3])[:, 2].min())
    return float(-low)
