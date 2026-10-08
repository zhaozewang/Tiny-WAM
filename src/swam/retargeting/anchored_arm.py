"""R1 arm kinematics solved backward from an immutable hand mount pose.

The arm base is floating and inferred, not a calibrated shared robot torso.
Native link dimensions and joint limits are retained.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from swam.r1 import SOURCE, origin_matrix


class WristAnchoredArm:
    def __init__(self, side, spacer=.012):
        self.side = side
        self.robot = ET.parse(SOURCE).getroot()
        self.joints, self.lower, self.upper = [], [], []
        for i in range(1,7):
            joint = self.robot.find(f"joint[@name='{side}_arm_joint{i}']")
            self.joints.append((joint.find('parent').get('link'),joint.find('child').get('link'),
                                origin_matrix(joint.find('origin')),
                                np.fromstring(joint.find('axis').get('xyz'),sep=' ')))
            self.lower.append(float(joint.find('limit').get('lower')))
            self.upper.append(float(joint.find('limit').get('upper')))
        self.lower,self.upper = np.array(self.lower),np.array(self.upper)
        self.mount = np.eye(4)
        self.mount[:3,:3] = Rotation.from_euler('y',np.pi/2).as_matrix()
        self.mount[0,3] = .03425+spacer
        self.spacer = spacer
        self.nodes = {}

    def backward(self, wrist, q):
        if np.asarray(q).shape != (6,):
            raise ValueError('Expected six arm joints')
        links = {self.side+'_arm_link6':wrist@np.linalg.inv(self.mount)}
        for (parent,child,origin,axis),angle in reversed(list(zip(self.joints,q))):
            local = origin.copy()
            local[:3,:3] @= Rotation.from_rotvec(axis*angle).as_matrix()
            links[parent] = links[child]@np.linalg.inv(local)
        return links

    def forward(self, base, q):
        links = {self.side+'_arm_base_link':base}
        for (parent,child,origin,axis),angle in zip(self.joints,q):
            local = origin.copy()
            local[:3,:3] @= Rotation.from_rotvec(axis*angle).as_matrix()
            links[child] = links[parent]@local
        return links[self.side+'_arm_link6']@self.mount

    def add_visuals(self, scene):
        import pyrender
        import trimesh
        # Retain the vendor's actual articulated geometry; match the accepted hand finish.
        for i in range(1,7):
            name = self.side+f'_arm_link{i}'
            visual = self.robot.find(f"link[@name='{name}']/visual")
            path = SOURCE.parent.parent/'meshes'/Path(visual.find('geometry/mesh').get('filename')).name
            mesh = trimesh.load(path,force='mesh',process=False)
            mesh.apply_transform(origin_matrix(visual.find('origin')))
            color = [.74,.77,.8,1.] if i in (2,3) else [.28,.31,.34,1.]
            material = pyrender.MetallicRoughnessMaterial(baseColorFactor=color,metallicFactor=.3,roughnessFactor=.6)
            self.nodes[name] = scene.add(pyrender.Mesh.from_trimesh(mesh,material=material))
        adapter = trimesh.creation.cylinder(radius=.024,height=self.spacer+.001,sections=48)
        transform = np.eye(4)
        transform[:3,:3] = self.mount[:3,:3]
        transform[0,3] = .03425+self.spacer/2
        adapter.apply_transform(transform)
        material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[.18,.2,.22,1],metallicFactor=.4,roughnessFactor=.5)
        self.nodes[self.side+'_adapter'] = scene.add(pyrender.Mesh.from_trimesh(adapter,material=material))

    def show(self, scene, wrist, q):
        links = self.backward(wrist,q)
        for name,node in self.nodes.items():
            scene.set_pose(node,links[self.side+'_arm_link6'] if name.endswith('_adapter') else links[name])
        return links
