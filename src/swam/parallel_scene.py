"""Candidate batched scene retaining baseline physics and camera settings."""
def make_scene(asset, height, count, device, partitions=False, solver_position_iterations=64, solver_velocity_iterations=8):
    import torch
    import isaaclab.sim as sim
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.sensors import TiledCamera, TiledCameraCfg
    from isaacsim.core.cloner import GridCloner
    from pxr import Sdf
    from swam.workcell import spawn_workcell
    from swam.preview import color_hands

    context = sim.SimulationContext(sim.SimulationCfg(dt=1 / 120, device=device))
    ground = sim.GroundPlaneCfg()
    ground.func('/World/ground', ground)
    light = sim.DomeLightCfg(intensity=1800)
    light.func('/World/light', light)
    cloner = GridCloner(spacing=8.)
    cloner.define_base_env('/World/envs')
    paths = cloner.generate_paths('/World/envs/env', count)
    context.stage.DefinePrim(paths[0], 'Xform')
    spawn_workcell(context.stage)
    layer = context.stage.GetRootLayer()
    Sdf.CopySpec(layer, '/World/Table', layer, paths[0] + '/Table')
    context.stage.RemovePrim('/World/Table')
    if partitions:
        # Visual backgrounds only: no collision or rigid-body schemas.
        for name, size, position in (
            ('back', (.08, 6., 3.), (-2.5, 0., 1.5)),
            ('front', (.08, 6., 3.), (3., 0., 1.5)),
            ('left', (5.58, .08, 3.), (.25, 3., 1.5)),
            ('right', (5.58, .08, 3.), (.25, -3., 1.5)),
        ):
            wall = sim.CuboidCfg(size=size, collision_props=None, rigid_props=None,
                visual_material=sim.PreviewSurfaceCfg(diffuse_color=(.55, .55, .55), roughness=1.))
            wall.func(paths[0] + '/Partitions/' + name, wall, translation=position)
    cfg = ArticulationCfg(prim_path='/World/envs/env_.*/Robot',
        spawn=sim.UsdFileCfg(usd_path=asset,
            articulation_props=sim.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=solver_position_iterations,
                solver_velocity_iteration_count=solver_velocity_iterations)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, height)),
        actuators={
            'arms': ImplicitActuatorCfg(joint_names_expr=['.*_arm_joint.*'], stiffness=1000., damping=80.),
            'hands': ImplicitActuatorCfg(joint_names_expr=['.*_hand_.*'], stiffness=3., damping=.1)})
    robot = Articulation(cfg)
    offsets = cloner.clone(paths[0], paths, replicate_physics=True, copy_from_source=True)
    cloner.filter_collisions(context.cfg.physics_prim_path, '/World/collisions', paths, ['/World/ground'])
    streams = []
    for name, focal in [('head', 18.), ('left', 28.), ('right', 28.)]:
        camera = TiledCamera(TiledCameraCfg(prim_path='/World/envs/env_.*/camera_' + name,
            height=512, width=512, data_types=['rgb'],
            spawn=sim.PinholeCameraCfg(focal_length=focal, horizontal_aperture=24.)))
        streams.append(camera)
    if partitions:
        from pxr import UsdPhysics, Usd
        for path in paths:
            for prim in Usd.PrimRange(context.stage.GetPrimAtPath(path + '/Partitions')):
                assert not prim.HasAPI(UsdPhysics.CollisionAPI), prim.GetPath()
                assert not prim.HasAPI(UsdPhysics.RigidBodyAPI), prim.GetPath()
    context.reset()
    color_hands(context.stage)
    assert robot.num_instances == count, (robot.num_instances, count)
    return context, robot, streams, torch.tensor(offsets, dtype=torch.float32, device=device)

