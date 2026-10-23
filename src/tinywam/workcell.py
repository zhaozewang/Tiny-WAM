"""Static ALOHA table for the R1 scene."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MENAGERIE=ROOT/'assets/menagerie'


def spawn_workcell(stage, *, include_body=False):
    import trimesh
    import isaaclab.sim as sim
    from pxr import Gf,Sdf,UsdGeom,UsdShade,UsdPhysics
    material_colors={'metal':(.48,.52,.56),'black':(.025,.03,.04),'table':(.55,.37,.20)}
    materials={}
    for name,color in material_colors.items():
        cfg=sim.PreviewSurfaceCfg(diffuse_color=color,roughness=.55)
        cfg.func(f'/World/Materials/{name}',cfg)
        materials[name]=UsdShade.Material(stage.GetPrimAtPath(f'/World/Materials/{name}'))
    # Preserve ALOHA's original UV-mapped table texture.
    texture=UsdShade.Shader.Define(stage,'/World/Materials/table/texture')
    texture.CreateIdAttr('UsdUVTexture')
    texture.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(MENAGERIE/'aloha/assets/small_meta_table_diffuse.png'))
    texture.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
    uv=UsdShade.Shader.Define(stage,'/World/Materials/table/uv')
    uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    texture.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    surface=UsdShade.Shader(materials['table'].ComputeSurfaceSource()[0])
    surface.GetInput('diffuseColor').ConnectToSource(texture.ConnectableAPI(),'rgb')

    def mesh(path,file,transform,material):
        geom=trimesh.load(str(file),force='mesh',process=False)
        usd=UsdGeom.Mesh.Define(stage,path)
        usd.CreatePointsAttr(geom.vertices.astype('float32').tolist())
        usd.CreateFaceVertexCountsAttr([3]*len(geom.faces))
        usd.CreateFaceVertexIndicesAttr(geom.faces.reshape(-1).astype('int32').tolist())
        usd.CreateSubdivisionSchemeAttr('none')
        usd.CreateNormalsAttr(geom.vertex_normals.astype('float32').tolist());usd.SetNormalsInterpolation('vertex')
        usd.AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
        if hasattr(geom.visual,'uv') and geom.visual.uv is not None:
            UsdGeom.PrimvarsAPI(usd).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(geom.visual.uv.astype('float32').tolist())
        UsdShade.MaterialBindingAPI.Apply(usd.GetPrim()).Bind(materials[material])
        points=geom.vertices@transform[:3,:3].T+transform[:3,3]
        return [points.min(0).tolist(),points.max(0).tolist()]

    assert not include_body, 'R1 supplies the robot body'
    table_transform=np.eye(4);table_transform[:3,3]=[.77,0,0]
    table_bounds={}
    for name in ['tabletop','tablelegs']:
        table_bounds[name]=mesh('/World/Table/'+name,MENAGERIE/f'aloha/assets/{name}.obj',table_transform,'table')
    # Stationary triangle collision is appropriate for a fixed table.
    for name in ['tabletop','tablelegs']:
        prim=stage.GetPrimAtPath('/World/Table/'+name)
        UsdPhysics.CollisionAPI.Apply(prim);UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
    floor=sim.PreviewSurfaceCfg(diffuse_color=(.23,.25,.28),roughness=1.)
    floor.func('/World/Materials/floor',floor)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath('/World/ground')).Bind(
        UsdShade.Material(stage.GetPrimAtPath('/World/Materials/floor')),bindingStrength=UsdShade.Tokens.strongerThanDescendants)
    return dict(menagerie_commit='4d038b3feae26ec82b46a4d586379114012a8ac7',
                body='R1 body supplied by robot assembly',
                table='ALOHA tabletop/tablelegs OBJ + original diffuse texture; translated +0.77m in X',
                table_bounds=table_bounds)
