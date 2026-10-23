"""Verify full continuous trajectories and build a three-view review video."""
import argparse,json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from tinywam.shards import atomic_json

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
report=[]
for folder in sorted((args.root/'staging').glob('episode_*')):
    complete=json.loads((folder/'complete.json').read_text())
    metadata=json.loads((folder/'metadata.json').read_text())
    assert complete['actions']==14400 and complete['observations']==14401
    assert not metadata.get('physics_only',False)
    assert metadata['reset_count']==1
    blocks=sorted(folder.glob('block_*.npz'));assert len(blocks)==120
    knots=np.asarray(metadata['policy']['knot_times']);targets=np.asarray(metadata['policy']['targets'])
    limits=np.asarray(metadata['limits']);max_excess=0.;max_speed=0.;prev=None
    lo=np.full(48,np.inf);hi=np.full(48,-np.inf);min_image_std=np.inf;temporal_change=np.zeros(3)
    for i,block in enumerate(blocks):
        with np.load(block) as source:
            d={key:source[key] for key in source.files}
            for key,value in d.items():
                if value.dtype.kind=="f":assert np.isfinite(value).all()
            assert d['rgb'].shape==(121,3,256,256,3)
            assert d['actions'].shape==(120,48)
            assert np.array_equal(d['frame_indices'],np.arange(i*120,i*120+121))
            if prev is not None:
                for k,v in prev.items():assert np.array_equal(v,d[k][0]),k
            prev={k:d[k][-1].copy() for k in ('qpos','qvel','rgb')}
            q=d['qpos'];a=d['actions']
            action_times=(np.arange(i*120,i*120+120)+1)/24
            indices=np.searchsorted(knots,action_times,side='right')-1
            phase=(action_times-knots[indices])/(knots[indices+1]-knots[indices])
            blend=phase*phase*(3-2*phase)
            expected=(targets[indices]*(1-blend[:,None])+targets[indices+1]*blend[:,None]).astype('float32')
            assert np.array_equal(a,expected), 'Recorded action differs from commanded policy'
            assert (a>=limits[:,0]-1e-5).all() and (a<=limits[:,1]+1e-5).all()
            max_excess=max(max_excess,float(np.maximum(limits[:,0]-q,q-limits[:,1]).max()))
            max_speed=max(max_speed,float(np.abs(d['qvel']).max()))
            lo=np.minimum(lo,q.min(0));hi=np.maximum(hi,q.max(0))
            temporal_change+=np.abs(np.diff(d['rgb'][::24].astype('float32'),axis=0)).mean(axis=(0,2,3,4))
            min_image_std=min(min_image_std,float(d['rgb'][::24].astype('float32').std(axis=(2,3)).mean(-1).min()))
    assert max_excess<.15 and max_speed<100 and min_image_std>2
    assert (temporal_change/len(blocks)).min()>.01, 'A camera stream appears frozen'
    assert (hi-lo).min()>.01, 'A joint did not move'
    report.append(dict(episode=complete['episode'],duration_s=600,blocks=120,
                       max_limit_excess_rad=max_excess,max_speed_rad_s=max_speed,
                       minimum_camera_pixel_std=min_image_std,camera_temporal_change=(temporal_change/len(blocks)).tolist(),joint_ranges_rad=(hi-lo).tolist()))
    if complete['episode']==0:
        review=args.root/'review';review.mkdir(exist_ok=True)
        with imageio.get_writer(review/'three_view_samples.mp4',fps=24,codec='libx264',quality=7,macro_block_size=1) as writer:
            for block_index in (0,24,60,96,119):
                with np.load(blocks[block_index]) as d:
                    for frame in range(120):
                        rgb=np.concatenate(d['rgb'][frame],axis=1)
                        canvas=Image.new('RGB',(768,288),'white');canvas.paste(Image.fromarray(rgb),(0,32))
                        draw=ImageDraw.Draw(canvas)
                        draw.text((8,8),f'Episode 0 | t={5*block_index+frame/24:.2f}s | head / left / right | actual simulation',fill='black')
                        writer.append_data(np.asarray(canvas))
                    canvas.save(review/f'frame_{block_index*5:03d}s.png')
atomic_json(args.root/'batch_review.json',dict(result='PASS',episodes=report))
print(json.dumps(dict(result='PASS',episodes=len(report))))
