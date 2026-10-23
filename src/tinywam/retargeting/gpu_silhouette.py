"""Experimental OpenGL compute reduction, keeping the existing MSAA rasterizer.

Two coordinate-descent candidates can queue scoring before one tiny readback.
This batches scoring/readback, not triangle draw calls. No CUDA interop required.
"""
import numpy as np
import pyrender
from OpenGL import GL as gl
from OpenGL.GL.shaders import compileProgram, compileShader
from OpenGL.raw.GL.VERSION.GL_1_1 import glGenTextures
from .silhouette_renderer import _MaskReadbackRenderer


REDUCE = """
#version 430
layout(local_size_x=256) in;
layout(std430,binding=0) readonly buffer Inputs { double distances[]; };
layout(std430,binding=1) readonly buffer Flags { uint flags[]; };
layout(std430,binding=2) buffer Partials { dvec4 partials[]; };
uniform sampler2D depthTex;
uniform uint width;
uniform uint height;
shared dvec4 values[256];
void main() {
    uint i=gl_GlobalInvocationID.x, t=gl_LocalInvocationID.x;
    dvec4 v=dvec4(0);
    if(i<width*height) {
        bool target=(flags[i]&1u)!=0u;
        bool visible=(flags[i]&2u)!=0u &&
            texelFetch(depthTex,ivec2(i%width,height-1u-i/width),0).r!=1.0;
        v=dvec4(visible&&target ? 1.0 : 0.0,visible||target ? 1.0 : 0.0,
                visible ? 1.0 : 0.0,visible ? distances[i] : 0.0);
    }
    values[t]=v; barrier();
    for(uint s=128u;s>0u;s/=2u) { if(t<s) values[t]+=values[t+s]; barrier(); }
    if(t==0u) partials[gl_WorkGroupID.x]=values[0];
}
"""
FINISH = """
#version 430
layout(local_size_x=256) in;
layout(std430,binding=2) readonly buffer Partials { dvec4 partials[]; };
layout(std430,binding=3) buffer Results { dvec4 results[]; };
uniform uint groups;
uniform uint slot;
shared dvec4 values[256];
void main() {
    uint t=gl_LocalInvocationID.x;dvec4 v=dvec4(0);
    for(uint i=t;i<groups;i+=256u) v+=partials[i];
    values[t]=v; barrier();
    for(uint s=128u;s>0u;s/=2u) { if(t<s) values[t]+=values[t+s]; barrier(); }
    if(t==0u) results[slot]=values[0];
}
"""


class _ComputeRenderer(_MaskReadbackRenderer):
    score_slot = None
    resources = None

    def configure(self, outside, target, allowed):
        h,w=target.shape
        if (w,h)!=(self.viewport_width,self.viewport_height):
            raise ValueError('Objective dimensions must equal raster dimensions')
        if self.resources is None:
            programs=[compileProgram(compileShader(s,gl.GL_COMPUTE_SHADER)) for s in (REDUCE,FINISH)]
            buffers=gl.glGenBuffers(4)
            texture_ids=np.zeros(1,dtype=np.uint32)
            glGenTextures(1,texture_ids)
            texture=int(texture_ids[0]);fbo=gl.glGenFramebuffers(1)
            gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
            gl.glTexImage2D(gl.GL_TEXTURE_2D,0,gl.GL_DEPTH_COMPONENT24,w,h,0,gl.GL_DEPTH_COMPONENT,gl.GL_FLOAT,None)
            gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MIN_FILTER,gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D,gl.GL_TEXTURE_MAG_FILTER,gl.GL_NEAREST)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,fbo)
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER,gl.GL_DEPTH_ATTACHMENT,gl.GL_TEXTURE_2D,texture,0)
            gl.glDrawBuffer(gl.GL_NONE);gl.glReadBuffer(gl.GL_NONE)
            if gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)!=gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError('Compute depth framebuffer incomplete')
            self.resources=(programs,buffers,texture,fbo)
        programs,buffers,texture,fbo=self.resources
        groups=(w*h+255)//256
        flags=target.astype(np.uint32)+2*allowed.astype(np.uint32)
        arrays=[np.ascontiguousarray(np.minimum(outside,20),dtype=np.float64),flags,
                np.zeros((groups,4),np.float64),np.zeros((2,4),np.float64)]
        for buffer,array in zip(buffers,arrays):
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,buffer)
            gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER,array.nbytes,array,gl.GL_DYNAMIC_DRAW)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER,0)
        gl.glBindTexture(gl.GL_TEXTURE_2D,0)

    def _read_main_framebuffer(self,scene,flags):
        if self.score_slot is None or not flags & pyrender.RenderFlags.DEPTH_ONLY:
            return super()._read_main_framebuffer(scene,flags)
        if self.resources is None:raise RuntimeError('Configure scoring before rendering')
        programs,buffers,texture,fbo=self.resources
        w,h=self._main_fb_dims;groups=(w*h+255)//256
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER,self._main_fb_ms)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER,fbo)
        gl.glBlitFramebuffer(0,0,w,h,0,0,w,h,gl.GL_DEPTH_BUFFER_BIT,gl.GL_NEAREST)
        for index,buffer in enumerate(buffers):gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER,index,buffer)
        gl.glActiveTexture(gl.GL_TEXTURE0);gl.glBindTexture(gl.GL_TEXTURE_2D,texture)
        gl.glUseProgram(programs[0])
        gl.glUniform1i(gl.glGetUniformLocation(programs[0],'depthTex'),0)
        gl.glUniform1ui(gl.glGetUniformLocation(programs[0],'width'),w)
        gl.glUniform1ui(gl.glGetUniformLocation(programs[0],'height'),h)
        gl.glDispatchCompute(groups,1,1);gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
        gl.glUseProgram(programs[1])
        gl.glUniform1ui(gl.glGetUniformLocation(programs[1],'groups'),groups)
        gl.glUniform1ui(gl.glGetUniformLocation(programs[1],'slot'),self.score_slot)
        gl.glDispatchCompute(1,1,1)
        gl.glMemoryBarrier(gl.GL_BUFFER_UPDATE_BARRIER_BIT|gl.GL_SHADER_STORAGE_BARRIER_BIT)
        gl.glUseProgram(0);gl.glBindTexture(gl.GL_TEXTURE_2D,0)
        return self.score_slot

    def delete(self):
        if self.resources is not None:
            programs,buffers,texture,fbo=self.resources
            for p in programs:gl.glDeleteProgram(p)
            gl.glDeleteBuffers(4,buffers);gl.glDeleteTextures([texture]);gl.glDeleteFramebuffers(1,[fbo])
            self.resources=None
        super().delete()


class GpuSilhouetteRenderer(pyrender.OffscreenRenderer):
    def _create(self):
        super()._create()
        self._renderer.delete()
        self._renderer=_ComputeRenderer(self.viewport_width,self.viewport_height)

    def configure_score(self,outside,target,allowed):
        self._platform.make_current()
        try:self._renderer.configure(outside,target,allowed)
        finally:self._platform.make_uncurrent()

    def queue_score(self,scene,slot):
        if slot not in (0,1):raise ValueError('At most two queued candidates')
        self._renderer.score_slot=slot
        try:self.render(scene,flags=pyrender.RenderFlags.DEPTH_ONLY)
        finally:self._renderer.score_slot=None

    def read_scores(self,count):
        if count not in (1,2):raise ValueError('One or two candidates required')
        self._platform.make_current()
        try:
            gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER,self._renderer.resources[1][3])
            data=gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER,0,count*32)
            return np.frombuffer(data,dtype=np.float64).reshape(count,4).copy()
        finally:self._platform.make_uncurrent()
