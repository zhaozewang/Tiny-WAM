"""Exact-coverage readback for fitting: no metric depth conversion is needed.

Opt-in pyrender adapter. Rasterization, geometry, MSAA and final RGB stay intact.
DEPTH_ONLY returns a boolean hit mask, not metric depth; use only for silhouettes.
"""
import numpy as np
import pyrender
from pyrender.renderer import Renderer
from OpenGL.GL import (glBindFramebuffer,glBlitFramebuffer,glReadPixels,
                       GL_READ_FRAMEBUFFER,GL_DRAW_FRAMEBUFFER,GL_DEPTH_BUFFER_BIT,
                       GL_NEAREST,GL_DEPTH_COMPONENT,GL_FLOAT)


class _MaskReadbackRenderer(Renderer):
    def _read_main_framebuffer(self,scene,flags):
        if not flags & pyrender.RenderFlags.DEPTH_ONLY:
            return super()._read_main_framebuffer(scene,flags)
        width,height=self._main_fb_dims
        glBindFramebuffer(GL_READ_FRAMEBUFFER,self._main_fb_ms)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,self._main_fb)
        glBlitFramebuffer(0,0,width,height,0,0,width,height,GL_DEPTH_BUFFER_BIT,GL_NEAREST)
        glBindFramebuffer(GL_READ_FRAMEBUFFER,self._main_fb)
        buffer=glReadPixels(0,0,width,height,GL_DEPTH_COMPONENT,GL_FLOAT)
        return np.frombuffer(buffer,dtype=np.float32).reshape(height,width)[::-1] != 1.


class SilhouetteRenderer(pyrender.OffscreenRenderer):
    def _create(self):
        super()._create()
        if not self._platform.supports_framebuffers():
            raise RuntimeError('Silhouette readback requires framebuffer support')
        self._renderer.delete()
        self._renderer=_MaskReadbackRenderer(self.viewport_width,self.viewport_height)
