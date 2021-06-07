import chainer
import chainer.functions as cf

import cupy as cp
import numpy as np


class Cross(chainer.Function):
    def check_type_forward(self, in_types):
        chainer.utils.type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].ndim == 2,
            in_types[0].shape[1] == 3,
            in_types[1].dtype.kind == 'f',
            in_types[1].ndim == 2,
            in_types[1].shape[1] == 3,
            in_types[0].shape[0] == in_types[1].shape[0],
        )

    def forward_cpu(self, inputs):
        a, b = inputs
        c = np.cross(a, b)
        return c,

    def forward_gpu(self, inputs):
        a, b = inputs
        c = cp.zeros_like(a, 'float32')
        chainer.cuda.elementwise(
            'int32 j, raw T a, raw T b',
            'raw T c',
            '''
                float* ap = (float*)&a[j * 3];
                float* bp = (float*)&b[j * 3];
                float* cp = (float*)&c[j * 3];
                cp[0] = ap[1] * bp[2] - ap[2] * bp[1];
                cp[1] = ap[2] * bp[0] - ap[0] * bp[2];
                cp[2] = ap[0] * bp[1] - ap[1] * bp[0];
            ''',
            'function',
        )(
            cp.arange(a.size / 3).astype('int32'), a, b, c,
        )
        return c,

    def backward_cpu(self, inputs, gradients):
        a, b = inputs
        gc = gradients[0]
        ga = np.cross(b, gc)
        gb = np.cross(gc, a)
        return ga, gb

    def backward_gpu(self, inputs, gradients):
        a, b = inputs
        gc = gradients[0]
        ga = self.forward_gpu((b, gc))[0]
        gb = self.forward_gpu((gc, a))[0]
        return ga, gb


def cross(a, b):
    return Cross()(a, b)


def look_at(vertices, eye, at=None, up=None):
    """
    "Look at" transformation of vertices.
    """
    assert (vertices.ndim == 3)

    xp = chainer.cuda.get_array_module(vertices)
    batch_size = vertices.shape[0]
    if at is None:
        at = xp.array([0, 0, 0], 'float32')
    if up is None:
        up = xp.array([0, 1, 0], 'float32')

    if isinstance(eye, list) or isinstance(eye, tuple):
        eye = xp.array(eye, 'float32')
    if eye.ndim == 1:
        eye = cf.tile(eye[None, :], (batch_size, 1))
    if at.ndim == 1:
        at = cf.tile(at[None, :], (batch_size, 1))
    if up.ndim == 1:
        up = cf.tile(up[None, :], (batch_size, 1))

    # create new axes
    z_axis = cf.normalize(at - eye)
    x_axis = cf.normalize(cross(up, z_axis))
    y_axis = cf.normalize(cross(z_axis, x_axis))

    # create rotation matrix: [bs, 3, 3]
    r = cf.concat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), axis=1)
    if r.shape[0] != vertices.shape[0]:
        r = cf.broadcast_to(r, (vertices.shape[0], 3, 3))

    # apply
    # [bs, nv, 3] -> [bs, nv, 3] -> [bs, nv, 3]
    if vertices.shape != eye.shape:
        eye = cf.broadcast_to(eye[:, None, :], vertices.shape)
    vertices = vertices - eye
    vertices = cf.matmul(vertices, r, transb=True)

    return vertices
