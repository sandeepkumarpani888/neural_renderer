import chainer
import chainer.functions as cf
import chainer.functions.math.basic_math as cfmath

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


def lighting(
        faces, textures, intensity_ambient=0.5, intensity_directional=0.5, color_ambient=(1, 1, 1),
        color_directional=(1, 1, 1), direction=(0, 1, 0)):
    xp = chainer.cuda.get_array_module(faces)
    bs, nf = faces.shape[:2]

    # arguments
    if isinstance(color_ambient, tuple) or isinstance(color_ambient, list):
        color_ambient = xp.array(color_ambient, 'float32')
    if isinstance(color_directional, tuple) or isinstance(color_directional, list):
        color_directional = xp.array(color_directional, 'float32')
    if isinstance(direction, tuple) or isinstance(direction, list):
        direction = xp.array(direction, 'float32')
    if color_ambient.ndim == 1:
        color_ambient = cf.broadcast_to(color_ambient[None, :], (bs, 3))
    if color_directional.ndim == 1:
        color_directional = cf.broadcast_to(color_directional[None, :], (bs, 3))
    if direction.ndim == 1:
        direction = cf.broadcast_to(direction[None, :], (bs, 3))

    # create light
    light = xp.zeros((bs, nf, 3), 'float32')

    # ambient light
    if intensity_ambient != 0:
        light = light + intensity_ambient * cf.broadcast_to(color_ambient[:, None, :], light.shape)

    # directional light
    if intensity_directional != 0:
        faces = faces.reshape((bs * nf, 3, 3))
        v10 = faces[:, 0] - faces[:, 1]
        v12 = faces[:, 2] - faces[:, 1]
        normals = cf.normalize(cross(v10, v12))
        normals = normals.reshape((bs, nf, 3))

        if direction.ndim == 2:
            direction = cf.broadcast_to(direction[:, None, :], normals.shape)
        cos = cf.relu(cf.sum(normals * direction, axis=2))
        light = (
            light + intensity_directional * cfmath.mul(*cf.broadcast(color_directional[:, None, :], cos[:, :, None])))

    # apply
    light = cf.broadcast_to(light[:, :, None, None, None, :], textures.shape)
    textures = textures * light
    return textures
