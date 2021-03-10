"""
module imported from https://github.com/stanfordnlp/color-describer/blob/master/vectorizers.py
"""

import numpy as np
try:
    import theano.tensor as T
    import skimage.color
    from lasagne.layers import InputLayer, EmbeddingLayer, reshape
    from stanza.research.rng import get_rng
    rng = get_rng()
except ImportError:
    pass
from collections import Sequence


RANGES_RGB = (256.0, 256.0, 256.0)
RANGES_HSV = (361.0, 101.0, 101.0)
C_EPSILON = 1e-4


class ColorVectorizer(object):
    def vectorize_all(self, colors, hsv=None):
        '''
        :param colors: A sequence of length-3 vectors or 1D array-like objects containing
                      RGB coordinates in the range [0, 256).
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 360], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 256). `None` (default) means take the
                         color space from the value given to the constructor.
        :return np.ndarray: An array of the vectorized form of each color in `colors`
                            (first dimension is the index of the color in the `colors`).

        >>> BucketsVectorizer((2, 2, 2)).vectorize_all([(0, 0, 0), (255, 0, 0)])
        array([0, 4], dtype=int32)
        '''
        return np.array([self.vectorize(c, hsv=hsv) for c in colors])

    def unvectorize_all(self, colors, random=False, hsv=None):
        '''
        :param Sequence colors: An array or sequence of vectorized colors
        :param random: If true, sample a random color from each bucket. Otherwise,
                       return the center of the bucket. Some vectorizers map colors
                       one-to-one to vectorized versions; these vectorizers will
                       ignore the `random` argument.
        :param hsv: If `True`, return colors in HSV format; otherwise, RGB.
                    `None` (default) means take the color space from the value
                    given to the constructor.
        :return list(tuple(int)): The unvectorized version of each color in `colors`

        >>> BucketsVectorizer((2, 2, 2)).unvectorize_all([0, 4])
        [(64, 64, 64), (192, 64, 64)]
        >>> BucketsVectorizer((2, 2, 2)).unvectorize_all([0, 4], hsv=True)
        [(0, 0, 25), (0, 67, 75)]
        '''
        return [self.unvectorize(c, random=random, hsv=hsv) for c in colors]

    def visualize_distribution(self, dist):
        '''
        :param dist: A distribution over the buckets defined by this vectorizer
        :type dist: array-like with shape `(self.num_types,)``
        :return images: `list(`3-D `np.array` with `shape[2] == 3)`, three images
            with the last dimension being the channels (RGB) of cross-sections
            along each axis, showing the strength of the distribution as the
            intensity of the channel perpendicular to the cross-section.
        '''
        raise NotImplementedError

    def get_input_vars(self, id=None, recurrent=False):
        '''
        :param id: The string tag to use as a prefix in the variable names.
            If `None`, no prefix will be added. (Passing an empty string will
            result in adding a bare `'/'`, which is legal but probably not what
            you want.)
        :type id: str or None
        :param bool recurrent: If `True`, return input variables reflecting
            copying the input `k` times, where `k` is the recurrent sequence
            length. This means the input variables will have one more dimension
            than they would if they were input to a simple feed-forward layer.
        :return list(T.TensorVariable): The variables that should feed into the
            color component of the input layer of a neural network using this
            vectorizer.
        '''
        id_tag = (id + '/') if id else ''
        return [(T.itensor3 if recurrent else T.imatrix)(id_tag + 'colors')]

    def get_input_layer(self, input_vars, recurrent_length=0, cell_size=20, context_len=1, id=None):
        '''
        :param input_vars: The input variables returned from
            `get_input_vars`.
        :type input_vars: list(T.TensorVariable)
        :param recurrent_length: The number of steps to copy color representations
            for input to a recurrent unit. If `None`, allow variable lengths; if 0,
            produce output for a non-recurrent layer (this will create an input layer
            producing a tensor of rank one lower than the recurrent version).
        :type recurrent_length: int or None
        :param int cell_size: The number of dimensions of the final color representation.
        :param id: The string tag to use as a prefix in the layer names.
            If `None`, no prefix will be added. (Passing an empty string will
            result in adding a bare `'/'`, which is legal but probably not what
            you want.)
        :return Lasagne.Layer, list(Lasagne.Layer): The layer producing the color
            representation, and the list of input layers corresponding to each of
            the input variables (in the same order).
        '''
        raise NotImplementedError(self.get_input_layer)


class FourierVectorizer(ColorVectorizer):
    '''
    Vectorizes colors by converting them to a truncated frequency representation.
    This vectorizer can only vectorize, not unvectorize.
    '''
    def __init__(self, resolution, hsv=False):
        '''
        :param resolution: The number of dimensions to truncate the frequency
                           representation (the vectorized representation will be
                           *twice* this, because the frequency representation uses
                           complex numbers). Should be an even number between 0 and
                           the range of each internal color space dimension, or a
                           length-3 sequence of such numbers.
        :param bool hsv: If `True`, the internal color space used by the vectorizer
                         will be HSV. Input and output color spaces can be configured
                         on a per-call basis by using the `hsv` parameter of
                         `vectorize` and `unvectorize`.
        '''
        if len(resolution) == 1:
            resolution = resolution * 3
        self.resolution = resolution
        self.output_size = np.prod(resolution) * 2
        self.hsv = hsv

    def vectorize(self, color, hsv=None):
        '''
        :param color: An length-3 vector or 1D array-like object containing
                      color coordinates.
        :param bool hsv: If `True`, input is assumed to be in HSV space in the range
                         [0, 360], [0, 100], [0, 100]; if `False`, input should be in RGB
                         space in the range [0, 255]. `None` (default) means take the
                         color space from the value given to the constructor.
        :return np.ndarray: The color in the Fourier representation,
                            a vector of shape `(prod(resolution) * 2,)`.

        >>> normalize = lambda v: np.where(v.round(2) == 0.0, 0.0, v.round(2))
        >>> normalize(FourierVectorizer([2]).vectorize((255, 0, 0)))
        array([ 1.,  1.,  1.,  1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.], dtype=float32)
        >>> normalize(FourierVectorizer([2]).vectorize((180, 100, 100), hsv=True))
        array([ 1., -1., -1.,  1.,  1., -1., -1.,  1.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.], dtype=float32)
        >>> normalize(FourierVectorizer([2], hsv=True).vectorize((0, 100, 100)))
        array([ 1., -1., -1.,  1.,  1., -1., -1.,  1.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.], dtype=float32)
        >>> normalize(FourierVectorizer([2], hsv=True).vectorize((0, 255, 255), hsv=False))
        array([ 1., -1., -1.,  1., -1.,  1.,  1., -1.,  0.,  0.,  0.,  0.,  0.,
                0.,  0.,  0.], dtype=float32)
        '''
        return self.vectorize_all([color], hsv=hsv)[0]

    def vectorize_all(self, colors, hsv=None):
        '''
        >>> normalize = lambda v: np.where(v.round(2) == 0.0, 0.0, v.round(2))
        >>> normalize(FourierVectorizer([2]).vectorize_all([(255, 0, 0), (0, 255, 255)]))
        array([[ 1.,  1.,  1.,  1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,  0.,
                 0.,  0.,  0.],
               [ 1., -1., -1.,  1.,  1., -1., -1.,  1.,  0.,  0.,  0.,  0.,  0.,
                 0.,  0.,  0.]], dtype=float32)
        '''
        if hsv is None:
            hsv = self.hsv

        colors = np.array([colors])
        assert len(colors.shape) == 3, colors.shape
        assert colors.shape[2] == 3, colors.shape

        ranges = np.array(RANGES_HSV if self.hsv else RANGES_RGB)
        if hsv and not self.hsv:
            c_hsv = colors
            color_0_1 = skimage.color.hsv2rgb(c_hsv / (np.array(RANGES_HSV) - 1.0))
        elif not hsv and self.hsv:
            c_rgb = colors
            color_0_1 = skimage.color.rgb2hsv(c_rgb / (np.array(RANGES_RGB) - 1.0))
        else:
            color_0_1 = colors / (ranges - 1.0)

        # Using a Fourier representation causes colors at the boundary of the
        # space to behave as if the space is toroidal: red = 255 would be
        # about the same as red = 0. We don't want this...
        xyz = color_0_1[0] / 2.0
        if self.hsv:
            # ...*except* in the case of HSV: H is in fact a polar coordinate.
            xyz[:, 0] *= 2.0

        # ax, ay, az = [np.hstack([np.arange(0, g / 2), np.arange(r - g / 2, r)])
        #               for g, r in zip(self.resolution, ranges)]
        ax, ay, az = [np.arange(0, g) for g, r in zip(self.resolution, ranges)]
        gx, gy, gz = np.meshgrid(ax, ay, az)

        arg = (np.multiply.outer(xyz[:, 0], gx) +
               np.multiply.outer(xyz[:, 1], gy) +
               np.multiply.outer(xyz[:, 2], gz))
        assert arg.shape == (xyz.shape[0],) + tuple(self.resolution), arg.shape
        repr_complex = np.exp(-2j * np.pi * (arg % 1.0)).swapaxes(1, 2).reshape((xyz.shape[0], -1))
        result = np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)
        return result

    def unvectorize(self, color, random='ignored', hsv=None):
        # Exact unvectorization for the frequency distribution is impossible
        # unless the representation is not truncated. For now this should
        # just be a speaker representation.
        raise NotImplementedError

    def get_input_vars(self, id=None, recurrent=False):
        id_tag = (id + '/') if id else ''
        return [(T.tensor3 if recurrent else T.matrix)(id_tag + 'colors')]

    def get_input_layer(self, input_vars, recurrent_length=0, cell_size=20,
                        context_len=1, id=None):
        id_tag = (id + '/') if id else ''
        (input_var,) = input_vars
        shape = ((None, self.output_size * context_len)
                 if recurrent_length == 0 else
                 (None, recurrent_length, self.output_size * context_len))
        l_color = InputLayer(shape=shape, input_var=input_var,
                             name=id_tag + 'color_input')
        return l_color, [l_color]