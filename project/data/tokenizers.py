import colorsys
import re
from ..utils.utils import START_SYMBOL, END_SYMBOL
from .vectorizers import FourierVectorizer


def tokenize_example(s):
    """
    Preprocess utterances used to describe a color by:
        _ splitting off punctuation and endings in -er, -est, -ish
        _ lowercasing
    :param s: string representing an utterance to describe a color.
    :return:
    """
    # split off punctuation and endings in -er, -est, -ish
    s_list = re.split('(\s|;|,|!|\?|\.|:|er|est|ish)', s)
    # remove empty strings an spaces
    s_list = [item for item in s_list if (item and item != ' ')]
    # lowercase words in s
    s_list = [item.lower() for item in s_list]
    s_list = [START_SYMBOL] + s_list + [END_SYMBOL]

    return s_list


def represent_color(color):
    """
    Represent an hsl normalized representation as the Fourier-transform method of its hsv normalized representation.
    :param color: hsl normalized representation, [h, s, l] where h, s and l are floats in [0, 1].
    :return:
    list of 54 floats obtained as the concatenation of the real and imaginary values of the 27 fourier transform of the
    permutations of the hsl normalized representation corresponding to color.
    """
    color_rgb_norm = colorsys.hls_to_rgb(color[0], color[2], color[1])
    color_hsv_norm = colorsys.rgb_to_hsv(*color_rgb_norm)
    color_hsv = (color_hsv_norm[0] * 360, color_hsv_norm[1] * 100, color_hsv_norm[2] * 100)
    t = FourierVectorizer([3], hsv=True)
    return t.vectorize(color_hsv)


def represent_color_context(colors):
    return [represent_color(color) for color in colors]
