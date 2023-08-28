
# import numpy as np


# class IndexJuggler:
#     @staticmethod
#     def merge_adjacent_axes(array, ax1, ax2):
#         """Multiplies out two axes to generate reduced order array."""
#         shape = array.shape
#         sz1, sz2 = shape[ax1], shape[ax2]
#         new_shape = shape[:ax1] + (sz1 * sz2,)
#         if ax2 == -1 or ax2 == len(shape): pass
#         else: new_shape = new_shape + shape[ax2 + 1:]
#         return array.reshape(new_shape)

#     @staticmethod
#     def merge_axes(array, ax1, ax2):
#         """Brings ax2 next to ax1 first, then combine the two axes into one."""
#         array2 = np.moveaxis(array, ax2, ax1 + 1)  # now, previously known as ax2 is located @ ax1 + 1
#         return IndexJuggler.merge_adjacent_axes(array2, ax1, ax1 + 1)

#     @staticmethod
#     def expand_axis(array, ax_idx, factor, curtail=False):
#         """opposite functionality of merge_axes.  While ``numpy.split`` requires the division number, this requies the split size."""
#         if curtail:  # if size at ax_idx doesn't divide evenly factor, it will be curtailed.
#             size_at_idx = array.shape[ax_idx]
#             extra = size_at_idx % factor
#             array = array[IndexJuggler.indexer(axis=ax_idx, myslice=slice(0, -extra))]
#         total_shape = list(array.shape)
#         for index, item in enumerate((int(total_shape.pop(ax_idx) / factor), factor)): total_shape.insert(ax_idx + index, item)
#         return array.reshape(tuple(total_shape))  # should be same as return __import__("numpy)s.plit(array, new_shape, ax_idx)

#     @staticmethod
#     def slicer(array, a_slice: slice, axis=0):
#         """Extends Numpy slicing by allowing rotation if index went beyond size."""
#         lower_, upper_ = a_slice.start, a_slice.stop
#         n = array.shape[axis]
#         lower_ = lower_ % n  # if negative, you get the positive equivalent. If > n, you get principal value.
#         roll = lower_
#         lower_, upper_ = lower_ - roll, upper_ - roll
#         array_ = np.roll(array, -roll, axis=axis)
#         upper_ = upper_ % n
#         new_slice = slice(lower_, upper_, a_slice.step)
#         return array_[IndexJuggler.indexer(axis=axis, myslice=new_slice, rank=array.ndim)]

#     @staticmethod
#     def indexer(axis, myslice, rank=None):
#         """Allows subseting an array of arbitrary shape, given console index to be subsetted and the range. Returns a tuple of slicers."""
#         if rank is None: rank = axis + 1
#         indices = [slice(None, None, None)] * rank  # slice(None, None, None) is equivalent to `:` `everything`
#         indices[axis] = myslice
#         # noinspection PyTypeChecker
#         indices.append(Ellipsis)  # never hurts to add this in the end.
#         return tuple(indices)
