""" Definition of predict/invert functions as partials of the underlying code.

"""
from functools import partial
from arl.imaging.imaging_context import predict_context, invert_context

invert_2d = partial(invert_context, context='2d')
predict_2d = partial(predict_context, context='2d')

invert_facets = partial(invert_context, context='facets')
predict_facets = partial(predict_context, context='facets')

invert_facets_slice = partial(invert_context, context='facets_slice')
predict_facets_slice = partial(predict_context, context='facets_slice')

invert_facets_timeslice = partial(invert_context, context='facets_timeslice', remove=True)
predict_facets_timeslice = partial(predict_context, context='facets_timeslice', remove=True)

invert_facets_wprojection = partial(invert_context, context='facets', kernel='wprojection')
predict_facets_wprojection = partial(predict_context, context='facets', kernel='wprojection')

invert_facets_wstack = partial(invert_context, context='facets_wstack', remove=True)
predict_facets_wstack = partial(predict_context, context='facets_wstack', remove=True)

invert_slice = partial(invert_context, context='slice')
predict_slice = partial(predict_context, context='slice')

invert_timeslice = partial(invert_context, context='timeslice')
predict_timeslice = partial(predict_context, context='timeslice')

invert_timeslice_wprojection = partial(invert_context, context='timeslice', kernel='wprojection', remove=True)
predict_timeslice_wprojection = partial(predict_context, context='timeslice', kernel='wprojection', remove=True)

# timeslice+wstack is not possible because both require visibility iteration

invert_wprojection = partial(invert_context, context='2d', kernel='wprojection')
predict_wprojection = partial(predict_context, context='2d', kernel='wprojection')

invert_wprojection_wstack = partial(invert_context, context='wstack', kernel='wprojection', remove=True)
predict_wprojection_wstack = partial(predict_context, context='wstack', kernel='wprojection', remove=True)

invert_wstack = partial(invert_context, context='wstack')
predict_wstack = partial(predict_context, context='wstack')
