""" Definition of predict/invert functions in terms of the underlying functions:
:py:mod:`arl.imaging.imaging_context.predict_function` and :py:mod:`arl.imaging.imaging_context.invert_function`

"""
from functools import partial
from arl.imaging.imaging_context import predict_function, invert_function

invert_2d = partial(invert_function, context='2d')
predict_2d = partial(predict_function, context='2d')

invert_facets = partial(invert_function, context='facets')
predict_facets = partial(predict_function, context='facets')

invert_facets_slice = partial(invert_function, context='facets_slice')
predict_facets_slice = partial(predict_function, context='facets_slice')

invert_facets_timeslice = partial(invert_function, context='facets_timeslice', remove=True)
predict_facets_timeslice = partial(predict_function, context='facets_timeslice', remove=True)

invert_facets_wprojection = partial(invert_function, context='facets', kernel='wprojection')
predict_facets_wprojection = partial(predict_function, context='facets', kernel='wprojection')

invert_facets_wstack = partial(invert_function, context='facets_wstack')
predict_facets_wstack = partial(predict_function, context='facets_wstack', remove=True)

invert_slice = partial(invert_function, context='slice')
predict_slice = partial(predict_function, context='slice')

invert_timeslice = partial(invert_function, context='timeslice')
predict_timeslice = partial(predict_function, context='timeslice')

invert_timeslice_wprojection = partial(invert_function, context='timeslice', kernel='wprojection')
predict_timeslice_wprojection = partial(predict_function, context='timeslice', kernel='wprojection', remove=True)

# timeslice+wstack is not possible because both require visibility iteration

invert_wprojection = partial(invert_function, context='2d', kernel='wprojection')
predict_wprojection = partial(predict_function, context='2d', kernel='wprojection')

invert_wprojection_wstack = partial(invert_function, context='wstack', kernel='wprojection')
predict_wprojection_wstack = partial(predict_function, context='wstack', kernel='wprojection', remove=True)

invert_wstack = partial(invert_function, context='wstack')
predict_wstack = partial(predict_function, context='wstack')
