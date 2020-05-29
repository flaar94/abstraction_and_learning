import itertools

import numpy as np

MAX_NLL = 10.

# NLL adjustments
SELECTOR_TARGET_REQUIRES_ENTITY_PENALTY = 1.
EMPTY_SELECT_PENALTY = 1.
MISMATCHED_OUTPUT_PENALTY = 10.
SELECTING_SAME_PROPERTY_BONUS = 1.
SELECT_NOT_0_NLL = np.log(2.)
NEW_COLOR_BONUS = 1.
SELECTOR_MAX_NLL_CORRECTION = np.log(2.)
DIM_CHANGE_PENALTY = np.log(10.)
SELECTOR_PAIR_PENALTY = np.log(2)
NONCONSTANT_BOOL_PENALTY = 2.

PROPERTY_CONSTRUCTION_COST = np.log(2)
SELECTOR_CONSTRUCTION_COST = np.log(2)
POINT_PROP_COST = 2.

# Feature flags
ALLOW_COMPOSITE_SELECTORS = False
USE_TRANSFORMER_NAMES = True
MAKE_PROPERTY_LIST = False

# Limiting constants
MAX_SMALL_TIME = 300.
MAX_LARGE_TIME = 300.
MAX_PREDICTORS = 300_000
MAX_PARTIAL_PREDICTORS = 3

# MAX NLL used to be the sixth element
CONSTANTS = (f"MAX_NLL = {MAX_NLL}",
             SELECTOR_TARGET_REQUIRES_ENTITY_PENALTY,
             EMPTY_SELECT_PENALTY,
             MISMATCHED_OUTPUT_PENALTY,
             ALLOW_COMPOSITE_SELECTORS,
             SELECTING_SAME_PROPERTY_BONUS,
             SELECT_NOT_0_NLL,
             NEW_COLOR_BONUS,
             SELECTOR_MAX_NLL_CORRECTION,
             DIM_CHANGE_PENALTY,
             SELECTOR_PAIR_PENALTY,
             PROPERTY_CONSTRUCTION_COST,
             POINT_PROP_COST
             )

CONSTANT_STRINGS = tuple((str(constant) for constant in CONSTANTS))

# All types of properties
# lines are of the form (float, UNION[0, 1, 0.5, -0.5]) where the first is the y intercept for the first (horizontal),
# and the x-intercept for the other three
TYPES = frozenset(
    {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity', 'shape', 'uncolored_shape', 'vector', 'color',
     'bool', 'point', 'line', 'group'})

STRAIGHT_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))
ALL_DIRECTIONS = tuple(itertools.product([1, 0, -1], [1, 0, -1]))

LINE_MAPPING = {0: 'horizontal', 1: 'vertical', 0.5: 'forward diagonal', -0.5: 'backward diagonal'}
FORWARD_DIAGONAL = 0.5
BACKWARD_DIAGONAL = -0.5
