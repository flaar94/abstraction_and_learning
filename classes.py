import itertools
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Union, Iterable

import numpy as np

from my_utils import display_case, ordinal
from constants import MISMATCHED_OUTPUT_PENALTY, USE_TRANSFORMER_NAMES, MAKE_PROPERTY_LIST, TYPES, EMPTY_SELECT_PENALTY, \
    LINE_MAPPING, POINT_PROP_COST
from nll_functions import combine_relation_selector_nll, combine_selector_nll, combine_move_nll, \
    combine_property_selector_nll, union_nlls, combine_pair_selector_nll


class Entity:
    """
    Properties:
        colors = subset of range(10)
        group = D4, Z4, Z2, Z2xZ2
    """
    all_entities = {}
    count = 0
    counts_to_entities = []

    def __init__(self, positions: dict, grid: tuple):
        self.positions = positions
        self.grid = grid
        self.shape_ = None
        self.colors_ = None
        self.uncolored_shape_ = None
        self.freeze_ = None
        self.count = self.__class__.count
        self.__class__.count += 1
        self.__class__.counts_to_entities.append(self.count)

    def __repr__(self):
        return f'Entity({self.positions})'

    def __eq__(self, other):
        return self.positions == other.positions and self.grid == other.grid

    def shape(self):
        if self.shape_ is None:
            shape_dict = {}
            center_0, center_1 = self.center(0), self.center(1)
            for position, color in self.positions.items():
                shape_dict[(position[0] - center_0, position[1] - center_1)] = color
            self.shape_ = frozenset(shape_dict.items())
        return self.shape_

    def uncolored_shape(self):
        if self.uncolored_shape_ is None:
            shape_set = set()
            center_0, center_1 = self.center(0), self.center(1)
            for position, color in self.positions.items():
                shape_set.add((position[0] - center_0, position[1] - center_1))
            self.uncolored_shape_ = frozenset(shape_set)
        return self.uncolored_shape_

    def colorblind_equals(self, other):
        return set(self.positions.keys()) == set(other.positions.keys())

    # def shape_equal(self, other):
    #     return set(self.positions.keys()) == set(other.positions.keys())

    def copy(self):
        return self.__class__(deepcopy(self.positions), self.grid)

    def change_grid(self, new_grid):
        new_positions = self.positions.copy()
        if new_grid and self.grid and (len(new_grid) != len(self.grid) or len(new_grid[0]) != len(self.grid[0])):
            for position, _ in self.positions.items():
                if position[0] >= len(new_grid) or position[1] >= len(new_grid[0]):
                    del new_positions[position]
        if not new_positions:
            return None
        return self.__class__.make(new_positions, new_grid)

    def colors(self):
        if self.colors_ is None:
            self.colors_ = frozenset(self.positions.values())
        return self.colors_

    def min_coord(self, axis=0):
        return int(min([key[axis] for key in self.positions.keys()]))

    def max_coord(self, axis=0):
        return int(max([key[axis] for key in self.positions.keys()]))

    def center(self, axis=0):
        output = (self.min_coord(axis) + self.max_coord(axis)) / 2.
        # if output.is_integer():
        #     output = int((self.min_coord(axis) + self.max_coord(axis))/2.)
        # else:
        #     output = float(output)
        return float(output)

    def num_points(self):
        return len(self.positions)

    def symmetry_group(self):
        pass

    def display(self):
        new_grid = np.array(self.grid)
        for i, j in itertools.product(range(new_grid.shape[0]), range(new_grid.shape[1])):
            if (i, j) not in set(self.positions):
                new_grid[i, j] = 10
        display_case(new_grid)

    def freeze(self):
        if self.freeze_ is None:
            self.freeze_ = frozenset(self.positions.items())
        return self.freeze_

    def is_a_rectangle(self):
        if len(self.positions) < 6:
            return False
        min_0, max_0 = self.min_coord(0), self.max_coord(0)
        min_1, max_1 = self.min_coord(1), self.max_coord(1)
        rectangle = {(sides_0, x) for x in range(min_1, max_1 + 1) for sides_0 in (min_0, max_0)}
        rectangle |= {(y, sides_1) for y in range(min_0 + 1, max_0) for sides_1 in (min_1, max_1)}
        return rectangle.issubset(self.positions.keys())

    def is_a_square(self):
        min_0, max_0 = self.min_coord(0), self.max_coord(0)
        min_1, max_1 = self.min_coord(1), self.max_coord(1)
        return max_0 - min_0 == max_1 - max_1 and self.is_a_rectangle()

    def is_a_line(self):
        return len(self.positions) > 2 and \
               (len({x for (x, y) in self.positions.keys()}) == 1 or len({y for (x, y) in self.positions.keys()}) == 1)

    @classmethod
    def make(cls, positions: dict, grid: tuple):
        key = frozenset(positions.items()), grid
        if key not in cls.all_entities:
            cls.all_entities[key] = cls(positions, grid)
        return cls.all_entities[key]

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.all_entities = {}
        cls.counts_to_entities = []

    @staticmethod
    def counts(entities):
        return frozenset((entity.count for entity in entities))

    @staticmethod
    def freeze_entities(entities):
        return frozenset((entity.freeze() for entity in entities))

    @staticmethod
    def shapes(entities):
        return Counter((entity.shape() for entity in entities))

    @staticmethod
    def uncolored_shapes(entities):
        return Counter((entity.uncolored_shape() for entity in entities))


class EntityFinder:
    def __init__(self, base_finder: callable, nll: float = 0., name: str = 'find all entities'):
        self.base_finder = base_finder
        self.nll = nll
        self.name = name
        self.cache = {}
        self.grid_distance_cache = {}

    def __call__(self, grid):
        if grid in self.cache:
            return self.cache[grid]
        out = self.base_finder(grid)
        self.cache[grid] = out
        return out

    def __str__(self):
        return self.name

    def grid_distance(self, grid1: tuple, grid2: tuple, shape_only=False) -> float:
        if not grid1 or not grid2:
            return float('inf')
        if (grid1, grid2) in self.grid_distance_cache:
            return self.grid_distance_cache[(grid1, grid2)]
        # print(f'len(cache) = {len(self.cache)}')
        entities1, entities2 = self(grid1), self(grid2)
        if shape_only:
            dist = 0
            for (i, entity1), (j, entity2) in itertools.product(enumerate(entities1), enumerate(entities2)):
                if i <= j and entity1.shape() == entity2.shape():
                    dist -= entity1.num_points()
        else:
            arr1, arr2 = np.array(grid1), np.array(grid2)
            if arr1.shape != arr2.shape:
                dist = max(arr1.shape[0] * arr1.shape[1], arr2.shape[0] * arr2.shape[1])
                # dist -= sum(np.nditer(arr1 == 0))
                # dist -= sum(np.nditer(arr2 == 0))
                # dist = np.abs(arr1.shape[0] - arr2.shape[0]) * np.abs(arr1.shape[1] - arr2.shape[1])
                # min_y = min(arr1.shape[0], arr2.shape[0])
                # min_x = min(arr1.shape[1], arr2.shape[1])
                # dist = sum(np.nditer(arr1[:min_y, :min_x] != arr2[:min_y, :min_x]))
                # dist += sum([arr.shape[0]*arr.shape[1] - min_y*min_x for arr in [arr1, arr2]])
                # dist = 1_000_000
                set1, set2 = {entity1.shape() for entity1 in entities1}, {entity2.shape() for entity2 in entities2}

                for shape in set1 & set2:
                    dist -= 0.5 * len(shape)
            else:
                dist = sum(np.nditer(arr1 != arr2))
                completely_the_sames = {frozenset(entity1.positions.items()) for entity1 in entities1} & {
                    frozenset(entity2.positions.items()) for entity2 in entities2}
                completely_the_sames = {frozenset((key for key, value in positions)) for positions in
                                        completely_the_sames}
                set1, set2 = {frozenset(entity1.positions.keys()) for entity1 in entities1}, {
                    frozenset(entity2.positions.keys()) for entity2 in entities2}
                for positions in (set1 & set2) - completely_the_sames:
                    dist -= 0.5 * len(positions)
        self.grid_distance_cache[(grid1, grid2)] = dist
        return dist

    def compose(self, selector: callable, composite: bool = False):
        return self.__class__(lambda grid: selector.select(self(grid), composite), self.nll + selector.nll,
                              name=f'{self.name} where {selector.name} {"as composite" if composite else ""}')

    def reset(self):
        self.cache = {}

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name


@dataclass
class OrdinalProperty:
    base_property: callable
    nll: float
    name: str
    input_types: frozenset = frozenset({})

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.base_property(*args, **kwargs)

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name


def nth_ordered(lst, n=0, use_max=True):
    if None in lst:
        return None
    ordered_list = list(sorted(lst, reverse=True))
    # Filter out repeats
    ordered_list = [element for i, element in enumerate(ordered_list) if i == 0 or element != ordered_list[i - 1]]
    if len(ordered_list) > n:
        return ordered_list[n] if use_max else ordered_list[-1 - n]
    else:
        return None


def pick_the_unique_value(lst):
    if None in lst:
        return None
    value_count = Counter(lst)
    output = None
    for value, count in value_count.items():
        if count == 1:
            if output is None:
                output = value
            else:
                return None
    return output


def my_sum(lst: list, n: int):
    if None in lst or lst == []:
        return None
    try:
        return sum(lst)
    except TypeError:
        return None


SINGLE_VALUE = OrdinalProperty(lambda x: next(iter(x)) if len(x) == 1 else None,
                               nll=0,
                               name=f'the single element',
                               input_types=TYPES)
# 45
ORDINAL_PROPERTIES = [SINGLE_VALUE]

# 46 - 51
ORDINAL_PROPERTIES.extend([OrdinalProperty(lambda x, n=n: nth_ordered(x, n, use_max=True),
                                           nll=n * np.log(2),
                                           name=f'take the {ordinal(n + 1)} largest',
                                           input_types=frozenset(
                                               {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity'}))
                           for n in range(6)])
# 52 - 57
ORDINAL_PROPERTIES.extend([OrdinalProperty(lambda x, n=n: nth_ordered(x, n, use_max=False),
                                           nll=n * np.log(2),
                                           name=f'take the {ordinal(n + 1)} smallest',
                                           input_types=frozenset(
                                               {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity'}))
                           for n in range(6)])

# 57.5
ORDINAL_PROPERTIES.append(OrdinalProperty(lambda x: pick_the_unique_value(x),
                                          nll=np.log(2),
                                          name=f'take the value that is unique',
                                          input_types=TYPES))

# 57.75
ORDINAL_PROPERTIES.append(OrdinalProperty(lambda x: any(x),
                                          nll=np.log(2),
                                          name=f'at least one is true of',
                                          input_types=frozenset({'bool'})))

ORDINAL_PROPERTIES.sort()


@dataclass
class PropertyInput:
    """
    A dataclass to help with standardization and readability
    """
    entity: Optional[Entity]
    entities: Optional[list]
    grid: Optional[tuple]

    def __init__(self, entity: Optional[Entity] = None,
                 entities: Optional[list] = None,
                 grid: Optional[tuple] = None):
        assert entity is None or isinstance(entity, Entity)
        self.entity = entity
        self.entities = entities
        self.grid = grid


class Property:
    count = 0
    if MAKE_PROPERTY_LIST:
        property_list = []
    of_type = {typ: [] for typ in TYPES}
    signatures_to_output_types = defaultdict(set)
    signatures_to_best_estimator = {}

    def __init__(self, prop: callable, nll: float, entity_finder: callable,
                 count=None, name=None, output_types: Optional[frozenset] = None, select=lambda x: x,
                 requires_entity=False,
                 is_constant=False, register=False, associated_objects=None):
        """

        :param prop: (entity, entities, grid) -> value
        :param nll:
        :param entity_finder:
        :param count:
        :param name:
        :param associated_objects: List of other properties that this property is derived from
        :param output_types: used to make combining different types of data more unlikely. Allowable inputs: 'color',
         'x_coordinate', 'y_coordinate', 'x_length', 'y_length', 'quantity'
        """
        self.prop = prop
        self.nll = nll
        self.entity_finder = entity_finder
        self.select = select
        self.requires_entity = requires_entity
        if name is None:
            name = f'Property({self.prop}, {self.nll}, {self.count}}})'
        self.name = name
        if output_types is None:
            output_types = TYPES
        self.output_types = output_types
        self.is_constant = is_constant
        self.associated_objects = associated_objects if associated_objects is not None else []
        self.cache = {}
        if register:
            self.count = count if count is not None else self.__class__.count
            self.__class__.count += 1
            if MAKE_PROPERTY_LIST:
                self.__class__.property_list.append(self)
            for output_type in output_types:
                if output_type in self.__class__.of_type:
                    self.__class__.of_type[output_type].append(self)
                else:
                    self.__class__.of_type[output_type] = [self]
        else:
            self.count = -1

    def __call__(self, entity: Optional[Entity], grid: Optional[tuple]):
        key = (entity.count if entity is not None else None, grid)
        if key in self.cache:
            return self.cache[key]
        output = self.prop(PropertyInput(entity, self.select(self.entity_finder(grid)), grid))
        self.cache[key] = output
        return output

    def register(self):
        self.count = self.__class__.count
        self.__class__.count += 1
        if MAKE_PROPERTY_LIST:
            self.__class__.property_list.append(self)
        for output_type in self.output_types:
            if output_type in self.__class__.of_type:
                self.__class__.of_type[output_type].append(self)
            else:
                self.__class__.of_type[output_type] = [self]

    def generate_output_signature(self, task):
        inputs = [case['input'] for case in task['train']] + [case['input'] for case in task['test']]
        if not self.requires_entity:
            signature = tuple((self(None, inp) for inp in inputs))
        else:
            signature = tuple(
                (frozenset({(entity.count, self(entity, inp)) for entity in self.entity_finder(inp)}) for inp in
                 inputs))
        return signature

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.count < other.count

    def __repr__(self):
        return f'Property({self.prop}, {self.nll}, {self.count}}})'

    def __eq__(self, other):
        return str(self) == str(other)

    def __str__(self):
        return self.name

    def add_selector(self, selector, ordinal_prop=SINGLE_VALUE, register=False):
        """
        Combines with a selector to generate a unique grid property

        :param selector:
        :param ordinal_prop: A function that takes a set, and returns a single value
        :param register: Determines if the property is automatically added to the list of properties "of_type"
        :return:
        """
        assert self.requires_entity
        new_nll = combine_property_selector_nll(self, selector, ordinal_prop)

        # if not (self.output_types & ordinal_prop.input_types):
        #     new_nll += MISMATCHED_OUTPUT_PENALTY

        def new_base_property(prop_input: PropertyInput):
            set_of_props = {self(entity, prop_input.grid)
                            for entity in selector.select(prop_input.entities)}
            return ordinal_prop(set_of_props)

        return self.__class__(prop=new_base_property,
                              nll=new_nll,
                              entity_finder=self.entity_finder,
                              output_types=self.output_types,
                              name=f'({self.entity_finder.name}) where ({selector.name}) then find the ({ordinal_prop})'
                                   f' of {self.name}',
                              register=register,
                              requires_entity=False,
                              associated_objects=[self])

    def create_distance_property(self, other, register=False, nll_red=0.):
        output_types = set()
        if 'x_coordinate' in self.output_types and 'x_coordinate' in other.output_types:
            output_types.add('x_length')
        if 'y_coordinate' in self.output_types and 'y_coordinate' in other.output_types:
            output_types.add('y_length')
        return self.__class__(prop=lambda x: float(self(x.entity, x.grid) - other(x.entity, x.grid))
        if self(x.entity, x.grid) is not None and other(x.entity, x.grid) is not None else None,
                              nll=self.nll + other.nll - nll_red,
                              entity_finder=self.entity_finder,
                              output_types=frozenset(output_types),
                              name=f'({self}) - ({other})',
                              register=register,
                              requires_entity=self.requires_entity or other.requires_entity)

    def coord_to_line_property(self, typ, register=False, nll_red=0.):
        """
        Creates a constant-type line property out of a coordinate property

        :param typ: 0., 1., 0.5, -0.5 corresponding to horizontal, vertical, forward diagonal, and backward diagonal line
        :param register: Determines if the property is automatically added to the list of properties "of_type"
        :param nll_red: Additional nll adjustment if necessary
        :return: the new property
        """
        return self.__class__(prop=lambda x: (float(self(x.entity, x.grid)), float(typ))
        if self(x.entity, x.grid) is not None else None,
                              nll=self.nll - nll_red + np.log(2),
                              entity_finder=self.entity_finder,
                              output_types=frozenset({'line'}),
                              name=f'the {LINE_MAPPING[typ]} line at position ({self})',
                              register=register,
                              requires_entity=self.requires_entity)

    def validate_and_register(self, task: dict, extra_validation: callable = lambda x: True) -> bool:
        output_signature = self.generate_output_signature(task)
        if self.requires_entity or (None not in output_signature and extra_validation(output_signature)):

            if output_signature in self.__class__.signatures_to_best_estimator:
                combined_nll = union_nlls(self.nll, self.__class__.signatures_to_best_estimator[output_signature].nll)
            else:
                combined_nll = self.nll
            if output_signature not in self.__class__.signatures_to_best_estimator or \
                    self.__class__.signatures_to_best_estimator[output_signature].nll > self.nll:
                self.__class__.signatures_to_best_estimator[output_signature] = self
            self.__class__.signatures_to_best_estimator[output_signature].nll = combined_nll

            if not self.output_types.issubset(
                    self.__class__.signatures_to_output_types[output_signature]):
                self.__class__.signatures_to_output_types[output_signature] |= self.output_types

                self.register()
                return True
        return False

    @classmethod
    def create_point_property(cls, first, second, register=False, nll_red=0.):
        return cls(prop=lambda x: (float(first(x.entity, x.grid)), float(second(x.entity, x.grid)))
        if first(x.entity, x.grid) is not None and second(x.entity, x.grid) is not None else None,
                   nll=first.nll + second.nll - nll_red + POINT_PROP_COST,
                   entity_finder=first.entity_finder,
                   output_types=frozenset({'point'}),
                   name=f'point (({first}), ({second}))',
                   register=register,
                   requires_entity=first.requires_entity or second.requires_entity)

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.property_list = []
        for key, value in cls.of_type.items():
            cls.of_type[key] = []
        cls.signatures_to_output_types = defaultdict(set)
        cls.signatures_to_best_estimator = {}

    @classmethod
    def from_relation_selector(cls, relation, selector, entity_finder,
                               ordinal_property: OrdinalProperty = SINGLE_VALUE,
                               register=True):
        new_nll = combine_relation_selector_nll(relation, selector, ordinal_property)

        def base_property(prop_input: PropertyInput):
            relations = [relation(prop_input.entity, selected_entity) for selected_entity in
                         selector.select(prop_input.entities)]
            return ordinal_property(relations) if ordinal_property(relations) is not None else None

        return cls(prop=base_property,
                   nll=new_nll,
                   entity_finder=entity_finder,
                   output_types=relation.output_types,
                   name=f'({ordinal_property.name}) of ({relation.name}) all entities where ({selector.name})',
                   register=register,
                   requires_entity=True)

    @classmethod
    def from_entity_prop_and_ordinal(cls, entity_prop, ordinal_property: OrdinalProperty = SINGLE_VALUE):
        nll = entity_prop.nll + ordinal_property.nll
        if not (entity_prop.output_types & ordinal_property.input_types):
            nll += MISMATCHED_OUTPUT_PENALTY

        def base_property(prop_input):
            prop_values = {entity_prop(entity, prop_input.grid) for entity in prop_input.entities}
            return ordinal_property(prop_values)

        return cls(base_property,
                   nll=nll,
                   entity_finder=entity_prop.entity_finder,
                   name=f'the ({ordinal_property}) of ({entity_prop})',
                   output_types=entity_prop.output_types,
                   requires_entity=False)

    @classmethod
    def xy_length_to_vector(cls, vert_prop, horiz_prop, register=False):
        return cls(lambda x: (
            vert_prop(x.entity, x.grid), horiz_prop(x.entity, x.grid)),
                   nll=combine_move_nll(vert_prop, horiz_prop),
                   entity_finder=vert_prop.entity_finder,
                   output_types=frozenset({'vector'}),
                   name=f'vertically ({vert_prop}) and horizontally ({horiz_prop})',
                   register=register,
                   requires_entity=vert_prop.requires_entity or horiz_prop.requires_entity)

    @classmethod
    def points_to_vector(cls, source_pt, target_pt, register=False):
        assert source_pt.entity_finder == target_pt.entity_finder
        return cls(lambda x: tuple((
            target_pt(x.entity, x.grid)[i] - source_pt(x.entity, x.grid)[i]
            for i in range(2)
        )) if target_pt(x.entity, x.grid) is not None and source_pt(x.entity, x.grid) is not None else None,
                   nll=source_pt.nll + target_pt.nll + np.log(2),
                   entity_finder=source_pt.entity_finder,
                   output_types=frozenset({'vector'}),
                   name=f'the vector from {source_pt} to {target_pt}',
                   register=register,
                   requires_entity=source_pt.requires_entity or target_pt.requires_entity)

    @classmethod
    def length_to_vector(cls, length_prop, axis, register=False):
        return cls(lambda x: tuple((
            (float(length_prop(x.entity, x.grid))) if i == axis else 0.
            for i in range(2)
        )) if length_prop(x.entity, x.grid) is not None else None,
                   nll=length_prop.nll + np.log(2),
                   entity_finder=length_prop.entity_finder,
                   output_types=frozenset({'vector'}),
                   name=f'the vector in the direction of axis={axis} with length={length_prop}',
                   register=register,
                   requires_entity=length_prop.requires_entity)


class Relation:
    """
    This is a raw relation_or_property which doesn't depend on the grid or entity finder in order to allow for relations
    between entities from different examples

    :param base_relation: (entity1, entity2) -> bool
    """
    count = 0
    relations = []

    def __init__(self, base_relation: callable, nll: float, name: str, output_types: frozenset):
        self.base_relation = base_relation
        self.nll = nll
        self.name = name
        self.output_types = output_types
        self.cache = {}
        self.count = self.__class__.count
        self.__class__.count += 1
        self.__class__.relations.append(self)

    def __call__(self, entity1: Entity, entity2: Entity) -> bool:
        # key = (entity1.freeze(), entity2.freeze())
        # if key in self.cache:
        #     return self.cache[key]
        out = self.base_relation(entity1, entity2)
        # self.cache[key] = out
        return out

    def __str__(self):
        return self.name

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.relations = []

    @classmethod
    def from_coordinate_properties(cls, property1: Property, property2: Property, reverse=False):
        sign = -1 if reverse else 1
        return cls(
            lambda entity1, entity2: sign * (property1(entity1, entity1.grid) - property2(entity2, entity2.grid)),
            nll=property1.nll + property2.nll,
            name=f'({property1}) minus ({property2})' if reverse else
            f'({property2}) minus ({property1})',
            output_types=frozenset({'vector'}))


class Selector:
    count = 0
    previous_selectors = set()
    output_signatures = {}

    def __init__(self, restriction: callable, nll=np.log(2), count=None, name=None, fixed_properties=None):
        if fixed_properties is None:
            fixed_properties = []
        if name is None:
            name = f'(Selector, restriction = {restriction}, NLL={nll})'
        self.restriction = restriction
        self.nll = nll
        self.name = name
        self.cache = {}
        self.fixed_properties = fixed_properties
        self.count = count if count is not None else self.__class__.count
        self.__class__.count += 1

    def __call__(self, entity, entities):
        return self.restriction(entity, entities)

    def select(self, entities: list, composite: bool = False):
        key = (Entity.counts(entities), composite)
        if key in self.cache:
            return self.cache[key]
        output = [entity for entity in entities if self(entity, entities)]
        if composite and output:
            new_dict = {position: color for entity in output for position, color in entity.positions.items()}
            composite_entity = Entity.make(new_dict, output[0].grid)
            output = [composite_entity]
        self.cache[key] = output
        return output

    def __str__(self):
        return self.name

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.count < other.count

    def generate_output_signature(self, base_entity_finder, task, output_raw_entities=False):
        inputs = [case['input'] for case in task['train']] + [case['input'] for case in task['test']]
        entities_list = [base_entity_finder(inp) for inp in inputs]
        if not output_raw_entities:
            return tuple((Entity.counts(self.select(entities)) for entities in entities_list))
        else:
            return tuple((Entity.counts(self.select(entities)) for entities in entities_list)), tuple((Entity.counts(entities) for entities in entities_list))

    def intersect(self, other):
        new_selector = self.__class__(lambda entity, entities: other(entity, entities) & self(entity, entities),
                                      nll=combine_pair_selector_nll(self, other),
                                      name=f'({self.name}) and ({other.name})')
        return new_selector

    def validate_and_register(self, task: dict, base_entity_finder, max_nll,
                              extra_validation: callable = lambda x: True) -> bool:
        output_signature, raw_signature = self.generate_output_signature(base_entity_finder, task, output_raw_entities=True)
        empty_selections = len([0 for x in output_signature if x == frozenset()])
        self.nll += empty_selections * EMPTY_SELECT_PENALTY

        for training_case, raw_case in zip(output_signature, raw_signature):
            if not training_case:
                self.nll += 10

            if self.name != 'true' and training_case == raw_case:
                # Add penalties for each training task that isn't affected by this selection
                self.nll += 2

        if self.nll > max_nll or str(self) in Selector.previous_selectors or not extra_validation(output_signature):
            return False
        if output_signature in Selector.output_signatures:
            Selector.output_signatures[output_signature].nll = \
                union_nlls(self.nll, Selector.output_signatures[output_signature].nll)
            Selector.previous_selectors.add(str(self))
            return False
        else:
            Selector.output_signatures[output_signature] = self
            Selector.previous_selectors.add(str(self))
            return True

    @classmethod
    def reset(cls):
        cls.count = 0
        cls.previous_selectors = set()
        cls.output_signatures = {}

    @classmethod
    def make_property_selector(cls, entity_property, target_property, the_same=True):
        def base_selector(entity, _):
            return (entity_property(entity, entity.grid) == target_property(entity, entity.grid)) == the_same

        new_nll = combine_selector_nll(entity_property, target_property)
        fixed_properties = [entity_property.count]
        prop_selector = cls(base_selector,
                            nll=new_nll,
                            name=f"({entity_property}) is "
                                 f"{'equal' if the_same else 'not equal'} to ({target_property})",
                            fixed_properties=fixed_properties)
        prop_selector.entity_prop = entity_property
        prop_selector.target_prop = target_property
        return prop_selector


class Transformer:
    """
    :param base_transform: A function that takes a set of entities and a grid, and applies a transformation to them all
    """

    def __init__(self, base_transform: callable, nll=np.log(2), name='<no name>', **kwargs):
        self.base_transform = base_transform
        self.nll = nll
        self.name = name if USE_TRANSFORMER_NAMES else ''
        self.kwargs = kwargs
        self.cache = {}

    def transform(self, entities: list, grid: Optional[tuple] = None):
        if len(entities) > 0:
            key = (Entity.counts(entities), entities[0].grid)
        else:
            return {}, ()
        if key in self.cache:
            return self.cache[key]

        new_entities, new_grid = self.base_transform(entities, grid, **self.kwargs)
        self.cache[key] = (new_entities, new_grid)
        return new_entities, new_grid

    def __str__(self):
        return self.name

    def compose(self, other):
        return self.__class__(lambda entities, grid: other.transform(*self.transform(entities, grid)),
                              nll=self.nll + other.nll,
                              name=f'({self.name}) then also ({other.name})')

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name

    # def create_entity_transformer(self, leaving_color=0,
    #                               entering_color_map=lambda orig_color, entity_color: entity_color, copy=False,
    #              extend_grid=True, nll=True, **property_kwargs):
    #     def base_transformer(entities: list, grid: Optional[tuple] = None):
    #         if len(entities) > 0:
    #             grid = entities[0].grid
    #             new_grid = np.array(grid)
    #         else:
    #             return {}, ()
    #         # new_grid = np.array(grid)
    #         new_entities = []
    #         new_positions_list = []
    #         for entity in entities:
    #             property_values = {key: prop(entity, grid) for key, prop in property_kwargs.items()}
    #             if None in property_values:
    #                 return {}, to_tuple(np.full_like(new_grid, 10))
    #             if not copy:
    #                 for position, color in entity.positions.items():
    #                     new_grid[position[0], position[1]] = leaving_color
    #             new_positions = move_entity(entity, grid, entering_color_map, extend_grid=extend_grid,
    #                                           **property_kwargs)
    #             if extend_grid and new_positions:
    #                 # First we compute the new shape of the grid
    #                 max_coordinates = [max((position[i] for position in new_positions.keys())) for i in range(2)]
    #                 positives = [max(max_coordinate, original_max - 1) + 1 for max_coordinate, original_max in
    #                              zip(max_coordinates, new_grid.shape)]
    #                 if tuple(positives) != new_grid.shape:
    #                     extended_grid = np.zeros(positives)
    #                     extended_grid[:new_grid.shape[0], :new_grid.shape[1]] = new_grid
    #                     new_grid = extended_grid
    #             for position, color in new_positions.items():
    #                 new_grid[position[0], position[1]] = new_positions[position[0], position[1]]
    #             new_positions_list.append(new_positions)
    #         new_grid_tuple = to_tuple(new_grid)
    #         new_entities = [Entity.make(new_positions, new_grid_tuple) for new_positions in new_positions_list]
    #         return new_entities, new_grid_tuple
    #     return Transformer(base_transformer, nll=nll)


@dataclass
class TransformerList:
    transformers: list
    nll: float

    @property
    def name(self):
        return ', '.join([str(transformer) for transformer in self.transformers])

    def __iter__(self):
        return iter(self.transformers)

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.name < other.name


class Predictor:
    count = 0

    def __init__(self, entity_finder: Union[EntityFinder, Iterable], transformer: Union[Transformer, Iterable],
                 nll=None, parallel=True):
        """

        :param entity_finder: A EntityFinder or iterable of EntityFinders that selects the desired entities
        :param transformer: A Transformer of iterable of Transformers that transforms the selected entities
        :param nll: The NLL of the predictor. If None, this will be automatically calculated
        :param parallel: Determines if the entity-transformer pairs are done in parallel or in sequence
        """
        self.parallel = parallel
        if isinstance(transformer, Transformer):
            self.transformers = (transformer,)
        else:
            self.transformers = tuple(transformer)
        if isinstance(entity_finder, EntityFinder):
            self.entity_finders = tuple((entity_finder for _ in self.transformers))
        else:
            self.entity_finders = tuple(entity_finder)
        if len(self.transformers) != len(self.entity_finders):
            raise Exception(
                f'number of entity finders ({len(self.entity_finders)}) '
                f'and transformers ({len(self.transformers)}) unequal')
        unique_entity_finders = {(entity_finder.name, entity_finder.nll) for entity_finder in self.entity_finders}
        unique_transformers = {(transformer.name, transformer.nll) for transformer in self.transformers}
        self.nll = sum([entity_finder_nll for entity_finder_name, entity_finder_nll in unique_entity_finders]) + \
                   sum([transformer_nll for transformer_name, transformer_nll in
                        unique_transformers]) if nll is None else nll
        self.net_distance_ = None
        # self.cache = {}
        self.count = self.__class__.count
        self.__class__.count += 1

    def predict(self, grid):
        out = grid
        # if grid in self.cache:
        #     return self.cache[grid]
        if self.parallel:
            selected_entities_list = [entity_finder(out) for entity_finder in self.entity_finders]
            for selected_entities, transformer in zip(selected_entities_list, self.transformers):
                edited_entities = []
                for entity in selected_entities:
                    new_entity = entity.change_grid(new_grid=out)
                    if new_entity is not None:
                        edited_entities.append(new_entity)
                _, out = transformer.transform(edited_entities)
        else:
            for entity_finder, transformer in zip(self.entity_finders, self.transformers):
                selected_entities = entity_finder(out)
                _, out = transformer.transform(selected_entities)
        # self.cache[grid] = out
        return out

    def __str__(self):
        return f"first ({[str(entity_finder) for entity_finder in self.entity_finders]}) then " \
               f"({[str(transformer) for transformer in self.transformers]}) " \
               f"{'in parallel' if self.parallel else 'sequentially'}"

    def __len__(self):
        assert len(self.entity_finders) == len(self.transformers)
        return len(self.entity_finders)

    def __lt__(self, other):
        if self.nll != other.nll:
            return self.nll < other.nll
        else:
            return self.count < other.count

    def copy(self, parallel=None):
        if parallel is None:
            parallel = self.parallel
        return self.__class__(self.entity_finders, self.transformers,
                              self.nll, parallel=parallel)

    # def add_transformer(self, new_transformer: Transformer):
    #     return self.__class__(entity_finder=self.entity_finder,
    #                           transformer=self.transformer.compose(new_transformer))

    def compose(self, other_predictor, parallel=None):
        if parallel is None:
            parallel = self.parallel
        return self.__class__(self.entity_finders + other_predictor.entity_finders,
                              self.transformers + other_predictor.transformers,
                              parallel=parallel)

    @classmethod
    def reset(cls):
        cls.count = 0
