import itertools
from collections import deque, Counter
from typing import Optional

import numpy as np

from classes import Entity, Property, Relation, EntityFinder
from constants import STRAIGHT_DIRECTIONS, ALL_DIRECTIONS
from my_utils import list2d_to_set, to_tuple, add_one, extend_grid_via_extreme_coordinates, extend_entities_positions, \
    stamp_entities_positions


def same_color(point1, point2):
    return point1[0] == point2[0]


def not_zero(_, point2):
    return bool(point2[0])


def find_components(grid: tuple, relation='same_color', directions=STRAIGHT_DIRECTIONS, ignore_zero=False):
    # Uses breadth first search to find all connected components and puts them in a list
    if len(grid) == 0:
        return {}
    arr_grid = np.array(grid)
    if relation == 'same_color':
        relation_function = same_color
    elif relation == 'not_zero':
        relation_function = not_zero
        ignore_zero = True
    else:
        raise Exception(f'Invalid input for relation_or_property: {relation}')
    array_directions = [np.array(direction) for direction in directions]
    entities = []
    reached = np.full_like(arr_grid, 0)
    for i, j in itertools.product(range(arr_grid.shape[0]), range(arr_grid.shape[1])):
        positions = {}
        if reached[i, j] or (ignore_zero and not arr_grid[i, j]):
            continue
        queue = deque([(i, j)])
        reached[i, j] = 1
        while queue:
            current = queue.pop()
            positions[current] = arr_grid[current[0], current[1]]
            for k, ell in array_directions:
                if (-1 < current[0] + k < arr_grid.shape[0]) and (-1 < current[1] + ell < arr_grid.shape[1]):
                    old_color = positions[current]
                    new_color = arr_grid[current[0] + k, current[1] + ell]
                    if not reached[current[0] + k, current[1] + ell] and \
                            relation_function((old_color, current[0], current[1]),
                                              (new_color, current[0] + k, current[1] + ell)):
                        reached[current[0] + k, current[1] + ell] = 1
                        queue.appendleft((current[0] + k, current[1] + ell))
        entities.append(Entity.make(positions, grid))
    return entities


find_color_entities_cache = {}


def find_color_entities(grid: tuple):
    # Uses breadth first search to find all connected components and puts them in a list
    if grid in find_color_entities_cache:
        return find_color_entities_cache[grid]
    if not grid:
        return {}

    appearing_colors = list2d_to_set(grid)
    arr_grid = np.array(grid)

    entity_dict = {color: [] for color in appearing_colors}
    for i, j in itertools.product(range(arr_grid.shape[0]), range(arr_grid.shape[1])):
        entity_dict[arr_grid[i, j]].append((i, j))

    entities = []
    for color, positions in entity_dict.items():
        new_entity = {}
        for position in positions:
            new_entity[position] = color
        entities.append(Entity.make(new_entity, grid))
    find_color_entities_cache[grid] = entities
    return entities


def move_entity(entity: Entity, new_grid: np.array, old_grid: tuple, vector: tuple,
                entering_color_map: callable, extend_grid: bool = False):
    # global move_entity_cache
    # if entity.freeze() in move_entity_cache:
    #     return move_entity_cache[entity.freeze()]
    old_grid = entity.grid
    new_positions = {}
    for position, color in entity.positions.items():
        if (extend_grid and 0 <= position[0] + vector[0] and 0 <= position[1] + vector[1]) or \
                (0 <= position[0] + vector[0] < new_grid.shape[0] and 0 <= position[1] + vector[1] < new_grid.shape[1]):
            if position[0] + vector[0] >= len(old_grid) or position[1] + vector[1] >= len(old_grid[0]):
                new_positions[(position[0] + vector[0], position[1] + vector[1])] = entering_color_map(0, old_grid[
                    position[0]][position[1]])
                new_color = 0
            else:
                new_color = old_grid[position[0] + vector[0]][position[1] + vector[1]]
            new_positions[(position[0] + vector[0], position[1] + vector[1])] = \
                entering_color_map(new_color, old_grid[position[0]][position[1]])
    # move_entity_cache[entity.freeze()] = new_positions
    return new_positions


def move(entities: list, grid: Optional[tuple] = None, vector_property: callable = None,
         leaving_color=0, entering_color_map=lambda orig_color, entity_color: entity_color, copy=False,
         extend_grid=True) -> tuple:
    # start_time = time.perf_counter()
    if len(entities) > 0:
        grid = entities[0].grid
        new_grid = np.array(grid)
    else:
        return {}, ()
    # accumulated_time[0] += time.perf_counter() - start_time
    new_positions_list = []
    for entity in entities:
        vector = vector_property(entity, grid)
        if vector is None \
                or not all((isinstance(length, int) or isinstance(length, np.int32) or (
                isinstance(length, float) and length.is_integer())) for length in vector):
            return {}, ()
        else:
            vector = tuple((int(length) for length in vector))
        if not copy:
            for position, color in entity.positions.items():
                new_grid[position[0], position[1]] = leaving_color
        # start_time = time.perf_counter()
        new_positions = move_entity(entity, new_grid, grid, vector, entering_color_map, extend_grid=extend_grid)
        # accumulated_time[1] += time.perf_counter() - start_time
        # start_time = time.perf_counter()
        if extend_grid and new_positions:
            # First we compute the new shape of the grid
            max_coordinates = [max((position[i] for position in new_positions.keys())) for i in range(2)]
            positives = [max(max_coordinate, original_max - 1) + 1 for max_coordinate, original_max in
                         zip(max_coordinates, new_grid.shape)]
            if tuple(positives) != new_grid.shape:
                extended_grid = np.zeros(positives)
                extended_grid[:new_grid.shape[0], :new_grid.shape[1]] = new_grid
                new_grid = extended_grid
        # accumulated_time[2] += time.perf_counter() - start_time
        # start_time = time.perf_counter()
        for position, color in new_positions.items():
            new_grid[position[0], position[1]] = new_positions[position[0], position[1]]
        new_positions_list.append(new_positions)
        # accumulated_time[3] += time.perf_counter() - start_time
    # start_time = time.perf_counter()
    new_grid_tuple = to_tuple(new_grid)
    new_entities = [Entity.make(new_positions, new_grid_tuple) for new_positions in new_positions_list]
    # accumulated_time[4] += time.perf_counter() - start_time
    return new_entities, new_grid_tuple


def reflect_about_line(entities: list, grid: Optional[tuple] = None, line_prop: callable = None,
                       color_prop: callable = lambda x, y: next(iter(x.color())), leaving_color=0,
                       color_strategy='replace', copy=True, copy_entities=False, extend_grid=True,
                       change_grid=True) -> tuple:
    """
    Reflects each entities about a line coming from a line property

    :param change_grid: Determines whether to stamp the new entities onto the grid. Used primarily as part of a more
    complicated function
    :param copy_entities: Determines whether to retain the old entities. Primarily used for rotation group functions
    :param extend_grid: Whether to extend the grid when reflecting
    :param copy: Whether to leave the original entity
    :param leaving_color: The color to leave behind if we're not copying
    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param line_prop: property that tells us the line to reflect about
    :param color_prop: the default color for the object after reflection
    :param color_strategy: How to handle the existing color of the grid when placing the object
    :return: the modified entities and grid in a tuple
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    if not old_grid:
        return {}, ()

    grid_arr = np.array(old_grid)
    new_entities = []
    positions_of_new_entities = []
    min_coordinates = [0., 0.]
    max_coordinates = [float(grid_arr.shape[0] - 1), float(grid_arr.shape[1] - 1)]
    for entity in entities:
        new_positions = {}
        line = line_prop(entity, entity.grid)
        if line is None:
            return {}, ()
        if not copy:
            for position in entity.positions:
                grid_arr[position[0], position[1]] = 0
        # color = color_prop(entity, entity.grid)
        for position, color in entity.positions.items():
            if line[1] in {0., 1.}:
                line1_int = int(line[1])
                new_coordinate = position[line1_int] + 2 * (line[0] - position[line1_int])
                if new_coordinate < min_coordinates[line1_int]:
                    min_coordinates[line1_int] = new_coordinate
                elif new_coordinate > max_coordinates[line1_int]:
                    max_coordinates[line1_int] = new_coordinate

                if not new_coordinate.is_integer():
                    return {}, ()
                new_position = tuple((position[i] if i != line1_int else int(new_coordinate) for i in range(2)))
            elif line[1] in {-0.5, 0.5}:
                sign = np.sign(line[1])
                steps = (position[0] + sign * position[1]) / 2 - sign * line[0]
                new_position = (position[0] - 2 * steps, position[1] - sign * 2 * steps)
                for i in range(2):
                    if new_position[i] < min_coordinates[i]:
                        min_coordinates[i] = new_position[i]
                    elif new_position[i] > max_coordinates[i]:
                        max_coordinates[i] = new_position[i]
                if any((not coordinate.is_integer() for coordinate in new_position)):
                    return {}, ()
                else:
                    new_position = tuple((int(coordinate) for coordinate in new_position))
            else:
                raise ValueError(f"line[1] is of invalid value {line[1]}")
            if extend_grid or (0 <= new_position[0] < grid_arr.shape[0] and 0 <= new_position[1] < grid_arr.shape[1]):
                new_positions[new_position] = color
            # grid_arr[new_position[0], new_position[1]] = color
        if new_positions:
            positions_of_new_entities.append(new_positions)
    if extend_grid:
        min_coordinates = tuple((int(min_coordinate) for min_coordinate in min_coordinates))
        max_coordinates = tuple((int(max_coordinate) for max_coordinate in max_coordinates))
        new_grid_arr = extend_grid_via_extreme_coordinates(min_coordinates, max_coordinates, grid_arr)
        positions_of_new_entities = extend_entities_positions(min_coordinates, positions_of_new_entities)
    else:
        new_grid_arr = grid_arr

    if change_grid:
        stamp_entities_positions(positions_of_new_entities, new_grid_arr)
    new_grid = to_tuple(new_grid_arr)
    new_entities = [Entity.make(positions, new_grid) for positions in positions_of_new_entities]

    if copy_entities:
        old_entities_positions = extend_entities_positions(min_coordinates, [entity.positions for entity in entities])
        old_entities = [Entity.make(positions, new_grid) for positions in old_entities_positions]
        new_entities.extend(old_entities)
    return new_entities, new_grid


def rotate_via_reflects(entities: list, grid: Optional[tuple] = None, line_prop1: callable = None,
                        line_prop2: callable = None,
                        color_prop: callable = lambda x, y: next(iter(x.color())), leaving_color=0,
                        color_strategy='replace', copy=True, copy_entities=False, extend_grid=True) -> tuple:
    """
    Applies two reflections sequentially giving a rotation

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :param color_prop:
    :param leaving_color:
    :param color_strategy:
    :param copy:
    :param extend_grid:
    :return:
    """
    line_prop2_values = [line_prop2(entity, entity.grid) for entity in entities]
    new_entities, new_grid = reflect_about_line(entities, grid, line_prop1, copy=True, copy_entities=copy_entities,
                                                change_grid=False, extend_grid=extend_grid)

    def const_line_prop2(entity, grid):
        return line_prop2_values[new_entities.index(entity)]

    new_entities, new_grid = reflect_about_line(new_entities, new_grid, const_line_prop2, copy=extend_grid,
                                                copy_entities=copy_entities, extend_grid=extend_grid)
    return new_entities, new_grid


def rotate_position(position: tuple, pivot: tuple, quarter_steps: int) -> tuple:
    if quarter_steps % 4 == 1:
        return position[1] + pivot[0] - pivot[1], pivot[0] + pivot[1] - position[0]
    elif quarter_steps % 4 == 2:
        return 2 * pivot[0] - position[0], 2 * pivot[1] - position[1]
    elif quarter_steps % 4 == 3:
        return pivot[0] + pivot[1] - position[1], position[0] + pivot[1] - pivot[0]
    else:
        return position[0], position[1]


def rotate_about_point(entities: list, grid: Optional[tuple] = None, point_prop: callable = None,
                       quarter_steps=1, color_prop: callable = lambda x, y: next(iter(x.color())), leaving_color=0,
                       color_strategy='replace', copy=True, extend_grid=True,
                       change_grid=True) -> tuple:
    """
    Applies two reflections sequentially giving a rotation

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :param color_prop:
    :param leaving_color:
    :param color_strategy:
    :param copy:
    :param extend_grid:
    :return:
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    if not old_grid:
        return {}, ()

    grid_arr = np.array(old_grid)
    new_entities = []
    positions_of_new_entities = []
    min_coordinates = [0., 0.]
    max_coordinates = [float(grid_arr.shape[0] - 1), float(grid_arr.shape[1] - 1)]
    for entity in entities:
        new_positions = {}
        point = point_prop(entity, entity.grid)
        if point is None:
            return {}, ()
        if not copy:
            for position in entity.positions:
                grid_arr[position[0], position[1]] = 0
        for position, color in entity.positions.items():
            new_position = rotate_position(position, point, quarter_steps)

            if not new_position[0].is_integer() or not new_position[0].is_integer():
                return {}, ()
            else:
                new_position = tuple((int(coord) for coord in new_position))

            for axis in range(2):
                min_coordinates[axis] = min(min_coordinates[axis], new_position[axis])
                max_coordinates[axis] = max(max_coordinates[axis], new_position[axis])
            if extend_grid or (0 <= new_position[0] < grid_arr.shape[0] and 0 <= new_position[1] < grid_arr.shape[1]):
                new_positions[new_position] = color
        if new_positions:
            positions_of_new_entities.append(new_positions)
    if extend_grid:
        min_coordinates = tuple((int(min_coordinate) for min_coordinate in min_coordinates))
        max_coordinates = tuple((int(max_coordinate) for max_coordinate in max_coordinates))
        new_grid_arr = extend_grid_via_extreme_coordinates(min_coordinates, max_coordinates, grid_arr)
        positions_of_new_entities = extend_entities_positions(min_coordinates, positions_of_new_entities)
    else:
        new_grid_arr = grid_arr

    if change_grid:
        stamp_entities_positions(positions_of_new_entities, new_grid_arr)
    new_grid = to_tuple(new_grid_arr)
    new_entities = [Entity.make(positions, new_grid) for positions in positions_of_new_entities]
    return new_entities, new_grid


def apply_rotation_group_old(entities: list, grid: Optional[tuple] = None,
                             line_prop1=None, line_prop2=None, extend_grid=True):
    """
    Applies a rotation four times. If the element has order four, this gives the action of Z_4

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :return:
    """
    line_prop1_values = [line_prop1(entity, entity.grid) for entity in entities]
    line_prop2_values = [line_prop2(entity, entity.grid) for entity in entities]
    new_entities, new_grid = rotate_via_reflects(entities, grid, line_prop1, line_prop2, extend_grid=extend_grid)
    for _ in range(2):
        def const_line_prop1(entity, grid):
            return line_prop1_values[new_entities.index(entity)]

        def const_line_prop2(entity, grid):
            return line_prop2_values[new_entities.index(entity)]

        new_entities, new_grid = rotate_via_reflects(new_entities, new_grid, const_line_prop1, const_line_prop2,
                                                     extend_grid=True)
    return new_entities, new_grid


def apply_rotation_group(entities: list, grid: Optional[tuple] = None,
                         point_prop=None, extend_grid=True):
    """
    Applies a rotation four times. If the element has order four, this gives the action of Z_4

    :param entities:
    :param grid:
    :param point_prop:
    :param extend_grid:
    :return:
    """
    point_values = [point_prop(entity, entity.grid) for entity in entities]

    # We apply rotations to the same entities to allow for better caching
    for i in range(1, 4):
        def const_point_prop(entity, grid):
            return point_values[entities.index(entity)]
        entities, grid = rotate_about_point(entities, grid, const_point_prop,
                                            extend_grid=extend_grid,
                                            quarter_steps=i)

    return entities, grid


def apply_klein_vier_group(entities: list, grid: Optional[tuple] = None, line_prop1=None, line_prop2=None,
                           extend_grid=True):
    """
    Applies the combinations of actions of two perpendicular reflections, giving a four element group

    :param entities:
    :param grid:
    :param line_prop1:
    :param line_prop2:
    :return:
    """
    line_prop1_values = [line_prop1(entity, entity.grid) for entity in entities]
    line_prop2_values = [line_prop2(entity, entity.grid) for entity in entities]

    new_entities, new_grid = reflect_about_line(entities, grid, line_prop1, extend_grid=extend_grid)

    def const_line_prop2(entity, grid):
        return line_prop2_values[new_entities.index(entity)]

    new_entities, new_grid = reflect_about_line(new_entities, new_grid, const_line_prop2, extend_grid=extend_grid)

    def const_line_prop1(entity, grid):
        return line_prop1_values[new_entities.index(entity)]

    new_entities, new_grid = reflect_about_line(new_entities, new_grid, const_line_prop1, extend_grid=extend_grid)

    return new_entities, new_grid


def place_line(entities: list, grid: Optional[tuple] = None, line_prop: Property = None,
               color_prop: Property = None,
               color_strategy: str = 'original') -> tuple:
    """
    INCOMPLETE

    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param line_prop: property that tells us the line to draw
    :param color_prop: the default color for the line
    :param color_strategy: How to handle the existing color of the grid when placing the line
    :return: the modified entities and grid in a tuple
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    grid_arr = np.array(old_grid)
    new_entities = []
    for entity in entities:
        line = line_prop(entity, entity.grid)
        color = color_prop(entity, entity.grid)
        ...

    return new_entities, to_tuple(grid_arr)


def place_line_segment(entities: list, grid: Optional[tuple] = None, point_prop1: Property = None,
                       point_prop2: Property = None, color_prop: Property = None,
                       color_strategy: str = 'original') -> tuple:
    """
    INCOMPLETE

    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param point_prop1: the start of the line
    :param point_prop2: the end of the line
    :param color_prop: the default color for the line
    :param color_strategy: How to handle the existing color of the grid when placing the line
    :return: the modified entities and grid
    """
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    grid_arr = np.array(old_grid)
    new_entities = []
    for entity in entities:
        point1 = point_prop1(entity, entity.grid)
        point2 = point_prop2(entity, entity.grid)
        color = color_prop(entity, entity.grid)
        ...

    return new_entities, to_tuple(grid_arr)


def place_shape(entities: list, grid: Optional[tuple] = None, point_prop: Property = None,
                shape_prop: Property = None, color_strategy: str = 'original') -> tuple:
    """

    :param entities: entities to use to get the property values
    :param grid: not used, necessary for compatibility
    :param point_prop: location for the center of the shape to be
    :param shape_prop: the shape to be placed
    :param color_strategy: Determines how to color the placed shape. Options are 'original', 'extend_non_0', and 'replace_0'
    :return:
    """
    # Adds a shape to the original board
    if len(entities) > 0:
        old_grid = entities[0].grid
    else:
        old_grid = grid
    if color_strategy not in {'original', 'extend_non_0', 'replace_0'}:
        raise Exception(f'Invalid color strategy: {color_strategy}')
    grid_arr = np.array(old_grid)
    new_entities = []
    for entity in entities:
        new_entity_positions = entity.positions.copy()
        point = point_prop(entity, entity.grid)
        fixed_shape = shape_prop(entity, entity.grid)
        if fixed_shape is None or point is None:
            return {}, ()

        y_coord, x_coord = point
        color_map = {}

        if color_strategy == 'original' or color_strategy == 'replace_0':
            color_map = {i: i for i in range(10)}
        elif color_strategy == 'extend_non_0':
            for position, color in fixed_shape:
                new_position_0, new_position_1 = position[0] + y_coord, position[1] + x_coord
                if new_position_0 < 0 or new_position_1 < 0:
                    return {}, ()
                if not isinstance(new_position_0, int) and new_position_0.is_integer():
                    new_position_0 = int(new_position_0)
                if not isinstance(new_position_1, int) and new_position_1.is_integer():
                    new_position_1 = int(new_position_1)
                try:
                    if grid_arr[new_position_0, new_position_1] != 0:
                        if color in color_map and grid_arr[new_position_0, new_position_1] != color_map[color]:
                            return {}, ()
                        else:
                            color_map[color] = grid_arr[new_position_0, new_position_1]
                except IndexError or KeyError:
                    return {}, ()
            for i in range(10):
                if i not in color_map:
                    color_map[i] = i

        for position, color in fixed_shape:
            new_position_0, new_position_1 = position[0] + y_coord, position[1] + x_coord
            if new_position_0 < 0 or new_position_1 < 0:
                return {}, ()
            if not isinstance(new_position_0, int) and new_position_0.is_integer():
                new_position_0 = int(new_position_0)
            if not isinstance(new_position_1, int) and new_position_1.is_integer():
                new_position_1 = int(new_position_1)
            try:
                new_color = color_map[color] \
                    if (color_strategy != 'replace_0' or grid_arr[new_position_0, new_position_1] == 0) \
                    else grid_arr[new_position_0, new_position_1]
                grid_arr[new_position_0, new_position_1] = new_color
                new_entity_positions[(new_position_0, new_position_1)] = new_color
            except IndexError or KeyError:
                return {}, ()
        new_entities.append(Entity.make(new_entity_positions, entity.grid))
    return new_entities, to_tuple(grid_arr)


def output_shape_as_grid(entities: list, grid: Optional[tuple] = None, shape_prop: Property = None):
    if len(entities) > 0:
        if grid is None:
            grid = entities[0].grid
    # else:
    #     return {}, ()

    shape = shape_prop(None, grid)
    if shape is None:
        return {}, ()
    min_y, max_y = min((y for (y, x), color in shape)), max((y for (y, x), color in shape))
    min_x, max_x = min((x for (y, x), color in shape)), max((x for (y, x), color in shape))
    new_grid = np.zeros((int(max_y + 1 - min_y), int(max_x + 1 - min_x)))
    for (y, x), color in shape:
        # print((y, x), color)
        new_grid[int(y + min_y), int(x + min_x)] = color
    return dict(shape), to_tuple(new_grid)


def replace_color(entities: list, grid: Optional[tuple] = None,
                  source_color_prop: Optional[Property] = None,
                  target_color_prop: Optional[Property] = None):
    if len(entities) > 0:
        if grid is None:
            grid = entities[0].grid
        new_grid = np.array(grid)
    else:
        return {}, ()

    if len(grid) == 0 or len(grid) != len(entities[0].grid) or len(grid[0]) != len(entities[0].grid[0]):
        return {}, ()

    new_positions_list = []
    for entity in entities:
        source_colors = source_color_prop(entity, entity.grid)
        target_colors = target_color_prop(entity, entity.grid)
        if source_colors is None or target_colors is None or len(target_colors) != 1:
            # If target color has more than just one element the map becomes ambiguous, so just return a trivial result
            return {}, ()
        target_color = next(iter(target_colors))

        new_positions = {}
        for position, color in entity.positions.items():
            new_color = target_color if color in source_colors else color
            new_positions[position] = new_color
            if new_color != color:
                new_grid[position[0], position[1]] = new_color
        new_positions_list.append(new_positions)
    new_entities = [Entity.make(new_positions, to_tuple(new_grid)) for new_positions in new_positions_list]
    return new_entities, to_tuple(new_grid)


def replace_color_in_frame(box_y_min, box_y_max, box_x_min, box_x_max, grid: tuple,
                           source_colors: frozenset,
                           target_color: int):
    min_y = len(grid) - 1
    max_y = 0
    assert len(grid) > 0
    min_x = len(grid[0]) - 1
    max_x = 0
    grid_arr = np.array(grid)
    min_y = min(min_y, box_y_min)
    max_y = max(max_y, box_y_max)
    min_x = min(min_x, box_x_min)
    max_x = max(max_x, box_x_max)
    source_colors = list(source_colors)
    places_grid_in_source = np.isin(grid_arr[min_y: max_y + 1, min_x: max_x + 1], source_colors)
    grid_arr[min_y: max_y + 1, min_x: max_x + 1] = np.where(places_grid_in_source, target_color,
                                                            grid_arr[min_y: max_y + 1, min_x: max_x + 1])
    return to_tuple(grid_arr)


def replace_colors_in_entities_frame(entities, grid: Optional[tuple] = None, offsets=(0, 0, 0, 0),
                                     source_color_prop: Optional[Property] = None,
                                     target_color_prop: Optional[Property] = None):
    if len(entities) > 0:
        grid = entities[0].grid
    else:
        # print('No entities')
        return {}, ()
    grid_min_y = len(grid) - 1
    grid_max_y = 0
    assert len(grid) > 0
    grid_min_x = len(grid[0]) - 1
    grid_max_x = 0
    # print(source_color_prop, target_color_prop)
    for entity in entities:
        source_colors = source_color_prop(entity, entity.grid)
        target_colors = target_color_prop(entity, entity.grid)
        # print(source_colors, target_colors)
        if source_colors is None or target_colors is None or len(target_colors) != 1:
            # If target color has more than just one element the map becomes ambiguous, so just return a trivial result
            # print(f'source_colors={source_colors}, target_colors={target_colors}')
            return {}, ()
        else:
            target_color = next(iter(target_colors))
        min_y = min(grid_min_y, entity.min_coord(0)) + offsets[0]
        max_y = max(grid_max_y, entity.max_coord(0)) + offsets[1]
        min_x = min(grid_min_x, entity.min_coord(1)) + offsets[2]
        max_x = max(grid_max_x, entity.max_coord(1)) + offsets[3]
        grid = replace_color_in_frame(min_y, max_y, min_x, max_x, grid, source_colors, target_color)
    return {}, grid


def crop_grid(grid: tuple, y_range: tuple, x_range: tuple) -> tuple:
    new_grid = np.array(grid)
    y_begin, y_end = y_range
    x_begin, x_end = x_range
    return to_tuple(new_grid[y_begin:y_end + 1, x_begin:x_end + 1])


def crop_entities(entities, grid: Optional[tuple] = None, offsets=(0, 0, 0, 0)):
    if len(entities) > 0:
        grid = entities[0].grid
    else:
        return {}, ()
    min_y = len(grid) - 1
    max_y = 0
    assert len(grid) > 0
    min_x = len(grid[0]) - 1
    max_x = 0
    for entity in entities:
        min_y = min(min_y, entity.min_coord(0)) + offsets[0]
        max_y = max(max_y, entity.max_coord(0)) + offsets[1]
        min_x = min(min_x, entity.min_coord(1)) + offsets[2]
        max_x = max(max_x, entity.max_coord(1)) + offsets[3]
    new_grid = crop_grid(grid, (min_y, max_y), (min_x, max_x))
    new_entities = []
    for entity in entities:
        temp_position = {}
        for position, color in entity.positions.items():
            temp_position[(position[0] - min_y, position[1] - min_x)] = color
        new_entities.append(Entity.make(temp_position, new_grid))

    return new_entities, new_grid


adjacent_direction_cache = {}


def adjacent_direction(entity1: Entity, entity2: Entity):
    if entity1 is None or entity2 is None:
        return []
    key = (frozenset(entity1.positions.keys()), frozenset(entity2.positions.keys()))
    if key in adjacent_direction_cache:
        return adjacent_direction_cache[key]
    directions = []
    test_directions = list(ALL_DIRECTIONS)
    test_directions.remove((0, 0))
    for position1, position2 in itertools.product(entity1.positions.keys(), entity2.positions.keys()):
        for direction in test_directions:
            if position1[0] + direction[0] == position2[0] and position1[1] + direction[1] == position2[1]:
                directions.append(direction)
                test_directions.remove(direction)
                break
    adjacent_direction_cache[key] = directions
    return directions


def direction_sign_to_vector(direction, sign, value):
    vector = [0, 0]
    vector[direction] = value * sign
    return tuple(vector)


collision_directions_cache = {}


def collision_directions(entity1: Entity, entity2: Entity, adjustment=0):
    if entity1 is None or entity2 is None:
        return []
    cache_key = (frozenset(entity1.positions.keys()), frozenset(entity2.positions.keys()), adjustment)
    if cache_key in collision_directions_cache:
        return collision_directions_cache[cache_key]
    min_distances = {(i, sign): float('inf') for i, sign in itertools.product(range(2), (1, -1))}
    for position1, position2 in itertools.product(entity1.positions.keys(), entity2.positions.keys()):
        for i, sign in itertools.product(range(2), (1, -1)):
            if position1[int(not i)] == position2[int(not i)] and position1[i] * sign < position2[i] * sign:
                min_distances[(i, sign)] = min(min_distances[(i, sign)],
                                               position2[i] * sign - position1[i] * sign - 1)
    out = frozenset(
        direction_sign_to_vector(*key, value + (adjustment if value != 0 else 0)) for key, value in
        min_distances.items() if
        value != float('inf'))
    collision_directions_cache[cache_key] = out
    return out


def surrounded(entity1, entity2):
    return all((entity2.min_coord(axis) < entity1.min_coord(axis) for axis in range(2))) and all(
        (entity1.max_coord(axis) < entity2.max_coord(axis) for axis in range(2)))


def generate_base_relations():
    # 39
    Relation(lambda entity1, entity2: adjacent_direction(entity1, entity2)[0] if len(
        adjacent_direction(entity1, entity2)) == 1 else None,
             nll=1 + np.log(2),
             name='find the unique touching direction to',
             output_types=frozenset({'vector'}))
    # 40
    Relation(lambda entity1, entity2: True if adjacent_direction(entity1, entity2) else False,
             nll=1 + np.log(2),
             name='are touching',
             output_types=frozenset({'bool'}))
    # 41
    Relation(lambda entity1, entity2: next(iter(collision_directions(entity1, entity2))) if len(
        collision_directions(entity1, entity2)) == 1 else None,
             nll=1 + np.log(2), name='the unique pre-collision vector to',
             output_types=frozenset({'vector'}))
    # 41.5
    Relation(lambda entity1, entity2: next(iter(collision_directions(entity1, entity2, adjustment=1)))
    if len(collision_directions(entity1, entity2)) == 1 else None,
             nll=1 + np.log(2), name='the unique collision vector to',
             output_types=frozenset({'vector'}))
    # 42
    Relation(lambda entity1, entity2: entity1.colors() == entity2.colors(),
             nll=1 + np.log(2), name='shares a color with',
             output_types=frozenset({'bool'}))
    # 43
    Relation(lambda entity1, entity2: entity1.shape() == entity2.shape(),
             nll=1 + np.log(2), name='has the same shape as',
             output_types=frozenset({'bool'}))
    # 44
    Relation(lambda entity1, entity2: entity1.uncolored_shape() == entity2.uncolored_shape(),
             nll=1 + np.log(2), name='has the same uncolored shape as',
             output_types=frozenset({'bool'}))
    # # 44.5
    Relation(lambda entity1, entity2: surrounded(entity1, entity2),
             nll=1 + np.log(2), name='is surrounded by',
             output_types=frozenset({'bool'}))


def generate_base_properties(case, entity_finder):
    entity_properties = [
        # 0
        Property(lambda x: x.entity.num_points(), nll=np.log(2),
                 name='the number of points',
                 output_types=frozenset({'quantity'}),
                 entity_finder=entity_finder),
        # 1
        Property(lambda x: x.entity.colors(), nll=0,
                 name='the colors',
                 output_types=frozenset({'color'}),
                 entity_finder=entity_finder),

        # Coordinate Properties
        # 2
        Property(lambda x: x.entity.center(axis=0), nll=np.log(2),
                 name='the center y coordinate',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 3
        Property(lambda x: x.entity.center(axis=1), nll=np.log(2),
                 name='the center x coordinate',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),
        # 4
        Property(lambda x: float(x.entity.min_coord(axis=0)), np.log(4),
                 name='the smallest y coordinate',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 5
        Property(lambda x: float(x.entity.min_coord(axis=1)), np.log(4),
                 name='the smallest x coordinate',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),
        # 6
        Property(lambda x: float(x.entity.max_coord(axis=0)), np.log(4),
                 name='the largest y coordinate',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 7
        Property(lambda x: float(x.entity.max_coord(axis=1)), np.log(4),
                 name='the largest x coordinate',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),

        # Line properties
        Property(lambda x: (float(x.entity.center(axis=1)) / 2., 1.),
                 np.log(4),
                 name="the entity\'s vertical center line",
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.center(axis=0) / 2., 0.),
                 np.log(4),
                 name='the entity\'s horizontal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=1), -0.5),
                 np.log(4),
                 name='the entity\'s back diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.max_coord(axis=1) / 2., 0.5),
                 np.log(4),
                 name='the entity\'s forward diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        Property(lambda x: (x.entity.max_coord(axis=1) - 0.5, 1.),
                 np.log(4),
                 name='the entity\'s vertical right-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=1) - 0.5, 1.),
                 np.log(4),
                 name='the entity\'s vertical left-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(x.entity.max_coord(axis=0) - 0.5), 0.),
                 np.log(4),
                 name='the entity\'s horizontal bottom-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=0) - 0.5, 0.),
                 np.log(4),
                 name='the entity\'s horizontal top-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        # Point properties
        # 8
        Property(lambda x: (x.entity.center(axis=0), x.entity.center(axis=1)), nll=0,
                 name='the center point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=0) - 0.5, x.entity.min_coord(axis=1) - 0.5), nll=np.log(4),
                 name='the top-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.min_coord(axis=0) - 0.5, x.entity.max_coord(axis=1) + 0.5), nll=np.log(4),
                 name='the top-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.max_coord(axis=0) + 0.5, x.entity.min_coord(axis=1) - 0.5), nll=np.log(4),
                 name='the bottom-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (x.entity.max_coord(axis=0) + 0.5, x.entity.max_coord(axis=1) + 0.5), nll=np.log(4),
                 name='the bottom-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),

        # Length Properties
        # 9
        Property(lambda x: float(x.entity.max_coord(axis=0) - x.entity.min_coord(axis=0) + 1),
                 np.log(4),
                 name='the y length',
                 output_types=frozenset({'y_length'}),
                 entity_finder=entity_finder),
        # 9.5
        Property(lambda x: -(float(x.entity.max_coord(axis=0) - x.entity.min_coord(axis=0) + 1)),
                 np.log(4),
                 name='the negative y length',
                 output_types=frozenset({'y_length'}),
                 entity_finder=entity_finder),
        # 10
        Property(lambda x: float(x.entity.max_coord(axis=1) - x.entity.min_coord(axis=1) + 1),
                 np.log(4),
                 name='the x length',
                 output_types=frozenset({'x_length'}),
                 entity_finder=entity_finder),
        # 10.5
        Property(lambda x: -(float(x.entity.max_coord(axis=1) - x.entity.min_coord(axis=1) + 1)),
                 np.log(4),
                 name='the negative x length',
                 output_types=frozenset({'x_length'}),
                 entity_finder=entity_finder),

        # Shape-based properties
        # 11
        Property(lambda x: x.entity.shape(), 0,
                 name='the shape',
                 output_types=frozenset({'shape'}),
                 entity_finder=entity_finder),
        # 12
        Property(lambda x: x.entity.is_a_rectangle(), np.log(4),
                 name='is a rectangle',
                 output_types=frozenset({'bool'}),
                 entity_finder=entity_finder),
        # 13
        Property(lambda x: x.entity.is_a_square(), np.log(2),
                 name='is a square',
                 output_types=frozenset({'bool'}),
                 entity_finder=entity_finder),
        # 14
        Property(lambda x: x.entity.is_a_line(), np.log(2),
                 name='is a line',
                 output_types=frozenset({'bool'}),
                 entity_finder=entity_finder)
    ]

    for prop in entity_properties:
        prop.requires_entity = True

    grid_properties = [
        # 15
        Property(lambda x: 2 + len(x.entities), 0, name='the number of entities',
                 output_types=frozenset({'quantity'}),
                 entity_finder=entity_finder),
        # 16
        Property(lambda x: float(np.array(x.grid).shape[0]), np.log(2), name='the height of the grid',
                 output_types=frozenset({'y_coordinate', 'y_length'}),
                 entity_finder=entity_finder),
        # 17
        Property(lambda x: float(np.array(x.grid).shape[1]), np.log(2), name='the width of the grid',
                 output_types=frozenset({'x_coordinate', 'x_length'}),
                 entity_finder=entity_finder),

        # 18
        Property(lambda x: (float(np.array(x.grid).shape[0] - 1) / 2., float(np.array(x.grid).shape[1] - 1) / 2.),
                 0,
                 name='the center point of the grid',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        # 101
        Property(lambda x: float(np.array(x.grid).shape[0] - 1) / 2.,
                 np.log(4),
                 name='the center y coordinate of the grid',
                 output_types=frozenset({'y_coordinate'}),
                 entity_finder=entity_finder),
        # 102
        Property(lambda x: float(np.array(x.grid).shape[1] - 1) / 2.,
                 np.log(4),
                 name='the center x coordinate of the grid',
                 output_types=frozenset({'x_coordinate'}),
                 entity_finder=entity_finder),
        # 19
        Property(lambda x: True,
                 0,
                 name='True',
                 output_types=frozenset({'bool'}),
                 is_constant=True,
                 entity_finder=entity_finder),

        Property(lambda x: (float(np.array(x.grid).shape[1] - 1) / 2., 1.),
                 np.log(4),
                 name='the vertical center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0] - 1) / 2., 0.),
                 np.log(4),
                 name='the horizontal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (0., -0.5),
                 np.log(4),
                 name='the back diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[1] - 1.) / 2., 0.5),
                 np.log(4),
                 name='the forward diagonal center line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        Property(lambda x: (float(np.array(x.grid).shape[1] - 0.5), 1.),
                 np.log(4),
                 name='the vertical right-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (- 0.5, 1.),
                 np.log(4),
                 name='the vertical left-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0] - 0.5), 0.),
                 np.log(4),
                 name='the horizontal bottom-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),
        Property(lambda x: (- 0.5, 0.),
                 np.log(4),
                 name='the horizontal top-most line',
                 output_types=frozenset({'line'}),
                 entity_finder=entity_finder),

        # Grid Corners
        Property(lambda x: (0 - 0.5, 0 - 0.5), nll=np.log(4),
                 name='the grid top-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (0 - 0.5, float(np.array(x.grid).shape[1]) - 0.5), nll=np.log(4),
                 name='the grid top-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0]) - 0.5, 0 - 0.5), nll=np.log(4),
                 name='the grid bottom-left corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder),
        Property(lambda x: (float(np.array(x.grid).shape[0]) - 0.5, float(np.array(x.grid).shape[1]) - 0.5), nll=np.log(4),
                 name='the grid bottom-right corner point',
                 output_types=frozenset({'point'}),
                 entity_finder=entity_finder)
    ]
    # 20 - 24
    grid_properties.extend([Property(lambda x, i=i: i, nll=2 * i, name=f'{i}',
                                     output_types=frozenset({'x_coordinate', 'y_coordinate',
                                                             'x_length', 'y_length',
                                                             'quantity'}),
                                     is_constant=True,
                                     entity_finder=entity_finder) for i in range(0, 5)])
    # 25 - 29
    grid_properties.extend([Property(lambda x, i=i: -i, nll=2 * i, name=f'-{i}',
                                     output_types=frozenset({'x_length', 'y_length',
                                                             'quantity'}),
                                     is_constant=True,
                                     entity_finder=entity_finder) for i in range(1, 5)])
    appearing_colors = Counter()
    shape_counter = Counter()
    non_black_shape_counter = Counter()
    uncolored_shape_counter = Counter()
    y_counter = Counter()
    x_counter = Counter()
    point_counter = Counter()
    non_black_uncolored_shape_counter = Counter()
    non_black_entity_finder = EntityFinder(
        lambda grid: find_components(grid, relation='not_zero', directions=ALL_DIRECTIONS))

    for grid in case['train']:
        # We use the data to determine the likely constant properties, and set their likelihood
        for color in (list2d_to_set(grid['output']) - list2d_to_set(grid['input'])) | \
                     (list2d_to_set(grid['input']) - list2d_to_set(grid['output'])):
            appearing_colors[color] = 10
        appearing_colors.update(list2d_to_set(grid['input']))
        appearing_colors.update(list2d_to_set(grid['output']))

        output_entities = entity_finder(grid['output'])
        output_entities_non_black = non_black_entity_finder(grid['output'])

        shape_counter.update(Entity.shapes(output_entities))
        non_black_shape_counter.update(Entity.shapes(output_entities_non_black))

        non_black_uncolored_shape_counter.update(Entity.uncolored_shapes(output_entities_non_black))
        uncolored_shape_counter.update(Entity.uncolored_shapes(output_entities))

        # Point and coordinate constants
        y_counter.update({(entity.center(0)) for entity in output_entities})
        x_counter.update({(entity.center(1)) for entity in output_entities})
        point_counter.update({(entity.center(0), entity.center(1)) for entity in output_entities})
    for color, count in appearing_colors.items():
        # If the color appears in every example, it is also very likely to be used
        if count == len(case['train']) * 2:
            appearing_colors[color] = 10

    # We determine which shapes most likely to be used as constants by counting the number of times the shape appears
    # in an uncolored sense to allow for otherwise identical shapes with different colorings
    common_uncolored_shapes = {uncolored_shape for uncolored_shape, count in
                               itertools.chain(uncolored_shape_counter.items(),
                                               non_black_uncolored_shape_counter.items()) if
                               count > 1}

    combined_shapes = {shape: max(shape_counter[shape], non_black_shape_counter[shape]) for shape in
                       itertools.chain(shape_counter.keys(), non_black_shape_counter.keys())
                       if frozenset({x[0] for x in shape}) in common_uncolored_shapes}

    # 30
    grid_properties.extend(
        [Property(lambda x, color=color, count=count: frozenset({color}),
                  np.log(max(len(appearing_colors) - count, 1)),
                  name=f'color {color}',
                  output_types=frozenset({'color'}),
                  is_constant=True,
                  entity_finder=entity_finder) for color, count in appearing_colors.items()])
    # 31
    shapes = [Property(lambda x, shape=shape: shape, nll=max(np.log(10) - count * np.log(2), 0), name=f'shape {shape}',
                       output_types=frozenset({'shape'}), is_constant=True, entity_finder=entity_finder) for
              shape, count in combined_shapes.items()]
    grid_properties.extend(shapes)
    # for shape in shapes:
    #     print(shape, shape.nll)
    # 32
    grid_properties.extend(
        [Property(lambda x: y_coordinate,
                  nll=max(np.log(len(case['train']) + 1 - count), 0),
                  name=f'y = {y_coordinate}',
                  output_types=frozenset({'y_coordinate'}),
                  is_constant=True,
                  entity_finder=entity_finder)
         for y_coordinate, count in y_counter.items()
         if count > 1 and (y_coordinate > 4 or not float(y_coordinate).is_integer())])
    # 33
    grid_properties.extend(
        [Property(lambda x: x_coordinate,
                  nll=max(len(case['train']) - count, 0),
                  name=f'x = {x_coordinate}',
                  output_types=frozenset({'x_coordinate'}),
                  is_constant=True,
                  entity_finder=entity_finder)
         for x_coordinate, count in x_counter.items()
         if count > 1 and (x_coordinate > 4 or not float(x_coordinate).is_integer())])

    # 100
    grid_properties.extend(
        [Property(lambda x: point,
                  nll=max(np.log(3) - count, 0),
                  name=f'point {point}',
                  output_types=frozenset({'point'}),
                  is_constant=True,
                  entity_finder=entity_finder)
         for point, count in point_counter.items()
         if count > 1])
    add_one(grid_properties), add_one(entity_properties)
    return grid_properties, entity_properties
