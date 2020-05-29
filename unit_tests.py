import pytest
import os
import json
import numpy as np
from collections import Counter
from dataclasses import dataclass
import time

import core_functions
from core_functions import reset_all
from atomic_objects import find_components, move, place_shape, replace_color, \
    replace_colors_in_entities_frame, crop_entities, collision_directions, generate_base_properties, surrounded, \
    reflect_about_line, rotate_via_reflects, apply_klein_vier_group, apply_rotation_group_old, rotate_about_point, \
    apply_rotation_group
from my_utils import combine_sorted_queues, to_tuple, display_case, tuplefy_task, ordinal
from constants import ALL_DIRECTIONS, TYPES
from classes import Entity, EntityFinder, OrdinalProperty, Property, Relation, Selector, Transformer, Predictor, \
    ORDINAL_PROPERTIES, nth_ordered, pick_the_unique_value, SINGLE_VALUE
import abstraction_and_learning as pst

color_map = {'black': 0, 'blue': 1, 'red': 2, 'green': 3, 'yellow': 4, 'grey': 5}

collision_relation = Relation(lambda entity1, entity2: next(iter(collision_directions(entity1, entity2))) if len(
    collision_directions(entity1, entity2)) == 1 else None,
                              nll=1 + np.log(2), name='the unique collision vector to',
                              output_types=frozenset({'vector'}))

trivial_selector = Selector(lambda entity, grid: True, name='true', nll=0)

base_entity_finder = EntityFinder(
    lambda grid: find_components(grid, directions=ALL_DIRECTIONS))


def test_composite_selections():
    with open('training/' + os.listdir('training/')[205]) as f:
        raw_cases = json.load(f)
    cases = tuplefy_task(raw_cases)
    color_0 = Property(lambda x: frozenset({0}), np.log(2), name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_5 = Property(lambda x: frozenset({5}), np.log(10) - 1, name=f'color {5}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    select_not_0 = Selector.make_property_selector(take_color, color_0, False)
    select_not_5 = Selector.make_property_selector(take_color, color_5, False)
    select_not_0_nor_5 = select_not_0.intersect(select_not_5)
    entity_finder = base_entity_finder.compose(select_not_0_nor_5, True)
    select_5 = Selector.make_property_selector(take_color, color_5)
    center_y = Property(lambda x: x.entity.center(axis=0), nll=np.log(2),
                        name='the center y coordinate',
                        output_types=frozenset({'y_coordinate'}),
                        entity_finder=base_entity_finder,
                        requires_entity=True)
    center_x = Property(lambda x: x.entity.center(axis=1), nll=np.log(2),
                        name='the center x coordinate',
                        output_types=frozenset({'x_coordinate'}),
                        entity_finder=base_entity_finder,
                        requires_entity=True)
    center_5y = center_y.add_selector(select_5)
    length_5y = Property.create_distance_property(center_5y, center_y)
    center_5x = center_x.add_selector(select_5)
    length_5x = Property.create_distance_property(center_5x, center_x)
    vect_prop = Property.xy_length_to_vector(length_5y, length_5x)
    move_to_5 = Transformer(lambda entities, grid, copy=True: move(entities, vector_property=vect_prop,
                                                                   copy=copy,
                                                                   extend_grid=False),
                            nll=vect_prop.nll + np.log(2),
                            name=f"{'copy' if True else 'move'} them by ({vect_prop})")
    my_predictor = Predictor(entity_finder, move_to_5)

    for case in cases['train']:
        assert my_predictor.predict(case['input']) == case['output']


def test_take_colors():
    with open('training/' + os.listdir('training/')[7]) as f:
        raw_case7 = json.load(f)
    case7 = tuplefy_task(raw_case7)
    inp = case7['train'][0]['input']
    base_entity_finder = EntityFinder(find_components)
    entities = base_entity_finder(inp)
    # print(collision_relation(entities[1], entities[2]))
    assert collision_relation(entities[1], entities[2]) == (6, 0)

    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    color_0 = Property(lambda x, i=0: frozenset({0}), np.log(10) - 2, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_2 = Property(lambda x, i=2: frozenset({2}), np.log(10) - 2, name=f'color {2}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_8 = Property(lambda x, i=8: frozenset({8}), np.log(10) - 2, name=f'color {8}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    assert take_color(entities[0], inp) == frozenset({0})
    assert take_color(entities[1], inp) == frozenset({2})
    assert take_color(entities[2], inp) == frozenset({8})
    select_0 = Selector.make_property_selector(take_color, color_0)
    select_2 = Selector.make_property_selector(take_color, color_2)
    select_8 = Selector.make_property_selector(take_color, color_8)
    assert select_0.select(entities) == [entities[0]]
    assert select_2.select(entities) == [entities[1]]
    assert select_8.select(entities) == [entities[2]]


def test_nth_ordered_ordinal_property():
    max_ord = OrdinalProperty(lambda x: nth_ordered(x, 0, use_max=True),
                              nll=0,
                              name=f'take the {1} largest')
    second_smallest_ord = OrdinalProperty(lambda x: nth_ordered(x, 1, use_max=False),
                                          nll=0,
                                          name=f'take the {2} smallest')
    assert max_ord([0, 5, 10, 20]) == 20
    assert second_smallest_ord([-2, 5, 10, 20]) == 5
    # max_ord_2 = ORDINAL_PROPERTIES[1]
    # assert max_ord_2([0, 5, 10, 20]) == 20
    # assert ORDINAL_PROPERTIES[2]([0, 5, 10, 20]) == 10


def test_from_relation_selector():
    with open('training/' + os.listdir('training/')[7]) as f:
        raw_case7 = json.load(f)
    case7 = tuplefy_task(raw_case7)
    inp = case7['train'][0]['input']
    base_entity_finder = EntityFinder(find_components)
    entities = base_entity_finder(inp)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    color_2 = Property(lambda x, i=2: frozenset({2}), np.log(10) - 2, name=f'color {2}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_8 = Property(lambda x, i=8: frozenset({8}), np.log(10) - 2, name=f'color {8}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    unique = OrdinalProperty(lambda x: pick_the_unique_value(x),
                             nll=np.log(2),
                             name=f'take the value that is unique',
                             input_types=TYPES)
    max_ord = OrdinalProperty(lambda x: nth_ordered(x, 0, use_max=True),
                              nll=0,
                              name=f'take the {1} largest')

    find_collision_vect_8 = Property.from_relation_selector(collision_relation,
                                                            Selector.make_property_selector(take_color, color_8),
                                                            entity_finder=base_entity_finder,
                                                            ordinal_property=unique)
    # print(type(find_collision_vect_8(entities[1], inp)))
    assert find_collision_vect_8(entities[1], inp) == (6, 0)

    find_collision_vect_2 = Property.from_relation_selector(collision_relation,
                                                            Selector.make_property_selector(take_color, color_2),
                                                            entity_finder=base_entity_finder,
                                                            ordinal_property=unique)
    assert find_collision_vect_2(entities[2], inp) == (-6, 0)


def test_transformers_predictors():
    with open('training/' + os.listdir('training/')[7]) as f:
        raw_case7 = json.load(f)
    case7 = tuplefy_task(raw_case7)
    inp = case7['train'][0]['input']
    out = case7['train'][0]['output']
    base_entity_finder = EntityFinder(find_components)
    entities = base_entity_finder(inp)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    color_2 = Property(lambda x, i=2: frozenset({2}), np.log(10) - 2, name=f'color {2}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_8 = Property(lambda x, i=8: frozenset({8}), np.log(10) - 2, name=f'color {8}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    select_8 = Selector.make_property_selector(take_color, color_8)
    select_2 = Selector.make_property_selector(take_color, color_2)
    max_ord = OrdinalProperty(lambda x: nth_ordered(x, 0, use_max=True),
                              nll=0,
                              name=f'take the {1} largest')
    find_collision_vect_to_8 = Property.from_relation_selector(collision_relation,
                                                               select_8,
                                                               entity_finder=base_entity_finder,
                                                               ordinal_property=max_ord)
    my_transformer = Transformer(lambda entities, grid: move(entities, vector_property=find_collision_vect_to_8),
                                 name=f'move them by ({find_collision_vect_to_8})',
                                 nll=1 + np.log(2))

    assert my_transformer.transform(select_2.select(entities))[1] == out

    select_2_finder = base_entity_finder.compose(select_2)
    my_predictor = Predictor(select_2_finder, my_transformer)
    assert my_predictor.predict(inp) == out


def test_case_29():
    with open('training/' + os.listdir('training/')[29]) as f:
        raw_task = json.load(f)
    base_entity_finder = EntityFinder(lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
    trivial_selector = Selector(lambda entity, grid: True, name='')
    task = tuplefy_task(raw_task)
    inp = task['train'][0]['input']
    out = task['train'][0]['output']
    # print(task['train'][0]['input'])
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1,
                          requires_entity=True)
    # color_2 = Property(lambda x, i=2: frozenset({2}), np.log(10) - 2, name=f'color {2}',
    #                    output_types=frozenset({'color'}))
    color_1 = Property(lambda x, i=2: frozenset({1}), np.log(10) - 1, name=f'color {1}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_0 = Property(lambda x, i=2: frozenset({0}), np.log(10) - 1, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    select_1 = Selector.make_property_selector(take_color, color_1)
    property_0 = Property(lambda x, i=0: i, nll=1, name=f'{0}',
                          output_types=frozenset({'x_coordinate', 'y_coordinate',
                                                  'x_length', 'y_length',
                                                  'quantity'}),
                          entity_finder=base_entity_finder)
    select_not_0 = Selector.make_property_selector(take_color, color_0, the_same=False)
    smallest_y = Property(lambda x: x.entity.max_coord(axis=0), 1 + np.log(4),
                          name='the largest y coordinate',
                          output_types=frozenset({'y_coordinate'}),
                          entity_finder=base_entity_finder,
                          requires_entity=True)
    min_y_of_blue = smallest_y.add_selector(select_1)
    distance_to_min_y_of_blue = Property.create_distance_property(min_y_of_blue, smallest_y)
    vector_to_min_y_of_blue = Property.xy_length_to_vector(distance_to_min_y_of_blue, property_0)
    move_transform = Transformer(
        lambda entities, grid, vector_prop=vector_to_min_y_of_blue: move(entities, vector_property=vector_prop),
        nll=vector_to_min_y_of_blue.nll + np.log(2),
        name=f'move them by ({vector_to_min_y_of_blue})')
    my_predictor = Predictor(base_entity_finder.compose(trivial_selector), move_transform)  # .compose(select_not_0)
    # display_case(my_predictor.predict(inp))
    # display_case(out)
    assert my_predictor.predict(inp) == out

    test_input = task['test'][0]['input']
    test_output = task['test'][0]['output']
    test_entities = base_entity_finder(test_input)
    assert len(test_entities) == 4

    selected_finder = base_entity_finder.compose(select_not_0)
    # selected_finder(test_input)
    assert len(selected_finder(test_input)) == 3

    assert my_predictor.predict(test_input) == test_output


def test_case_30():
    with open('training/' + os.listdir('training/')[30]) as f:
        raw_task = json.load(f)
    base_entity_finder = EntityFinder(lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
    task = tuplefy_task(raw_task)
    inp = task['train'][0]['input']
    output = task['train'][0]['output']
    entities = base_entity_finder(inp)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1,
                          requires_entity=True)
    color_0 = Property(lambda x, i=2: frozenset({0}), np.log(10) - 1, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    select_not_0 = Selector.make_property_selector(take_color, color_0, the_same=False)
    crop_transform = Transformer(crop_entities, nll=np.log(2), name='crop them')
    _, trivial_transformed_grid = crop_transform.transform(entities)
    assert trivial_transformed_grid == inp

    selected_entities = select_not_0.select(entities)
    _, transformed_grid = crop_transform.transform(selected_entities)
    assert transformed_grid == ((0, 2, 2, 2), (0, 0, 2, 0), (2, 2, 2, 0), (2, 0, 2, 0))

    my_predictor = Predictor(base_entity_finder.compose(select_not_0), crop_transform)

    for case in task['train']:
        assert my_predictor.predict(case['input']) == case['output']

    test_case = task['test'][0]
    print(my_predictor)
    assert my_predictor.predict(test_case['input']) == test_case['output']


def test_case_34():
    with open('training/' + os.listdir('training/')[34]) as f:
        raw_task = json.load(f)
    base_entity_finder = EntityFinder(lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
    task = tuplefy_task(raw_task)
    inp = task['train'][0]['input']
    out = task['train'][0]['output']
    entities = base_entity_finder(inp)

    color_8 = Property(lambda x: frozenset({8}), np.log(10) - 1, name=f'color {8}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_0 = Property(lambda x: frozenset({0}), np.log(10) - 1, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1,
                          requires_entity=True)
    select_8 = Selector.make_property_selector(take_color, color_8, True)
    select_not_8 = Selector.make_property_selector(take_color, color_8, False)
    select_not_0 = Selector.make_property_selector(take_color, color_0, False)
    select_not_0.nll = np.log(2)
    select_not_0_nor_8 = Selector.intersect(select_not_0, select_not_8)

    selected_entities = select_not_0_nor_8.select(entities)

    collision = Relation(lambda entity1, entity2: next(iter(collision_directions(entity1, entity2, adjustment=1)))
    if len(collision_directions(entity1, entity2)) == 1 else None,
                         nll=1 + np.log(2), name='the unique collision vector to',
                         output_types=frozenset({'vector'}))
    collision_with_8 = Property.from_relation_selector(collision, select_8, base_entity_finder)
    move_into_8 = Transformer(
        lambda entities, grid: move(entities,
                                    vector_property=collision_with_8,
                                    copy=True,
                                    extend_grid=False),
        nll=collision_with_8.nll + np.log(2),
        name=f"{'copy' if True else 'move'} them by ({collision_with_8})")
    new_entities, new_grid = move_into_8.transform(selected_entities, inp)
    assert new_grid == out
    my_entity_finder = base_entity_finder.compose(select_not_0_nor_8)
    my_predictor = Predictor(my_entity_finder, move_into_8)
    for case in task['train'] + task['test']:
        assert my_predictor.predict(case['input']) == case['output']

    my_predictor_2 = Predictor(base_entity_finder, move_into_8)


def test_entity_finder_distance():
    with open('training/' + os.listdir('training/')[9]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    base_entity_finder = EntityFinder(lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
    case = np.array(task['train'][0]['input'])
    case[:, 5] = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    case = to_tuple(case)
    original_distance = base_entity_finder.grid_distance(task['train'][0]['input'], task['train'][0]['output'])
    print(original_distance)
    new_distance = base_entity_finder.grid_distance(case, task['train'][0]['output'])
    print(new_distance)
    assert original_distance > new_distance


def test_replace_color():
    with open('training/' + os.listdir('training/')[9]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    input_grid = task['train'][0]['input']
    output_grid = task['train'][0]['output']
    entities = base_entity_finder(input_grid)
    # for selected_entity in entities:
    #     selected_entity.display()
    size_prop = Property(lambda x: x.entity.num_points(), nll=np.log(2),
                         name='the number of points',
                         output_types=frozenset({'quantity'}),
                         entity_finder=base_entity_finder,
                         requires_entity=True)
    minimum = OrdinalProperty(lambda x, n=0: nth_ordered(x, 0, use_max=False),
                              nll=0,
                              name=f'take the {ordinal(0 + 1)} largest',
                              input_types=frozenset(
                                  {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity'}))
    smallest_size = Property.from_entity_prop_and_ordinal(size_prop, minimum)
    color_4 = Property(lambda x: frozenset({4}), np.log(10) - 2, name=f'color {4}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1,
                          requires_entity=True)

    select_smallest = Selector.make_property_selector(size_prop, smallest_size, the_same=True)
    selected_entities = select_smallest.select(entities)
    assert len(selected_entities) == 1

    assert selected_entities[0].positions == {(6, 7): 5, (7, 7): 5, (8, 7): 5}

    # smallest_color = take_color.add_selector(select_smallest, ORDINAL_PROPERTIES[0])

    recolor_yellow = Transformer(lambda entities, grid,
                                        source_color_prop=take_color,
                                        target_color_prop=color_4:
                                 replace_color(entities,
                                               source_color_prop=source_color_prop,
                                               target_color_prop=target_color_prop),
                                 nll=take_color.nll + color_4.nll + np.log(2),
                                 name=f'recolor ({take_color}) with ({color_4})')
    _, prediction_grid = recolor_yellow.transform(selected_entities)

    assert base_entity_finder.grid_distance(prediction_grid, output_grid) < base_entity_finder.grid_distance(input_grid,
                                                                                                             output_grid)


def test_place_shape():
    with open('training/' + os.listdir('training/')[94]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    input_grid = task['train'][0]['input']
    output_grid = task['train'][0]['output']
    entities = base_entity_finder(input_grid)
    appearing_shapes = Counter()

    for grid in task['train']:
        output_entities = base_entity_finder(grid['output'])
        appearing_shapes += Entity.shapes(output_entities)
    desired_shape = frozenset({((0.0, 1.0), 1), ((1.0, 0.0), 1), ((-1.0, 1.0), 1), ((1.0, 1.0), 1), ((1.0, -1.0), 1),
                               ((0.0, -1.0), 1), ((-1.0, -1.0), 1), ((-1.0, 0.0), 1)})
    assert desired_shape in appearing_shapes
    color_5 = Property(lambda x: frozenset({5}), np.log(10) - 1, name=f'color {5}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    center_0 = Property(lambda x: x.entity.center(axis=0), nll=1 + np.log(2),
                        name='the center y coordinate',
                        output_types=frozenset({'y_coordinate'}),
                        entity_finder=base_entity_finder)
    center_1 = Property(lambda x: x.entity.center(axis=1), nll=1 + np.log(2),
                        name='the center x coordinate',
                        output_types=frozenset({'x_coordinate'}),
                        entity_finder=base_entity_finder)
    center = Property.create_point_property(center_0, center_1)
    desired_shape_prop = Property(lambda x: desired_shape, np.log(10) - 1, name=f'shape {desired_shape}',
                                  output_types=frozenset({'shape'}),
                                  is_constant=True,
                                  entity_finder=base_entity_finder)
    # shape_entity_prop = Property(lambda x: x.entity.shape(), 1, name=f'the shape',
    #                              output_types=frozenset({'shape'}),
    #                              entity_finder=base_entity_finder)
    place_desired_shape = Transformer(lambda entities, grid:
                                      place_shape(entities,
                                                  point_prop=center,
                                                  shape_prop=desired_shape_prop),
                                      nll=center.nll + desired_shape_prop.nll + np.log(2),
                                      name=f'place ({desired_shape_prop}) at position ({center}))')
    select_5 = Selector.make_property_selector(take_color, color_5)
    find_entities_5 = base_entity_finder.compose(select_5)
    my_predictor = Predictor(find_entities_5, place_desired_shape)
    assert my_predictor.predict(input_grid) == output_grid

    with open('training/' + os.listdir('training/')[14]) as f:
        raw_task14 = json.load(f)
    task14 = tuplefy_task(raw_task14)
    input_grid14 = task14['train'][0]['input']
    output_grid14 = task14['train'][0]['output']
    color_1 = Property(lambda x: frozenset({1}), np.log(10) - 1, name=f'color {1}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    select_1 = Selector.make_property_selector(take_color, color_1)
    # print(input_grid14)
    diamond = frozenset({((1.0, 0.0), 7), ((-1.0, 0.0), 7), ((0.0, 1.0), 7), ((0.0, -1.0), 7)})
    diamond_prop = Property(lambda x: diamond, np.log(10) - 1, name=f'shape {diamond}',
                            output_types=frozenset({'shape'}),
                            is_constant=True,
                            entity_finder=base_entity_finder)
    place_diamond = Transformer(lambda entities, grid: place_shape(entities, grid, center, diamond_prop),
                                name=f'place ({diamond_prop}) at position ({center})')
    diamond_predictor = Predictor(base_entity_finder.compose(select_1), place_diamond)
    print(diamond_predictor)
    for case in task14['train']:
        # print(case['input'])
        output_grid = diamond_predictor.predict(case['input'])
        assert (base_entity_finder.grid_distance(case['output'], diamond_predictor.predict(case['input'])) +
                base_entity_finder.grid_distance(diamond_predictor.predict(case['input']), case['input']) <=
                base_entity_finder.grid_distance(case['output'], case['input']))


def test_not_zero():
    with open('training/' + os.listdir('training/')[11]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    input_grid = task['train'][0]['input']
    output_grid = task['train'][0]['output']
    assert len(find_components(output_grid, relation='not_zero', directions=ALL_DIRECTIONS)) == 2


def test_grid_distance():
    reset_all()
    with open('training/' + os.listdir('training/')[56]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    input_grid = task['train'][0]['input']
    output_grid = task['train'][0]['output']
    same_shape = ((8, 8, 0, 8, 8, 0, 0), (0, 8, 0, 0, 8, 0, 0), (8, 8, 8, 8, 8, 8, 0), (0, 0, 0, 0, 0, 0, 0))
    different_shape = ((8, 8, 0, 8, 8, 0, 0), (0, 8, 0, 0, 8, 0, 0), (8, 8, 8, 8, 8, 8, 0), (0, 8, 0, 0, 0, 0, 0))
    # print(base_entity_finder.grid_distance(output_grid, same_shape), base_entity_finder.grid_distance(output_grid, different_shape))
    assert base_entity_finder.grid_distance(output_grid, same_shape) < base_entity_finder.grid_distance(output_grid,
                                                                                                        different_shape)

    color_0 = Property(lambda x: frozenset({0}), np.log(10) - 1, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    select_not_0 = Selector.make_property_selector(take_color, color_0, False)
    # min_x = Property(lambda x: x.entity.min_coord(axis=1), np.log(4),
    #                  name='the largest x coordinate',
    #                  output_types=frozenset({'x_coordinate'}),
    #                  entity_finder=base_entity_finder)
    # max_x = Property(lambda x: x.entity.max_coord(axis=1), np.log(4),
    #                  name='the largest x coordinate',
    #                  output_types=frozenset({'x_coordinate'}),
    #                  entity_finder=base_entity_finder)
    # x_length = Property.create_distance_property(max_x, min_x)
    x_length = Property(lambda x: x.entity.max_coord(axis=1) - x.entity.min_coord(axis=1) + 1, np.log(2),
                        name='the x length',
                        output_types=frozenset({'x_length'}),
                        entity_finder=base_entity_finder)
    zero = Property(lambda x: 0, 1,
                    name='0',
                    output_types=frozenset({'y_length'}),
                    entity_finder=base_entity_finder)
    train_input = task['train'][0]['input']
    train_output = task['train'][0]['output']
    entities = base_entity_finder(train_input)
    assert x_length(entities[1], train_input) == 3
    x_length_vect = Property.xy_length_to_vector(zero, x_length)
    # print(select_not_0.select(entities))
    _, prediction = move(select_not_0.select(entities), train_input, x_length_vect, copy=True)

    assert base_entity_finder.grid_distance(train_output, prediction) < \
           base_entity_finder.grid_distance(train_output, train_input)


def test_sequential():
    with open('training/' + os.listdir('training/')[56]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    input_grid = task['train'][0]['input']
    output_grid = task['train'][0]['output']
    color_0 = Property(lambda x: frozenset({0}), np.log(10) - 1, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    select_not_0 = Selector.make_property_selector(take_color, color_0, False)
    x_length = Property(lambda x: x.entity.max_coord(axis=1) - x.entity.min_coord(axis=1) + 1, np.log(2),
                        name='the x length',
                        output_types=frozenset({'x_length'}),
                        entity_finder=base_entity_finder)
    zero = Property(lambda x: 0, 1,
                    name='0',
                    output_types=frozenset({'y_length'}),
                    entity_finder=base_entity_finder)
    x_length_vect = Property.xy_length_to_vector(zero, x_length)
    copy_move_x_length = Transformer(lambda entities, grid: move(entities, grid, x_length_vect, copy=True),
                                     name=f'copy them by ({x_length_vect})')
    my_entity_finder = base_entity_finder.compose(select_not_0)
    cropper = Transformer(crop_entities, nll=np.log(2), name='crop them')
    single_predictor = Predictor(my_entity_finder, copy_move_x_length, parallel=False)
    predictor_1 = Predictor(my_entity_finder, copy_move_x_length)
    predictor_2 = Predictor(my_entity_finder, cropper)
    sequential_predictor = Predictor([my_entity_finder, my_entity_finder], [copy_move_x_length, cropper],
                                     parallel=False)
    composed_predictor = predictor_1.compose(predictor_2, parallel=False)
    train_input = task['train'][0]['input']
    train_output = task['train'][0]['output']
    print(composed_predictor)
    assert sequential_predictor.predict(train_input) == train_output
    assert composed_predictor.predict(train_input) == train_output


def test_combine_sorted_queues():
    @dataclass
    class DummyNLL:
        nll: float

    def nllify(queue):
        return [DummyNLL(x) for x in queue]

    queue1 = [0., 5., 12., 13.]
    queue1 = nllify(queue1)
    queue2 = [1., 2., 2., 3.5]
    queue2 = nllify(queue2)
    queue3 = [5., 30., 100.]
    queue3 = nllify(queue3)
    queue4 = []
    queue4 = nllify(queue4)

    def denllify_solution(solution):
        return [tuple((x.nll for x in xs)) for xs in solution]

    assert denllify_solution(list(combine_sorted_queues((queue1, queue2), 7.))) == [(0., 1.), (5., 1.), (0., 2.),
                                                                                    (5., 2.), (0., 2.), (5., 2.),
                                                                                    (0., 3.5)]
    assert denllify_solution(list(combine_sorted_queues((queue1, queue4), 10.))) == []
    assert denllify_solution(list(combine_sorted_queues((queue1, queue2, queue3), 7.))) == [(0., 1., 5.), (0., 2., 5.),
                                                                                            (0., 2., 5.)]


def test_is_rectangle():
    with open('training/' + os.listdir('training/')[28]) as f:
        raw_case = json.load(f)
    case = tuplefy_task(raw_case)
    rectangle = Property(lambda x: x.entity.is_a_rectangle(), 0,
                         name='is a rectangle',
                         output_types=frozenset({'bool'}),
                         entity_finder=base_entity_finder)
    color_2 = Property(lambda x: frozenset({2}), np.log(10) - 1, name=f'color {2}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    is_true = Property(lambda x: True, 0, name='True',
                       output_types=frozenset({'bool'}),
                       entity_finder=base_entity_finder)
    select_2 = Selector.make_property_selector(rectangle, color_2, True)
    first_case = case['train'][0]['input']
    entities = select_2.select(base_entity_finder(first_case))
    for entity in entities:
        assert rectangle(entity, first_case)

    is_true = Property(lambda x: True, 0, name='True',
                       output_types=frozenset({'bool'}),
                       entity_finder=base_entity_finder)
    select_rectangle = Selector.make_property_selector(rectangle, is_true, True)
    rect_entities = select_rectangle.select(base_entity_finder(first_case))
    assert len(rect_entities) == 1
    # for entity in rect_entities:
    #     entity.display()
    crop = Transformer(lambda entities, grid, offsets=(1, -1, 1, -1):
                       crop_entities(entities, grid, offsets=offsets),
                       nll=np.log(2) + sum(
                           (abs(offset) for offset in (1, -1, 1, -1))) * np.log(2),
                       name='crop them')
    _, output = crop.transform(rect_entities)
    assert output == case['train'][0]['output']


def test_replace_color_entity_frame():
    with open('training/' + os.listdir('training/')[80]) as f:
        raw_case = json.load(f)
    case = tuplefy_task(raw_case)

    color_0 = Property(lambda x: frozenset({0}), np.log(10) - 1, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_1 = Property(lambda x: frozenset({1}), np.log(10) - 1, name=f'color {1}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    color_8 = Property(lambda x: frozenset({8}), np.log(10) - 1, name=f'color {1}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1)
    select_8 = Selector.make_property_selector(take_color, color_8, True)

    select_not_0 = Selector.make_property_selector(take_color, color_0, False)

    color_frame_blue = Transformer(lambda entities, grid, offsets=(0, 0, 0, 0):
                                   replace_colors_in_entities_frame(entities, grid=None,
                                                                    offsets=offsets,
                                                                    source_color_prop=color_0,
                                                                    target_color_prop=color_1),
                                   name=f'replace ({color_0}) with ({color_1}) in a box around them with offsets {(0, 0, 0, 0)}')
    first_case = case['train'][0]['input']
    entity_finder = base_entity_finder.compose(select_8)
    my_predictor = Predictor(entity_finder, color_frame_blue)
    # print(my_predictor.predict(first_case))
    assert my_predictor.predict(first_case) == case['train'][0]['output']
    assert my_predictor.predict(case['test'][0]['input']) == case['test'][0]['output']

    entity_finder_2 = base_entity_finder.compose(select_not_0)
    my_predictor_2 = Predictor(entity_finder_2, color_frame_blue)
    assert my_predictor_2.predict(first_case) == case['train'][0]['output']
    assert my_predictor_2.predict(case['test'][0]['input']) == case['test'][0]['output']
    print(my_predictor_2)


def test_is_contained():
    with open('training/' + os.listdir('training/')[181]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    case = task['train'][0]
    entities = base_entity_finder(case['input'])
    assert surrounded(entities[-4], entities[0])


def test_reflect_about_line():
    with open('training/' + os.listdir('training/')[86]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    inp = task['train'][0]['input']
    out = task['train'][0]['output']
    vert_center_line = Property(lambda x: (float(np.array(x.grid).shape[1] - 1) / 2., 1.),
                                np.log(4),
                                name='the vertical center line',
                                output_types=frozenset({'line'}),
                                entity_finder=base_entity_finder)
    entities = base_entity_finder(inp)
    new_entities, new_grid = reflect_about_line(entities, inp, vert_center_line)
    # original = ((2, 2, 1),
    #             (2, 1, 2),
    #             (2, 8, 1))
    assert new_grid == ((1, 2, 2), (2, 1, 2), (1, 8, 2))
    horiz_center_line = Property(lambda x: (float(np.array(x.grid).shape[0] - 1) / 2., 0.),
                                 np.log(4),
                                 name='the horizontal center line',
                                 output_types=frozenset({'line'}),
                                 entity_finder=base_entity_finder)
    new_entities, new_grid = reflect_about_line(entities, inp, horiz_center_line)
    assert new_grid == ((2, 8, 1), (2, 1, 2), (2, 2, 1))
    back_diagonal_center_line = Property(lambda x: (0., -0.5),
                                         np.log(4),
                                         name='the back diagonal center line',
                                         output_types=frozenset({'line'}),
                                         entity_finder=base_entity_finder)
    new_entities, new_grid = reflect_about_line(entities, inp, back_diagonal_center_line)
    assert new_grid == ((2, 2, 2), (2, 1, 8), (1, 2, 1))
    forward_diagonal_center_line = Property(lambda x: (float(np.array(x.grid).shape[1] - 1.) / 2., 0.5),
                                            np.log(4),
                                            name='the forward diagonal center line',
                                            output_types=frozenset({'line'}),
                                            entity_finder=base_entity_finder)
    new_entities, new_grid = reflect_about_line(entities, inp, forward_diagonal_center_line)
    assert new_grid == ((1, 2, 1), \
                        (8, 1, 2), \
                        (2, 2, 2))
    new_entities, new_grid = reflect_about_line(entities, inp, vert_center_line)
    new_entities, new_grid = reflect_about_line(new_entities, new_grid, horiz_center_line)
    assert new_grid == out

    new_entities, new_grid = rotate_via_reflects(entities, inp, vert_center_line, horiz_center_line)
    assert len(new_entities) == 3
    assert new_grid == out
    transformer = Transformer(lambda entities, grid:
                              rotate_via_reflects(entities, grid, vert_center_line, horiz_center_line),
                              nll=vert_center_line.nll + horiz_center_line.nll + np.log(2),
                              name=f'reflect about ({vert_center_line}) then ({horiz_center_line})')
    entities = base_entity_finder(inp)
    # new_entities, new_grid = transformer.transform(entities, inp)
    my_predictor = Predictor(base_entity_finder, transformer, parallel=False)
    assert my_predictor.predict(inp) == out

    grid_center = Property(
        lambda x: (float(np.array(x.grid).shape[0] - 1) / 2., float(np.array(x.grid).shape[1] - 1) / 2.),
        0,
        name='the center point of the grid',
        output_types=frozenset({'point'}),
        entity_finder=base_entity_finder)
    new_entities, new_grid = rotate_about_point(entities, inp, grid_center, quarter_steps=2)
    assert new_grid == out


def test_klein_vier():
    with open('training/' + os.listdir('training/')[82]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    inp = task['train'][0]['input']
    out = task['train'][0]['output']
    vertical_right_line = Property(lambda x: (float(np.array(x.grid).shape[1] - 0.5), 1.),
                                   np.log(4),
                                   name='the vertical right-most line',
                                   output_types=frozenset({'line'}),
                                   entity_finder=base_entity_finder)
    bottom_line = Property(lambda x: (float(np.array(x.grid).shape[0] - 0.5), 0.),
                           np.log(4),
                           name='the horizontal bottom-most line',
                           output_types=frozenset({'line'}),
                           entity_finder=base_entity_finder)
    entities = base_entity_finder(inp)
    new_entities, new_grid = reflect_about_line(entities, inp, vertical_right_line, copy_entities=True)
    new_entities, new_grid = reflect_about_line(new_entities, new_grid, bottom_line)
    assert new_grid == out

    new_entities, new_grid = apply_klein_vier_group(entities, inp, vertical_right_line, bottom_line)
    assert new_grid == out


def test_rotation_group():
    with open('training/' + os.listdir('training/')[193]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    inp = task['train'][0]['input']
    out = task['train'][0]['output']
    vertical_right_line = Property(lambda x: (float(np.array(x.grid).shape[1] - 0.5), 1.),
                                   np.log(4),
                                   name='the vertical right-most line',
                                   output_types=frozenset({'line'}),
                                   entity_finder=base_entity_finder)
    back_diagonal_line = Property(lambda x: (0, -0.5),
                                  np.log(4),
                                  name='the back diagonal center line',
                                  output_types=frozenset({'line'}),
                                  entity_finder=base_entity_finder)
    entities = base_entity_finder(inp)

    new_entities, new_grid = apply_rotation_group_old(entities, inp, line_prop1=back_diagonal_line,
                                                      line_prop2=vertical_right_line)
    assert new_grid == out
    bottom_right_corner = Property(
        lambda x: (float(np.array(x.grid).shape[0]) - 0.5, float(np.array(x.grid).shape[1]) - 0.5), nll=np.log(4),
        name='the grid bottom-right corner point',
        output_types=frozenset({'point'}),
        entity_finder=base_entity_finder)
    new_entities, new_grid = apply_rotation_group(entities, inp, bottom_right_corner)
    assert new_grid == out


def test_144():
    with open('training/' + os.listdir('training/')[144]) as f:
        raw_task = json.load(f)
    task = tuplefy_task(raw_task)
    inp = task['train'][0]['input']
    out = task['train'][0]['output']
    color_0 = Property(lambda x: frozenset({0}), np.log(10) - 1, name=f'color {0}',
                       output_types=frozenset({'color'}),
                       entity_finder=base_entity_finder)
    take_color = Property(lambda x: x.entity.colors(),
                          name='the colors',
                          output_types=frozenset({'color'}),
                          entity_finder=base_entity_finder,
                          nll=1,
                          requires_entity=True)
    select_0 = Selector.make_property_selector(take_color, color_0, True)
    entities = base_entity_finder(inp)
    num_points = Property(lambda x: x.entity.num_points(), nll=np.log(2),
                          name='the number of points',
                          output_types=frozenset({'quantity'}),
                          entity_finder=base_entity_finder,
                          requires_entity=True)
    smallest = OrdinalProperty(lambda x, n=0: nth_ordered(x, 0, use_max=False),
                               nll=0,
                               name=f'take the {ordinal(1)} smallest',
                               input_types=frozenset(
                                   {'x_length', 'y_length', 'x_coordinate', 'y_coordinate', 'quantity'}))
    my_selector = Selector.make_property_selector(num_points, num_points.add_selector(select_0, smallest))
    assert len(my_selector.select(entities)) == 2


def test_test_case():
    with open('training/' + os.listdir('training/')[7]) as f:
        raw_case7 = json.load(f)
    case7 = tuplefy_task(raw_case7)
    predictors = core_functions.test_case(case7)
    test_input = case7['test'][0]['input']
    test_output = case7['test'][0]['output']
    predictions = [predictor.predict(test_input) for predictor in predictors]
    # display_case(predictions[0])
    assert predictions[0] == test_output
