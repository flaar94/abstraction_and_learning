import heapq
import itertools
import time

import numpy as np

import atomic_objects
from atomic_objects import generate_base_properties, move, crop_entities, replace_colors_in_entities_frame, \
    replace_color, place_shape, find_components, generate_base_relations, reflect_about_line, \
    apply_klein_vier_group, apply_rotation_group, rotate_about_point
from classes import EntityFinder, Selector, ORDINAL_PROPERTIES, Property, Relation, Transformer, Predictor, Entity
from constants import MAX_SMALL_TIME, MAKE_PROPERTY_LIST, ALLOW_COMPOSITE_SELECTORS, \
    ALL_DIRECTIONS, MAX_PREDICTORS, MAX_LARGE_TIME, MAX_PARTIAL_PREDICTORS, SELECT_NOT_0_NLL, DIM_CHANGE_PENALTY, \
    NEW_COLOR_BONUS, SELECTOR_MAX_NLL_CORRECTION, POINT_PROP_COST, PROPERTY_CONSTRUCTION_COST, \
    SELECTOR_CONSTRUCTION_COST
from my_utils import filter_unlikelies, combine_sorted_queues
from nll_functions import combine_property_selector_nll, combine_selector_nll, combine_relation_selector_nll, \
    combine_pair_selector_nll


def reset_all():
    Relation.reset()
    Property.reset()
    Selector.reset()
    Predictor.reset()
    Entity.reset()
    # global move_entity_cache
    # move_entity_cache = {}
    atomic_objects.adjacent_direction_cache = {}
    atomic_objects.find_color_entities_cache = {}
    atomic_objects.collision_directions_cache = {}


def selector_iterator(task: dict, base_entity_finder: EntityFinder, max_nll: float = 20.):
    start_time = time.perf_counter()
    inputs = [case['input'] for case in task['train']]
    inputs.extend([case['input'] for case in task['test']])
    earlier_selectors = []
    rounds = 0
    grid_properties, entity_properties = generate_base_properties(task, base_entity_finder)
    grid_properties, entity_properties = filter_unlikelies(grid_properties, max_nll), filter_unlikelies(
        entity_properties, max_nll)

    entity_properties.sort()
    entity_properties = [entity_property for entity_property in entity_properties
                         if entity_property.validate_and_register(task)]

    # # Make all grid properties that are the same for all training examples less likely
    # for grid_property in grid_properties:
    #     if len({grid_property(None, case['input']) for case in task['train']}) == 1 and not grid_property.is_constant:
    #         grid_property.nll += 1

    grid_properties = filter_unlikelies(grid_properties, max_nll)
    grid_properties.sort()
    grid_properties = [grid_property for grid_property in grid_properties if grid_property.validate_and_register(task)]

    entity_properties.sort()
    grid_properties.sort()

    all_properties = entity_properties + grid_properties
    all_properties.sort()

    trivial_selector = Selector(lambda entity, grid: True, name='true', nll=0)
    queue = [trivial_selector]
    for entity_property in entity_properties:
        for target_property in all_properties:
            if (entity_property.count != target_property.count) and (
                    combine_selector_nll(entity_property, target_property) <= max_nll):
                for the_same in [True, False]:
                    new_selector = Selector.make_property_selector(entity_property, target_property, the_same)
                    if str(entity_property) == "the colors" and str(target_property) == "color 0" and not the_same:
                        # Specially make selecting all the non-0 color entities more likely
                        new_selector.nll = SELECT_NOT_0_NLL
                    if new_selector.validate_and_register(task, base_entity_finder, max_nll):
                        heapq.heappush(queue, new_selector)

    earlier_selectors = []
    while queue:
        my_selector = heapq.heappop(queue)

        Selector.previous_selectors.add(str(my_selector))
        yield my_selector
        rounds += 1
        # print(my_selector, my_selector.nll)
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            return

        # Makes properties where the selection produces unique values in all the training cases
        new_grid_properties = []
        common_property_indices = []
        for i, training_case in enumerate(task['train']):
            training_grid = training_case['input']
            training_entities = base_entity_finder(training_grid)
            training_selected_entities = my_selector.select(training_entities)
            if not training_selected_entities:
                common_property_indices = [set()]
                break
            common_properties = {(i, prop(training_selected_entities[0], training_grid))
                                 for i, prop in enumerate(entity_properties) if
                                 prop.count not in my_selector.fixed_properties}
            for entity in training_selected_entities:
                common_properties &= {(i, prop(entity, training_grid)) for i, prop in enumerate(entity_properties)
                                      if prop.count not in my_selector.fixed_properties}
            # Extract which entity properties give the same result for all entities selected
            common_property_indices.append({prop[0] for prop in common_properties})
        valid_common_properties = set.intersection(*common_property_indices)

        common_grid_properties = [entity_properties[index].add_selector(my_selector) for index in
                                  valid_common_properties]
        common_grid_properties.sort()

        for prop in (prop for prop in entity_properties if prop.count not in my_selector.fixed_properties):
            for ordinal_property in ORDINAL_PROPERTIES:
                if combine_property_selector_nll(prop, my_selector, ordinal_property) <= max_nll:
                    grid_property = prop.add_selector(my_selector, ordinal_property)
                    if grid_property in common_grid_properties:
                        # not a problem
                        grid_property.nll -= 1

                    if grid_property.validate_and_register(task):
                        new_grid_properties.append(grid_property)

        # Makes sure properties are potentially valid on training examples
        # all_cases = task['train'] + task['test']
        # for i, case in enumerate(all_cases):
        #     training_grid = case['input']
        #     training_entities = base_entity_finder(training_grid)
        #     valid_indices = []
        #     for j, entity_prop in enumerate(new_entity_properties):
        #         for entity in training_entities:
        #             if entity_prop(entity, training_grid) is None:
        #                 break
        #         else:
        #             valid_indices.append(j)

        #     new_entity_properties = [new_entity_properties[valid_index] for valid_index in valid_indices]

        # # Makes relational entity properties from a chosen entity to some selection
        # new_entity_properties = [
        #                          for relation, ordinal_property in
        #                          itertools.product(Relation.relations, ORDINAL_PROPERTIES)
        #                          if combine_relation_selector_nll(relation, my_selector,
        #                                                           ordinal_property) <= max_nll
        #                          ]
        # # Now register the properties
        # for prop in new_entity_properties:
        #     prop.validate_and_register(task)

        new_entity_properties = []
        for relation, ordinal_property in combine_sorted_queues((Relation.relations, ORDINAL_PROPERTIES),
                                                                max_nll - my_selector.nll - PROPERTY_CONSTRUCTION_COST):
            if combine_relation_selector_nll(relation, my_selector, ordinal_property) < max_nll:
                prop = Property.from_relation_selector(relation, my_selector,
                                                       entity_finder=base_entity_finder,
                                                       ordinal_property=ordinal_property,
                                                       register=False)
                if prop.validate_and_register(task):
                    new_entity_properties.append(prop)

        # Make new ordinal grid properties
        for entity_prop, selector, ordinal_prop in combine_sorted_queues((new_entity_properties, earlier_selectors,
                                                                          ORDINAL_PROPERTIES),
                                                                         max_nll - PROPERTY_CONSTRUCTION_COST):
            if combine_property_selector_nll(entity_prop, selector, ordinal_prop) <= max_nll:
                grid_property = entity_prop.add_selector(selector, ordinal_prop)
                if grid_property.validate_and_register(task):
                    new_grid_properties.append(grid_property)

        new_grid_properties.sort()
        # Now add in the new selectors to the queue
        for entity_prop, new_prop in combine_sorted_queues((entity_properties, new_grid_properties),
                                                           max_nll - SELECTOR_CONSTRUCTION_COST):
            # Makes a new selector from the base property and the new property
            if combine_selector_nll(entity_prop, new_prop) <= max_nll:
                for the_same in [True, False]:
                    new_selector = Selector.make_property_selector(entity_prop, new_prop, the_same=the_same)
                    if new_selector.validate_and_register(task, base_entity_finder, max_nll):
                        heapq.heappush(queue, new_selector)

        grid_properties.extend(new_grid_properties)
        grid_properties.sort()

        all_properties = grid_properties + entity_properties
        all_properties.sort()

        for new_prop, grid_prop in combine_sorted_queues((new_entity_properties, all_properties),
                                                         max_nll - SELECTOR_CONSTRUCTION_COST):
            # Makes a new selector from the base property and the new property
            if combine_selector_nll(new_prop, grid_prop) <= max_nll:
                for the_same in [True, False]:
                    new_selector = Selector.make_property_selector(new_prop, grid_prop, the_same=the_same)
                    if new_selector.validate_and_register(task, base_entity_finder, max_nll):
                        heapq.heappush(queue, new_selector)

        entity_properties.extend(new_entity_properties)
        entity_properties.sort()

        earlier_selectors.append(my_selector)
        earlier_selectors.sort()

        # if not (rounds % 10):
        #     print(f'my_selector.nll = {my_selector.nll}')
        #     print(f'len(entity_properties) = {len(entity_properties)}')
        #     print(f'len(properties) = {len(grid_properties)}')


def create_predictor_queue(task, max_nll, base_entity_finder, allow_selector_pairs=False):
    for i, example in enumerate(task['train']):
        if len(base_entity_finder(example['input'])) == 0:
            return []
    start_time = time.perf_counter()
    selector_list = list(selector_iterator(task, base_entity_finder, max_nll=max_nll - SELECTOR_MAX_NLL_CORRECTION))
    selector_list.sort()
    print(f"selecting time = {time.perf_counter() - start_time}")

    if MAKE_PROPERTY_LIST:
        Property.property_list.sort()
        print(f"len(Property.property_list) = {len(Property.property_list)}")

    print(f'built selector list (1), length={len(selector_list)}')

    if allow_selector_pairs:
        for selector1, selector2 in itertools.combinations(selector_list, 2):
            if combine_pair_selector_nll(selector1, selector2) < max_nll - SELECTOR_MAX_NLL_CORRECTION:
                new_selector = selector1.intersect(selector2)
                if new_selector.validate_and_register(task, base_entity_finder, max_nll - SELECTOR_MAX_NLL_CORRECTION):
                    selector_list.append(new_selector)
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            print('Out of time')
            return []

    selector_list.sort()
    print(f'built selector list (2), length={len(selector_list)}')
    # print('Time after selectors created = ', time.perf_counter() - start_time)
    # Create distance properties out of coordinate properties
    Property.of_type['x_coordinate'].sort()
    Property.of_type['y_coordinate'].sort()
    # LENGTH PROPERTIES
    x_length_props = (prop1.create_distance_property(prop2, register=False)
                      for prop1, prop2 in
                      combine_sorted_queues((Property.of_type['x_coordinate'], Property.of_type['x_coordinate']),
                                            max_nll - np.log(2))
                      if prop1.count != prop2.count
                      and (not prop1.is_constant or not prop2.is_constant))
    y_length_props = (prop1.create_distance_property(prop2, register=False)
                      for prop1, prop2 in
                      combine_sorted_queues((Property.of_type['y_coordinate'], Property.of_type['y_coordinate']),
                                            max_nll - np.log(2))
                      if prop1.count != prop2.count
                      and (not prop1.is_constant or not prop2.is_constant))

    length_props = sorted(list(itertools.chain(x_length_props, y_length_props)))
    for length_prop in length_props:
        length_prop.validate_and_register(task,
                                          extra_validation=lambda output_signature:
                                          all((value.is_integer() for value in output_signature)))

    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []

    Property.of_type['x_length'].sort()
    Property.of_type['y_length'].sort()

    # Constructing point properties
    point_props = [Property.create_point_property(prop1, prop2, register=False)
                   for prop1, prop2 in
                   combine_sorted_queues((Property.of_type['y_coordinate'], Property.of_type['x_coordinate']),
                                         max_nll - 2 - POINT_PROP_COST)]
    for point_prop in point_props:
        point_prop.validate_and_register(task)
    Property.of_type['point'].sort()

    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []

    # Constructing vector properties

    # Create vectors from single lengths
    for axis, name in enumerate(['y_length', 'x_length']):
        for length in Property.of_type[name]:
            vect_prop = Property.length_to_vector(length, axis, register=False)
            vect_prop.validate_and_register(task)

    # Create vectors from pairs of points
    for source_pt, target_pt in combine_sorted_queues((Property.of_type['point'],
                                                       Property.of_type['point']),
                                                      max_nll - np.log(2)):
        vect_prop = Property.points_to_vector(source_pt, target_pt, register=False)
        vect_prop.validate_and_register(task,
                                        extra_validation=lambda output_signature: all(
                                            (value[i].is_integer() for value in output_signature for i in range(2))))
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            print('Out of time')
            return []

    penalize_dim_change = True if all((len(case['input']) == len(case['output']) and len(case['input'][0]) == len(case['output'][0]) for case in task['train'])) else False

    transformers = (
        # 34
        Transformer(
            lambda entities, grid, vector_prop=vector_prop, copy=copy: move(entities,
                                                                            vector_property=vector_prop,
                                                                            copy=copy,
                                                                            extend_grid=not penalize_dim_change),
            nll=vector_prop.nll + np.log(2),
            name=f"{'copy' if copy else 'move'} them by ({vector_prop})")
        for vector_prop in Property.of_type['vector']
        for copy in [True, False] if vector_prop.nll + np.log(2) <= max_nll
    )
    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []
    Property.of_type['color'].sort()
    # 35
    composite_transformers = (Transformer(lambda entities, grid, offsets=offsets:
                                          crop_entities(entities, grid, offsets=offsets),
                                          nll=np.log(2) + sum(
                                              (abs(offset) for offset in offsets)) * np.log(2) + \
                                              penalize_dim_change * DIM_CHANGE_PENALTY,
                                          name=f'crop them with offset {offsets}')
                              for offsets in itertools.product([-1, 0, 1], repeat=4)
                              if np.log(2) + sum((abs(offset) for offset in offsets)) * np.log(2) + \
                              penalize_dim_change * DIM_CHANGE_PENALTY < max_nll)
    if any(({entry for row in case['input'] for entry in row} == {entry for row in case['output'] for entry in row}
            for case in task['train'])):
        new_colors = False
    else:
        new_colors = True

    # 36
    composite_transformers = itertools.chain(composite_transformers,
                                             (Transformer(lambda entities, grid, offsets=offsets,
                                                                 source_color_prop=source_color_prop,
                                                                 target_color_prop=target_color_prop:
                                                          replace_colors_in_entities_frame(entities, grid,
                                                                                           offsets=offsets,
                                                                                           source_color_prop=source_color_prop,
                                                                                           target_color_prop=target_color_prop),
                                                          nll=np.log(
                                                              2) + source_color_prop.nll + target_color_prop.nll + sum(
                                                              (abs(offset) for offset in offsets)) * np.log(2)
                                                              - new_colors * NEW_COLOR_BONUS,
                                                          name=f'replace ({source_color_prop}) '
                                                               f'with ({target_color_prop}) '
                                                               f'in a box around them with offsets {offsets}')
                                              for source_color_prop, target_color_prop in
                                              combine_sorted_queues(
                                                  (Property.of_type['color'], Property.of_type['color']),
                                                  max_nll=max_nll - np.log(2) + new_colors * NEW_COLOR_BONUS)
                                              for offsets in [(0, 0, 0, 0), (1, -1, 1, -1)]))
    # 37
    composite_transformers = itertools.chain(composite_transformers,
                                             (Transformer(lambda entities, grid,
                                                                 source_color_prop=source_color_prop,
                                                                 target_color_prop=target_color_prop:
                                                          replace_color(entities,
                                                                        source_color_prop=source_color_prop,
                                                                        target_color_prop=target_color_prop),
                                                          nll=source_color_prop.nll + target_color_prop.nll + np.log(
                                                              2) - new_colors * NEW_COLOR_BONUS,
                                                          name=f'recolor ({source_color_prop}) with ({target_color_prop})')
                                              for source_color_prop, target_color_prop in
                                              combine_sorted_queues(
                                                  (Property.of_type['color'], Property.of_type['color']),
                                                  max_nll - np.log(2) + new_colors * NEW_COLOR_BONUS)))
    Property.of_type['shape'].sort()
    # 38
    transformers = itertools.chain(transformers,
                                   (Transformer(lambda entities, grid,
                                                       point_prop=point_prop,
                                                       shape_prop=shape_prop,
                                                       color_strategy=color_strategy:
                                                place_shape(entities,
                                                            point_prop=point_prop,
                                                            shape_prop=shape_prop,
                                                            color_strategy=color_strategy),
                                                nll=point_prop.nll + shape_prop.nll + np.log(2),
                                                name=f'place ({shape_prop}) at position ({point_prop})')
                                    for point_prop, shape_prop in
                                    combine_sorted_queues((Property.of_type['point'],
                                                           Property.of_type['shape']), max_nll - np.log(2))
                                    for color_strategy in ('original', 'extend_non_0', 'replace_0'))
                                   )

    reflections = [
        Transformer(lambda entities, grid, line_prop=line_prop: reflect_about_line(entities,
                                                                                   line_prop=line_prop,
                                                                                   extend_grid=not penalize_dim_change),
                    nll=line_prop.nll + np.log(2), name=f'reflect about {line_prop}') for line_prop in
        Property.of_type['line']]

    # rotations = [Transformer(lambda entities, grid, line_prop1=line_prop1, line_prop2=line_prop2:
    #                          rotate_via_reflects(entities, grid, line_prop1, line_prop2,
    #                                              extend_grid=not penalize_dim_change),
    #                          nll=line_prop1.nll + line_prop2.nll + np.log(2),
    #                          name=f'reflect about ({line_prop1}) then ({line_prop2})')
    #              for line_prop1, line_prop2 in itertools.permutations(Property.of_type['line'], 2)
    #              if line_prop1.nll + line_prop2.nll + np.log(2) < max_nll]

    rotations = [Transformer(lambda entities, grid, point_prop=point_prop:
                             rotate_about_point(entities, grid, point_prop,
                                                extend_grid=not penalize_dim_change),
                             nll=point_prop.nll + np.log(2),
                             name=f'rotate {steps} steps clockwise about {point_prop}')
                 for point_prop in Property.of_type['point']
                 for steps in range(1, 4)
                 if point_prop.nll + np.log(2) < max_nll]

    # rotation_groups = [Transformer(lambda entities, grid, line_prop1=line_prop1, line_prop2=line_prop2:
    #                                apply_rotation_group_old(entities, grid, line_prop1, line_prop2,
    #                                                         extend_grid=not penalize_dim_change),
    #                                nll=line_prop1.nll + line_prop2.nll + 2 * np.log(2),
    #                                name=f'rotate via ({line_prop1}, {line_prop2}) 4 times')
    #                    for line_prop1, line_prop2 in itertools.permutations(Property.of_type['line'], 2)
    #                    if line_prop1.nll + line_prop2.nll + np.log(2) + 2 * np.log(2) < max_nll]

    rotation_groups = [Transformer(lambda entities, grid, point_prop=point_prop:
                                   apply_rotation_group(entities, grid, point_prop,
                                                        extend_grid=not penalize_dim_change),
                                   nll=point_prop.nll + np.log(2),
                                   name=f'apply full rotation group about {point_prop}')
                       for point_prop in Property.of_type['point']
                       if point_prop.nll + 2 * np.log(2) < max_nll]

    klein_viers = [Transformer(lambda entities, grid, line_prop1=line_prop1, line_prop2=line_prop2:
                               apply_klein_vier_group(entities, grid, line_prop1, line_prop2,
                                                      extend_grid=not penalize_dim_change),
                               nll=line_prop1.nll + line_prop2.nll + 2 * np.log(2),
                               name=f'apply the group generated by {line_prop1} and {line_prop2}')
                   for line_prop1, line_prop2 in itertools.combinations(Property.of_type['line'], 2)
                   if line_prop1.nll + line_prop2.nll + 2 * np.log(2) < max_nll]

    transformers = itertools.chain(transformers,
                                   reflections,
                                   rotations,
                                   rotation_groups,
                                   klein_viers
                                   )
    # print('transformer lengths:')
    # print(len(reflections))
    # print(len(Property.of_type['point']))
    # print(len(rotations))
    # print(len(rotation_groups))
    # print(len(klein_viers))

    # rotations.sort()
    # for rotation in rotations:
    #     print(rotation, rotation.nll)

    # for shape_prop in Property.of_type['shape']:
    #     print(shape_prop)
    # transformers = itertools.chain(transformers,
    #                                (Transformer(lambda entities, grid, shape_prop=shape_prop:
    #                                                           output_shape_as_grid(entities, grid, shape_prop),
    #                                                           nll=shape_prop.nll + np.log(2))
    #                                 for shape_prop in Property.of_type['shape']
    #                                 if shape_prop.nll + np.log(2) <= max_nll-2 and not shape_prop.requires_entity))
    # print('Time after transformers list =', time.perf_counter() - start_time)
    # print(f"sys.getsizeof(transformers) = {sys.getsizeof(transformers)}", f"len(transformers) = {len(transformers)}")

    transformers = itertools.chain(transformers, composite_transformers)
    transformers = list(transformers)
    transformers.sort()

    if not ALLOW_COMPOSITE_SELECTORS:
        transformers = itertools.chain(transformers, composite_transformers)
        transformers = list(transformers)
        transformers.sort()
        entity_finders = [base_entity_finder.compose(selector) for selector in selector_list if
                          selector.nll + base_entity_finder.nll <= max_nll]
        predictor_queue = [Predictor(entity_finder, transformer)
                           for entity_finder, transformer in
                           combine_sorted_queues((entity_finders, transformers), max_nll)]
    else:
        composite_transformers = list(composite_transformers)
        transformers = list(transformers)
        transformers.sort()
        composite_transformers.sort()
        entity_finders_noncomposite = [base_entity_finder.compose(selector, False) for selector in
                                       selector_list if selector.nll + base_entity_finder.nll <= max_nll]

        entity_finders_composite = entity_finders_noncomposite + \
                                   [base_entity_finder.compose(selector, True) for selector in selector_list
                                    if selector.nll + base_entity_finder.nll <= max_nll]

        entity_finders_composite.sort()

        predictor_queue = [Predictor(entity_finder, transformer)
                           for entity_finder, transformer in
                           combine_sorted_queues((entity_finders_composite, transformers), max_nll)]
        predictor_queue += [Predictor(entity_finder, transformer)
                            for entity_finder, transformer in
                            combine_sorted_queues((entity_finders_noncomposite, composite_transformers), max_nll)]
    # print('Time after predictor queue =', time.perf_counter() - start_time)
    # for key, properties in Property.of_type.items():
    #     print(key, len(properties))
    if time.perf_counter() - start_time > MAX_SMALL_TIME:
        print('Out of time')
        return []
    print(f'built predictor queue, length = {len(predictor_queue)}')
    predictor_queue.sort()
    print('sorted predictor queue')

    return predictor_queue


def test_case(task,
              max_nll=10.,
              base_entity_finder=EntityFinder(lambda grid: find_components(grid, directions=ALL_DIRECTIONS)),
              allow_multiple_predictors=False,
              allow_selector_pairs=False):
    start_time = time.perf_counter()
    print(len(atomic_objects.collision_directions_cache))
    reset_all()
    print(len(atomic_objects.collision_directions_cache))
    base_entity_finder.reset()
    generate_base_relations()
    inputs = [case['input'] for case in task['train']]
    inputs.extend([case['input'] for case in task['test']])
    predictor_queue = create_predictor_queue(task=task,
                                             max_nll=max_nll,
                                             base_entity_finder=base_entity_finder,
                                             allow_selector_pairs=allow_selector_pairs)
    my_count = 0

    if len(predictor_queue) > MAX_PREDICTORS:
        predictor_queue = predictor_queue[:MAX_PREDICTORS]

    predictions = set()
    predictor_indices = []
    good_predictors = []

    for i, predictor in enumerate(predictor_queue):
        if time.perf_counter() - start_time > MAX_SMALL_TIME:
            print('Out of time')
            return []

        if all((predictor.predict(case['input']) == case['output'] for case in task['train'])):
            good_predictors.append(predictor)
            if len(good_predictors) == 3:
                break
        if not good_predictors and allow_multiple_predictors:
            prediction = tuple((predictor.predict(case['input']) for case in task['train'] + task['test']))
            if () not in prediction and prediction not in predictions:
                predictions.add(prediction)
                predictor_indices.append(i)

    print(f"before filtering: {len(predictor_queue)}")
    predictor_queue = [predictor_queue[i] for i in predictor_indices]
    print(f"after filtering: {len(predictor_queue)}")
    # If there is no single predictor solution, expand our search to multiple-predictor solutions
    partial_predictors = []
    if allow_multiple_predictors:
        previous_predictor_outputs = set()
        original_grids = [case['input'] for case in task['train'] + task['test']]
        depth = 0
        while not good_predictors and depth < 10:
            # print(f'len(adjacent_direction_cache)= {len(adjacent_direction_cache)}')
            # print(f'len(find_color_entities_cache)= {len(find_color_entities_cache)}')
            # print(f'len(collision_directions_cache)= {len(collision_directions_cache)}')
            if time.perf_counter() - start_time > MAX_LARGE_TIME:
                print('Out of time')
                return []
            print(f'round {my_count + 1}')
            old_partial_predictors = partial_predictors
            partial_predictors = []
            for predictor in predictor_queue:
                if partial_predictors and predictor.nll > partial_predictors[0][0].nll + 3:
                    break
                # desired_string = 'replace (color 0) with (color 1) in a box around them'
                # if desired_string in str(predictor):
                #     print(predictor, predictor.nll)
                if not old_partial_predictors:
                    # On the first pass, we create a trivial list of just the predictor
                    predictors = [(predictor, original_grids)]
                else:
                    # On future passes, we combine the "predictor" index with all of the useful partial predictors
                    predictors = ((partial_predictor.compose(predictor, parallel), old_predictions) for
                                  partial_predictor, old_predictions in
                                  old_partial_predictors for parallel in ([True, False]
                                                                          if depth == 1 else [None]))
                for new_predictor, old_predictions in predictors:
                    # test_output = [new_predictor.predict(train_case['input']) for train_case in task['test']]
                    # if full_output in previous_predictor_outputs:
                    #     continue
                    # else:
                    #     previous_predictor_outputs.add(full_output)
                    # if 'place' in str(predictor):
                    #     print(predictor)
                    no_superfluous_changes = all(
                        (base_entity_finder.grid_distance(case['output'], new_predictor.predict(case['input'])) +
                         base_entity_finder.grid_distance(new_predictor.predict(case['input']), old_prediction) ==
                         base_entity_finder.grid_distance(case['output'], old_prediction)
                         for case, old_prediction in
                         zip(task['train'], old_predictions)))
                    # Note: old predictions also contains the test predictions, but zip cuts this part off

                    if no_superfluous_changes:
                        # We test if the prediction is on a "straight line" between the input and the output to prevent
                        # the algorithm making lots of horizontal changes
                        new_predicted = [new_predictor.predict(train_case['input']) for train_case in task['train']]
                        test_output = [new_predictor.predict(train_case['input']) for train_case in task['test']]
                        if () in test_output:
                            continue
                        full_output = tuple(new_predicted + test_output)
                        if full_output in previous_predictor_outputs:
                            continue
                        else:
                            previous_predictor_outputs.add(full_output)

                        if new_predictor.entity_finders[-1].nll + new_predictor.transformers[-1].nll > 5. or depth < 3:
                            min_changes = 2
                        else:
                            min_changes = 1
                        num_changes = 0
                        for prediction, old_prediction in zip(full_output, old_predictions):
                            if prediction != old_prediction:
                                num_changes += 1
                                if num_changes >= min_changes:
                                    non_trivial_change = True
                                    break
                        else:
                            non_trivial_change = False

                        if non_trivial_change:
                            # If we've made a change, by the previous equality we must have made an improvement

                            if all((prediction == case['output'] for prediction, case in
                                    zip(new_predicted, task['train']))):
                                # If there's a perfect match, we add it to the good_predictors list
                                good_predictors.append(new_predictor)
                                if len(good_predictors) == 3:
                                    break
                                else:
                                    continue

                            if not good_predictors:
                                # If there's no perfect match, we look for a partial match
                                new_predictor.net_distance_ = sum(
                                    (base_entity_finder.grid_distance(case['output'], prediction)
                                     for prediction, case in zip(new_predicted, task['train'])))
                                if len(partial_predictors) < MAX_PARTIAL_PREDICTORS:
                                    # If the list of predictors is small we just add any new predictor
                                    partial_predictors.append((new_predictor,
                                                               full_output))
                                    partial_predictors.sort(key=lambda x: (x[0].net_distance_, x[0]))
                                elif (new_predictor.net_distance_, new_predictor) < \
                                        (partial_predictors[-1][0].net_distance_, partial_predictors[-1][0]):
                                    # If the list of predictors is greater than MAX_PARTIAL_PREDICTORS we check if it's
                                    # better then the worst one in the list
                                    partial_predictors.pop()
                                    partial_predictors.append((new_predictor,
                                                               full_output))
                                    partial_predictors.sort(key=lambda x: (x[0].net_distance_, x[0]))
                if len(good_predictors) >= 3:
                    break
            for predictor, predictions in partial_predictors:
                print(predictor)
            my_count += 1
            print(f'len(partial_predictors) = {len(partial_predictors)}')
            print(f'len(good_predictors) = {len(good_predictors)}')
            if len(partial_predictors) == 0:
                break
            depth += 1
    if not good_predictors:
        print("No good predictors")
        good_predictors = [predictor for predictor, _ in partial_predictors]
    return good_predictors
