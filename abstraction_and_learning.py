import os
import json
import time
import logging
import csv

from atomic_objects import find_components, find_color_entities

from classes import EntityFinder
from constants import CONSTANT_STRINGS, ALL_DIRECTIONS, MAX_NLL
from core_functions import test_case

from my_utils import tuplefy_task, WindowsInhibitor, multi_array_flattener

logging.basicConfig(filename='property_select_transform.log',
                    level=logging.INFO,
                    format='%(asctime)s:%(name)s:%(levelname)s:%(message)s')

expense_logger = logging.getLogger('expense')
file_handler = logging.FileHandler('expense.log')
expense_logger.addHandler(file_handler)
expense_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)

expense_logger.info("New test: " + ", ".join(CONSTANT_STRINGS))


def test_main():
    file_prefix = '../input/abstraction-and-reasoning-challenge/test/'
    fieldnames = ['output_id', 'output']
    output_file_name = 'submission.csv'
    with open(output_file_name, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for i, filename in enumerate(os.listdir(file_prefix)):
        with open(file_prefix + filename) as f:
            raw_task = json.load(f)
        task = tuplefy_task(raw_task, 'test' in file_prefix)
        component_entity_finder = EntityFinder(
            lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
        component_entities = component_entity_finder(task['train'][0]['input'])
        if len(component_entities) <= 30:
            base_entity_finder = component_entity_finder
        else:
            base_entity_finder = EntityFinder(lambda grid: find_color_entities(grid))
        predictors = test_case(task,
                               max_nll=MAX_NLL,
                               base_entity_finder=base_entity_finder,
                               allow_multiple_predictors=True,
                               allow_selector_pairs=True)
        for case in task['test']:
            print(f"case {case}")
            test_input = case['input']
            predictions = [predictor.predict(test_input) for predictor in predictors]
            prediction_set = set()
            for prediction in predictions:
                if prediction not in prediction_set:
                    prediction_set.add(prediction)

            if not prediction_set:
                prediction_set = [((0,),)]
            root, _ = os.path.splitext(filename)
            output_id = f"{os.path.basename(root)}_{i}"
            with open(output_file_name, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow({'output_id': output_id, 'output': multi_array_flattener(prediction_set)})
            print(f"Time elapsed = {time.perf_counter()}")


# Problematics: 64
def main():
    with WindowsInhibitor():
        move_list = {7, 24, 29, 34, 52, 72, 77, 92, 127}
        crop_list = {13, 30, 35, 48, 64, 90, 110}
        recolor_list = {9, 15, 39, 54}
        fill_list = {1, 16, 40, 60, 73, 80}
        shape_stamp_list = {11, 14, 18, 67, 74, 75, 88, 94, 100}
        draw_list = {12, 23, 33, 36, 42, 44, 46, 50, 59, 63, 83}
        good_list = {30, 35, 48, 64, 7, 29, 52, 72, 127, 9, 15, 94, 28, 51, 81, 78, 80, 11, 27, 149, 154}
        other_goods = {14, 34, 82, 86, 97, 105, 112, 115, 138, 139, 141, 144, 149, 151, 154, 163, 165, 168, 171, 178,
                       186, 193,
                       195, 197, 206, 209, 240}

        slow_goods = {111}
        sequential_list = {56, 120}
        select_square = {80, 87}
        line_reflect = {5, 25, 71, 143, 235}
        symmetry_cut = {66, 187}
        cleaning = {73, 70, 60, 16}

        poss_error_list = {96}
        solvable_list = {90, 87, 74, 69, 68, 67, 56}  # 61, 39, #18
        """
        Todo: Better distance function (56, )
        """
        corrects = []
        incorrects = []
        cases_tested = 0
        file_prefix = 'training/'
        logging.info(f'----------------------Starting new test file_prefix={file_prefix}--------------------------')
        some_tests = {205, 215, 217, 219, 221, 222, 227, 228, 229}
        nlls = []
        times = []
        for i, filename in enumerate(os.listdir(file_prefix)):
            if i not in other_goods:
                continue
            # if i != 163:
            #     continue
            start_time = time.perf_counter()
            print(f'Case {i}')
            logging.info(f'Case {i}')
            with open(file_prefix + filename) as f:
                raw_task = json.load(f)
            task = tuplefy_task(raw_task)
            test_input = task['test'][0]['input']
            test_output = task['test'][0]['output']
            # display_case(test_input)
            # display_case(test_output)
            # base_entity_finder = EntityFinder(find_components)
            component_entity_finder = EntityFinder(
                lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
            # # base_entity_finder = EntityFinder(lambda grid: find_components(grid, directions=ALL_DIRECTIONS))
            component_entities = component_entity_finder(task['train'][0]['input'])
            # print(f'len(component_entity_finder.cache) = {len(component_entity_finder.cache)}')
            # for entities in component_entity_finder.cache.values():
            #     for entity in entities:
            #         entity.display()
            # print(f'len(component_entities) = {len(component_entities)}')
            if len(component_entities) <= 30:
                base_entity_finder = component_entity_finder
            else:
                base_entity_finder = EntityFinder(lambda grid: find_color_entities(grid))
                print('Using color entity finder')
            # try:
            start_time = time.perf_counter()
            # First attempt a lower nll
            print(f'First attempt, NLL = {MAX_NLL - 2}')
            predictors = test_case(task,
                                   max_nll=MAX_NLL - 2,
                                   base_entity_finder=base_entity_finder,
                                   allow_multiple_predictors=False,
                                   allow_selector_pairs=True)
            time_elapsed = time.perf_counter() - start_time
            print(f"time_elapsed = {time_elapsed}")
            # Complexity is roughly 3^n or 4^n, so if the first try fails we up the nll without going over 5 min
            if (not predictors and time_elapsed < 60.) or (len(predictors) < 3 and time_elapsed < 3.):
                if len(predictors) > 0:
                    print(f"Found {len(predictors)} predictors, now looking for more")
                print(f'Second attempt, NLL = {MAX_NLL - (1 if time_elapsed > 18.75 else 0)}')
                predictors = test_case(task,
                                       max_nll=MAX_NLL - (1 if time_elapsed > 18.75 else 0),
                                       base_entity_finder=base_entity_finder,
                                       allow_multiple_predictors=True,
                                       allow_selector_pairs=True)

            # except IndexError:
            #     print("Index Error!")
            #     predictors = []

            test_input = task['test'][0]['input']

            test_output = task['test'][0]['output']
            for predictor in predictors:
                print(predictor, predictor.nll)
            predictions = [predictor.predict(test_input) for predictor in predictors]
            prediction_set = set()
            for prediction in predictions:
                if prediction not in prediction_set:
                    prediction_set.add(prediction)
                    if len(prediction_set) == 3:
                        break
            correct_predictors = [(str(predictor), predictor.entity_finders[0].nll + predictor.transformers[0].nll) for
                                  prediction, predictor in
                                  zip(predictions, predictors) if prediction == test_output]
            print(correct_predictors)

            nlls.append(
                (i, min(predictor[1] for predictor in correct_predictors) if correct_predictors else float('inf')))

            if True in [prediction == test_output for prediction in prediction_set]:
                print('Correct!')
                logging.info('Correct!')
                corrects.append(i)
                print(f'corrects = {corrects}')
                logging.info(f'corrects = {corrects}')
            else:
                print('Incorrect!')
                logging.info('Incorrect!')
                incorrects.append(i)
                print(f'incorrects = {incorrects}')
                logging.info(f'incorrects = {incorrects}')
            print(nlls)
            print(f'Time elapsed for task {i} is {time.perf_counter() - start_time}')
            logging.info(f'Time elapsed for task {i} is {time.perf_counter() - start_time}')
            times.append((i, time.perf_counter() - start_time))
            cases_tested += 1
            print(f'Accuracy = {len(corrects) * 100 / cases_tested}%')
            logging.info(f'Accuracy = {len(corrects) * 100 / cases_tested}%')
        expense_logger.info(f'nlls = {nlls}')
        print(corrects)
        expense_logger.info(f'times = {times}')
        expense_logger.info(f'total_time = {time.perf_counter()}')
        print(time.perf_counter())


# NLLs = 4.7, 29.3,

if __name__ == '__main__':
    main()
