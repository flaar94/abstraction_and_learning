from typing import Iterable
from constants import SELECTOR_TARGET_REQUIRES_ENTITY_PENALTY, MISMATCHED_OUTPUT_PENALTY, SELECTING_SAME_PROPERTY_BONUS, \
    SELECTOR_PAIR_PENALTY, PROPERTY_CONSTRUCTION_COST, SELECTOR_CONSTRUCTION_COST, NONCONSTANT_BOOL_PENALTY


def union_nlls(nll1, nll2):
    return min(nll1, nll2)


def combine_relation_selector_nll(relation_or_property, selector, ordinal_property):
    new_nll = relation_or_property.nll + selector.nll + ordinal_property.nll + PROPERTY_CONSTRUCTION_COST
    if not (relation_or_property.output_types & ordinal_property.input_types):
        new_nll += MISMATCHED_OUTPUT_PENALTY
    # if_neg(new_nll)
    return new_nll


combine_property_selector_nll = combine_relation_selector_nll


def combine_selector_nll(entity_property, target_property):
    new_nll = entity_property.nll + target_property.nll + SELECTOR_CONSTRUCTION_COST
    if not (entity_property.output_types & target_property.output_types):
        new_nll += MISMATCHED_OUTPUT_PENALTY
    if entity_property.output_types == frozenset({'bool'}) and not target_property.is_constant:
        new_nll += NONCONSTANT_BOOL_PENALTY
    if target_property.requires_entity:
        new_nll += SELECTOR_TARGET_REQUIRES_ENTITY_PENALTY
    if entity_property in target_property.associated_objects:
        new_nll -= SELECTING_SAME_PROPERTY_BONUS
        new_nll = max(new_nll, 0)
    return new_nll


def combine_pair_selector_nll(selector1, selector2):
    return selector1.nll + selector2.nll + SELECTOR_PAIR_PENALTY


def combine_move_nll(property0, property1):
    combined_nll = property0.nll + property1.nll
    if 'y_length' not in property0.output_types:
        combined_nll += MISMATCHED_OUTPUT_PENALTY
    if 'x_length' not in property1.output_types:
        combined_nll += MISMATCHED_OUTPUT_PENALTY
    return combined_nll


def combine_color_nll(property_pairs: Iterable):
    nll = 0
    for property1, property2 in property_pairs:
        nll += property1.nll + property2.nll
    return nll


