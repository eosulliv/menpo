from collections import OrderedDict
import numpy as np

from ..base import (
    validate_input,
    connectivity_from_array,
    labeller_func,
)


@labeller_func(group_label='skull_gosh_50')
def skull_gosh_50_to_skull_gosh_50(pcloud):
    r"""
    Apply the GOSH 50-point semantic labels.

    The semantic labels are as follows:

      - midline
      - maxilla
      - left_eye
      - right_eye
      - mandible
      - midline_skull
      - inner_mandible
      - skull_base

    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 50
    validate_input(pcloud, n_expected_points)

    midline_indices = np.arange(0, 11)
    max_indices = np.arange(11, 19)
    leye_indices = np.arange(19, 22)
    reye_indices = np.arange(22, 25)
    mand_indices = np.arange(25, 35)
    mid_skull_indices = np.arange(35, 38)
    inner_mand_indices = np.arange(38, 42)
    lskull_base_indices = np.arange(42, 46)
    rskull_base_indices = np.arange(46, 50)

    midline_connectivity = connectivity_from_array(midline_indices)
    max_connectivity = connectivity_from_array(max_indices)
    leye_connectivity = connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = connectivity_from_array(reye_indices, close_loop=True)
    mand_connectivity = connectivity_from_array(mand_indices)
    mid_skull_connectivity = connectivity_from_array(mid_skull_indices)
    inner_mand_connectivity = connectivity_from_array(inner_mand_indices)
    skull_base_connectivity = np.vstack([
        connectivity_from_array(lskull_base_indices),
        connectivity_from_array(rskull_base_indices)
    ])
    

    all_connectivity = np.vstack([
        midline_connectivity, max_connectivity, leye_connectivity,
        reye_connectivity, mand_connectivity, mid_skull_connectivity,
        inner_mand_connectivity, skull_base_connectivity
    ])

    mapping = OrderedDict()
    mapping['midline'] = midline_indices
    mapping['maxilla'] = max_indices
    mapping['left_eye'] = leye_indices
    mapping['right_eye'] = reye_indices
    mapping['mandible'] = mand_indices
    mapping['midline_skull'] = mid_skull_indices
    mapping['inner_mandible'] = inner_mand_indices
    mapping['skull_base'] = np.hstack((lskull_base_indices, rskull_base_indices))

    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points, all_connectivity, mapping)

    return new_pcloud, mapping


@labeller_func(group_label='skull_gosh_44')
def skull_gosh_50_to_skull_gosh_44(pcloud):
    r"""
    Apply the GOSH 50-point semantic labels, but ignore the mental foramen
    and anterior skull landmarks.

    The semantic labels are as follows:

      - midline
      - maxilla
      - left_eye
      - right_eye
      - mandible
      - midline_skull
      - inner_mandible
      - skull_base

    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 50
    validate_input(pcloud, n_expected_points)

    midline_indices = np.hstack((0, 33, np.arange(1, 11)))
    max_indices = np.arange(11, 19)
    leye_indices = np.arange(19, 22)
    reye_indices = np.arange(22, 25)
    rmand_indices = np.arange(25, 29)
    lmand_indices = np.arange(29, 33)
    inner_mand_indices = np.arange(34, 36)
    rskull_base_indices = np.arange(36, 40)
    lskull_base_indices = np.arange(40, 44)

    midline_connectivity = connectivity_from_array(midline_indices)
    max_connectivity = connectivity_from_array(max_indices)
    leye_connectivity = connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = connectivity_from_array(reye_indices, close_loop=True)
    mand_connectivity = np.vstack([
        connectivity_from_array(rmand_indices),
        connectivity_from_array(lmand_indices)
    ])
    inner_mand_connectivity = connectivity_from_array(inner_mand_indices)
    skull_base_connectivity = np.vstack([
        connectivity_from_array(rskull_base_indices),
        connectivity_from_array(lskull_base_indices)
    ])

    all_connectivity = np.vstack([
        midline_connectivity, max_connectivity, leye_connectivity, reye_connectivity,
        mand_connectivity, inner_mand_connectivity, skull_base_connectivity
    ])

    mapping = OrderedDict()
    mapping['midline'] = midline_indices
    mapping['maxilla'] = max_indices
    mapping['left_eye'] = leye_indices
    mapping['right_eye'] = reye_indices
    mapping['mandible'] = np.hstack((rmand_indices, lmand_indices))
    mapping['inner_mandible'] = inner_mand_indices
    mapping['skull_base'] = np.hstack((rskull_base_indices, lskull_base_indices))

    ind = np.hstack((np.arange(0, 29), np.arange(31, 36), np.arange(38, 39), np.arange(41, 50)))
    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points[ind], all_connectivity, mapping)

    return new_pcloud, mapping


@labeller_func(group_label='skull_gosh_44')
def skull_gosh_44_to_skull_gosh_44(pcloud):
    r"""
    Apply the GOSH 50-point semantic labels, but ignore the mental foramen
    and anterior skull landmarks.

    The semantic labels are as follows:

      - midline
      - maxilla
      - left_eye
      - right_eye
      - mandible
      - midline_skull
      - inner_mandible
      - skull_base

    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 44
    validate_input(pcloud, n_expected_points)

    midline_indices = np.hstack((0, 33, np.arange(1, 11)))
    max_indices = np.arange(11, 19)
    leye_indices = np.arange(19, 22)
    reye_indices = np.arange(22, 25)
    lmand_indices = np.arange(25, 29)
    rmand_indices = np.arange(29, 33)
    inner_mand_indices = np.arange(34, 36)
    lskull_base_indices = np.arange(36, 40)
    rskull_base_indices = np.arange(40, 44)

    midline_connectivity = connectivity_from_array(midline_indices)
    max_connectivity = connectivity_from_array(max_indices)
    leye_connectivity = connectivity_from_array(leye_indices, close_loop=True)
    reye_connectivity = connectivity_from_array(reye_indices, close_loop=True)
    mand_connectivity = np.vstack([
        connectivity_from_array(lmand_indices),
        connectivity_from_array(rmand_indices)
    ])
    inner_mand_connectivity = connectivity_from_array(inner_mand_indices)
    skull_base_connectivity = np.vstack([
        connectivity_from_array(lskull_base_indices),
        connectivity_from_array(rskull_base_indices)
    ])

    all_connectivity = np.vstack([
        midline_connectivity, max_connectivity, leye_connectivity, reye_connectivity,
        mand_connectivity, inner_mand_connectivity, skull_base_connectivity
    ])

    mapping = OrderedDict()
    mapping['midline'] = midline_indices
    mapping['maxilla'] = max_indices
    mapping['left_eye'] = leye_indices
    mapping['right_eye'] = reye_indices
    mapping['mandible'] = np.hstack((lmand_indices, rmand_indices))
    mapping['inner_mandible'] = inner_mand_indices
    mapping['skull_base'] = np.hstack((lskull_base_indices, rskull_base_indices))

    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points, all_connectivity, mapping)

    return new_pcloud, mapping
