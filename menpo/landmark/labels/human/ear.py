from collections import OrderedDict
import numpy as np

from ..base import (validate_input, connectivity_from_array, labeller_func)


@labeller_func(group_label='ear_ibug_110')
def ears_ibug_110_to_ears_ibug_110(pcloud):
    r"""
    Apply the IBUG 55-point semantic labels for left and right ears.

    The semantic labels applied are as follows:
      - left_outer_helix
      - left_inner_helix
      - left_tragus_concha_inf_crus
      - left_sup_crus
      - right_outer_helix
      - right_inner_helix
      - right_tragus_concha_inf_crus
      - right_sup_crus

    References
    ----------
    .. [1] http://www.multipie.org/
    .. [2] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 110
    validate_input(pcloud, n_expected_points)

    l_outer_helix_indices = np.arange(0, 20)
    l_inner_helix_indices = np.arange(20, 35)
    l_tragus_concha_inf_crus_indices = np.arange(35, 50)
    l_sup_crus_indices = np.arange(50, 55)
    r_outer_helix_indices = np.arange(55, 75)
    r_inner_helix_indices = np.arange(75, 90)
    r_tragus_concha_inf_crus_indices = np.arange(90, 105)
    r_sup_crus_indices = np.arange(105, 110)
    
    l_outer_helix_connectivity = connectivity_from_array(l_outer_helix_indices)
    l_inner_helix_connectivity = connectivity_from_array(l_inner_helix_indices)
    l_tragus_concha_inf_crus_connectivity = connectivity_from_array(l_tragus_concha_inf_crus_indices)
    l_sup_crus_connectivity = connectivity_from_array(l_sup_crus_indices)
    r_outer_helix_connectivity = connectivity_from_array(r_outer_helix_indices)
    r_inner_helix_connectivity = connectivity_from_array(r_inner_helix_indices)
    r_tragus_concha_inf_crus_connectivity = connectivity_from_array(r_tragus_concha_inf_crus_indices)
    r_sup_crus_connectivity = connectivity_from_array(r_sup_crus_indices)

    all_connectivity = np.vstack([
        l_outer_helix_connectivity, l_inner_helix_connectivity, l_tragus_concha_inf_crus_connectivity,
        l_sup_crus_connectivity, r_outer_helix_connectivity, r_inner_helix_connectivity,
        r_tragus_concha_inf_crus_connectivity, r_sup_crus_connectivity])

    mapping = OrderedDict()
    mapping['left_outer_helix'] = l_outer_helix_indices
    mapping['left_inner_helix'] = l_inner_helix_indices
    mapping['left_tragus_concha_inf_crus'] = l_tragus_concha_inf_crus_indices
    mapping['left_sup_crus'] = l_sup_crus_indices
    mapping['right_outer_helix'] = r_outer_helix_indices
    mapping['right_inner_helix'] = r_inner_helix_indices
    mapping['right_tragus_concha_inf_crus'] = r_tragus_concha_inf_crus_indices
    mapping['right_sup_crus'] = r_sup_crus_indices

    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points, all_connectivity, mapping)

    return new_pcloud, mapping


@labeller_func(group_label='ear_ibug_55')
def ears_ibug_110_to_ears_ibug_55(pcloud):
    r"""
    Apply the IBUG 55-point semantic labels for the left and right ears.

    The semantic labels are applied as follows:
      - outer_helix
      - inner_helix
      - tragus_concha_inf_crus
      - sup_crus

    References
    ----------
    .. [1] http://www.multipie.org/
    .. [2] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 110
    validate_input(pcloud, n_expected_points)

    outer_helix_indices = np.arange(0, 20)
    inner_helix_indices = np.arange(20, 35)
    tragus_concha_inf_crus_indices = np.arange(35, 50)
    sup_crus_indices = np.arange(50, 55)

    # Left and right ears have the same connectivity
    outer_helix_connectivity = connectivity_from_array(outer_helix_indices)
    inner_helix_connectivity = connectivity_from_array(inner_helix_indices)
    tragus_concha_inf_crus_connectivity = connectivity_from_array(tragus_concha_inf_crus_indices)
    sup_crus_connectivity = connectivity_from_array(sup_crus_indices)

    all_connectivity = np.vstack([outer_helix_connectivity, inner_helix_connectivity,
        tragus_concha_inf_crus_connectivity, sup_crus_connectivity])

    # Left and right ears have the same mapping
    mapping = OrderedDict()
    mapping['outer_helix'] = outer_helix_indices
    mapping['inner_helix'] = inner_helix_indices
    mapping['tragus_concha_inf_crus'] = tragus_concha_inf_crus_indices
    mapping['sup_crus'] = sup_crus_indices

    l_new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points[0:55], all_connectivity, mapping)
    r_new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points[55:110], all_connectivity, mapping)

    return [l_new_pcloud, r_new_pcloud], mapping


@labeller_func(group_label='ear_ibug_55')
def ears_ibug_55_to_ears_ibug_55(pcloud):
    r"""
    Apply the IBUG 55-point semantic labels for a single ears.
    
    The semantic labels applied are as follows:
      - outer_helix
      - inner_helix
      - tragus_concha_inf_crus
      - sup_crus
    
    References
    ----------
    .. [1] http://www.multipie.org/
    .. [2] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 55
    validate_input(pcloud, n_expected_points)

    outer_helix_indices = np.arange(0, 20)
    inner_helix_indices = np.arange(20, 35)
    tragus_concha_inf_crus_indices = np.arange(35, 50)
    sup_crus_indices = np.arange(50, 55)
    
    outer_helix_connectivity = connectivity_from_array(outer_helix_indices)
    inner_helix_connectivity = connectivity_from_array(inner_helix_indices)
    tragus_concha_inf_crus_connectivity = connectivity_from_array(tragus_concha_inf_crus_indices)
    sup_crus_connectivity = connectivity_from_array(sup_crus_indices)

    all_connectivity = np.vstack([outer_helix_connectivity, inner_helix_connectivity,
                                  tragus_concha_inf_crus_connectivity, sup_crus_connectivity])

    mapping = OrderedDict()
    mapping['outer_helix'] = outer_helix_indices
    mapping['inner_helix'] = inner_helix_indices
    mapping['tragus_concha_inf_crus'] = tragus_concha_inf_crus_indices
    mapping['sup_crus'] = sup_crus_indices

    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points, all_connectivity, mapping)

    return new_pcloud, mapping