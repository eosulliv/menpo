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


@labeller_func(group_label='ear_ibug_50')
def ears_ibug_55_to_ears_ibug_50(pcloud):
    r"""
    Apply the IBUG 50-point semantic labels for a single ears.

    The semantic labels applied are as follows:
      - outer_helix
      - inner_helix
      - tragus_concha_inf_crus

    References
    ----------
    .. [1] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 55
    validate_input(pcloud, n_expected_points)

    outer_helix_indices = np.arange(0, 20)
    inner_helix_indices = np.arange(20, 35)
    tragus_concha_inf_crus_indices = np.arange(35, 50)

    outer_helix_connectivity = connectivity_from_array(outer_helix_indices)
    inner_helix_connectivity = connectivity_from_array(inner_helix_indices)
    tragus_concha_inf_crus_connectivity = connectivity_from_array(tragus_concha_inf_crus_indices)

    all_connectivity = np.vstack([outer_helix_connectivity, inner_helix_connectivity,
                                  tragus_concha_inf_crus_connectivity])

    mapping = OrderedDict()
    mapping['outer_helix'] = outer_helix_indices
    mapping['inner_helix'] = inner_helix_indices
    mapping['tragus_concha_inf_crus'] = tragus_concha_inf_crus_indices

    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points, all_connectivity, mapping)

    return new_pcloud, mapping

@labeller_func(group_label='ear_ibug_14')
def ears_ibug_55_to_ears_ibug_14(pcloud):
    r"""
    Apply the IBUG 14-point semantic labels for a single ears.

    The semantic labels applied are as follows:
      - outer_helix
      - inner_helix
      - tragus_concha_inf_crus
      - sup_crus

    References
    ----------
    .. [1] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 55
    validate_input(pcloud, n_expected_points)

    outer_helix_indices = np.arange(0, 6)
    inner_helix_indices = np.arange(6, 10)
    tragus_concha_inf_crus_indices = np.arange(10, 14)

    outer_helix_connectivity = connectivity_from_array(outer_helix_indices)
    inner_helix_connectivity = connectivity_from_array(inner_helix_indices)
    tragus_concha_inf_crus_connectivity = connectivity_from_array(tragus_concha_inf_crus_indices)

    all_connectivity = np.vstack([outer_helix_connectivity, inner_helix_connectivity,
                                  tragus_concha_inf_crus_connectivity])

    mapping = OrderedDict()
    mapping['outer_helix'] = outer_helix_indices
    mapping['inner_helix'] = inner_helix_indices
    mapping['tragus_concha_inf_crus'] = tragus_concha_inf_crus_indices

    ind = np.hstack(([0, 3, 7, 11, 16, 19], [20, 25, 31, 34], [35, 39, 44, 49]))
    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points[ind], all_connectivity, mapping)

    return new_pcloud, mapping


@labeller_func(group_label='ear_ibug_50')
def ears_ibug_50_to_ears_ibug_50(pcloud):
    r"""
    Apply the IBUG 55-point semantic labels for a single ears.

    The semantic labels applied are as follows:
      - outer_helix
      - inner_helix
      - tragus_concha_inf_crus

    References
    ----------
    .. [1] http://www.multipie.org/
    .. [2] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 50
    validate_input(pcloud, n_expected_points)

    outer_helix_indices = np.arange(0, 20)
    inner_helix_indices = np.arange(20, 35)
    tragus_concha_inf_crus_indices = np.arange(35, 50)

    outer_helix_connectivity = connectivity_from_array(outer_helix_indices)
    inner_helix_connectivity = connectivity_from_array(inner_helix_indices)
    tragus_concha_inf_crus_connectivity = connectivity_from_array(tragus_concha_inf_crus_indices)

    all_connectivity = np.vstack([outer_helix_connectivity, inner_helix_connectivity,
                                  tragus_concha_inf_crus_connectivity])

    mapping = OrderedDict()
    mapping['outer_helix'] = outer_helix_indices
    mapping['inner_helix'] = inner_helix_indices
    mapping['tragus_concha_inf_crus'] = tragus_concha_inf_crus_indices

    ind = np.arange(0, 50)
    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points[ind], all_connectivity, mapping)

    return new_pcloud, mapping


@labeller_func(group_label='ear_ibug_14')
def ears_ibug_50_to_ears_ibug_14(pcloud):
    r"""
    Apply the IBUG 14-point semantic labels for a single ears.

    The semantic labels applied are as follows:
      - outer_helix
      - inner_helix
      - tragus_concha_inf_crus

    References
    ----------
    .. [1] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 50
    validate_input(pcloud, n_expected_points)

    outer_helix_indices = np.arange(0, 6)
    inner_helix_indices = np.arange(6, 10)
    tragus_concha_inf_crus_indices = np.arange(10, 14)

    outer_helix_connectivity = connectivity_from_array(outer_helix_indices)
    inner_helix_connectivity = connectivity_from_array(inner_helix_indices)
    tragus_concha_inf_crus_connectivity = connectivity_from_array(tragus_concha_inf_crus_indices)

    all_connectivity = np.vstack([outer_helix_connectivity, inner_helix_connectivity,
                                  tragus_concha_inf_crus_connectivity])

    mapping = OrderedDict()
    mapping['outer_helix'] = outer_helix_indices
    mapping['inner_helix'] = inner_helix_indices
    mapping['tragus_concha_inf_crus'] = tragus_concha_inf_crus_indices

    ind = np.hstack(([0, 3, 7, 11, 16, 19], [20, 25, 31, 34], [35, 39, 44, 49]))
    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points[ind], all_connectivity, mapping)

    return new_pcloud, mapping


@labeller_func(group_label='ear_ibug_14')
def ears_ibug_14_to_ears_ibug_14(pcloud):
    r"""
    Apply the IBUG 14-point semantic labels for a single ears.

    The semantic labels applied are as follows:
      - outer_helix
      - inner_helix
      - tragus_concha_inf_crus

    References
    ----------
    .. [2] http://ibug.doc.ic.ac.uk/resources/ibug-ears/
    """
    from menpo.shape import LabelledPointUndirectedGraph

    n_expected_points = 14
    validate_input(pcloud, n_expected_points)

    outer_helix_indices = np.arange(0, 6)
    inner_helix_indices = np.arange(6, 10)
    tragus_concha_inf_crus_indices = np.arange(10, 14)

    outer_helix_connectivity = connectivity_from_array(outer_helix_indices)
    inner_helix_connectivity = connectivity_from_array(inner_helix_indices)
    tragus_concha_inf_crus_connectivity = connectivity_from_array(tragus_concha_inf_crus_indices)

    all_connectivity = np.vstack([outer_helix_connectivity, inner_helix_connectivity,
                                  tragus_concha_inf_crus_connectivity])

    mapping = OrderedDict()
    mapping['outer_helix'] = outer_helix_indices
    mapping['inner_helix'] = inner_helix_indices
    mapping['tragus_concha_inf_crus'] = tragus_concha_inf_crus_indices

    new_pcloud = LabelledPointUndirectedGraph.init_from_indices_mapping(
        pcloud.points, all_connectivity, mapping)

    return new_pcloud, mapping


@labeller_func(group_label='ear_ibug_55_trimesh')
def ears_ibug_55_to_face_ibug_55_trimesh(pcloud):
    r"""
    Apply the IBUG 55-point semantic labels, with trimesh connectivity.
      - tri
    References
    ----------
    .. [1] http://www.multipie.org/
    .. [2] http://ibug.doc.ic.ac.uk/resources/300-W/
    """
    from menpo.shape import TriMesh

    n_expected_points = 55
    validate_input(pcloud, n_expected_points)

    tri_list = np.array([[38, 19,  0], [45, 52, 51], [20, 35,  0],
                         [35, 20, 47], [36, 38,  0], [35, 36,  0],
                         [ 4, 24,  3], [45, 43, 42], [43, 33, 42],
                         [44, 45, 51], [50, 44, 51], [44, 43, 45],
                         [43, 44, 50], [31, 12, 13], [31, 50, 51],
                         [52, 46, 47], [46, 52, 45], [46, 35, 47],
                         [46, 36, 35], [46, 45, 42], [36, 46, 42],
                         [33, 15, 16], [31, 32, 50], [15, 32, 14],
                         [32, 15, 33], [32, 43, 50], [43, 32, 33],
                         [32, 13, 14], [33, 34, 42], [24, 23,  3],
                         [32, 31, 13], [41, 36, 42], [34, 33, 16],
                         [34, 41, 42], [34, 18, 19], [23,  2,  3],
                         [53, 52, 47], [25,  4,  5], [25, 24,  4],
                         [ 7,  8, 27], [41, 37, 36], [41, 40, 38],
                         [ 8,  9, 27], [36, 37, 38], [37, 41, 38],
                         [17, 34, 16], [34, 17, 18], [15, 17, 16],
                         [34, 40, 41], [52, 30, 51], [30, 11, 12],
                         [30, 31, 51], [ 1, 22,  0], [54, 48, 49],
                         [31, 30, 12], [ 2, 22,  1], [23, 22,  2],
                         [20, 48, 47], [ 9, 28, 27], [28, 53, 27],
                         [28,  9, 10], [26,  7, 27], [26,  6,  7],
                         [ 6, 26,  5], [38, 39, 19], [54, 49, 24],
                         [26, 25,  5], [39, 34, 19], [39, 40, 34],
                         [40, 39, 38], [49, 22, 23], [49, 48, 20],
                         [29, 28, 10], [11, 29, 10], [53, 29, 52],
                         [28, 29, 53], [54, 26, 27], [25, 54, 24],
                         [29, 30, 52], [30, 29, 11], [53, 54, 27],
                         [26, 54, 25], [54, 53, 47], [48, 54, 47],
                         [21, 20,  0], [21, 49, 20], [22, 21,  0],
                         [49, 23, 24], [49, 21, 22]])
    new_pcloud = TriMesh(pcloud.points, trilist=tri_list)

    mapping = OrderedDict()
    mapping['tri'] = np.arange(new_pcloud.n_points)

    return new_pcloud, mapping
