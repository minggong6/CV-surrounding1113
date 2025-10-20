import logging
import sys

sys.setrecursionlimit(100000)  # 例如这里设置为十万

import kimimaro
import numpy as np
import toolkit_3D as tk3
from math import sqrt, log
from scipy import ndimage
from skimage import morphology


class Point:

    def __init__(self, coordinate, radius, path_id=-1):
        self.coordinate = (int(coordinate[0]), int(coordinate[1]), int(coordinate[2]))
        self.radius = radius
        self.path_id = path_id
        self.is_end = False
        self.is_cross = False
        self.cross_block = None

    def __str__(self):
        if self.is_end:
            return '{' + 'END    ' + str(self.path_id) + ', ' + str(self.coordinate) + ', ' + str(self.radius) + '}'
        elif self.is_cross:
            return '{' + 'CROSS  ' + str(self.coordinate) + ', ' + str(self.radius) + ', in cross block ' + str(
                self.cross_block) + '}'
        else:
            return '{' + 'NORMAL ' + str(self.path_id) + ', ' + str(self.coordinate) + ', ' + str(self.radius) + '}'


class ThinPoint:
    def __init__(self, point):
        self.coordinate = point.coordinate
        self.radius = point.radius
        self.path_id = point.path_id
        self.is_end = point.is_end
        self.is_cross = point.is_cross
        self.cross_block = point.cross_block
        self.radis_dist = 0.0
        self.vessel_tumor_dist = 0.0

    def __str__(self):
        return str(self.coordinate) + '&' + str(self.radius) + '&' + str(self.radis_dist) + '&' + str(
            self.vessel_tumor_dist) + ';'


def get_point_by_coordinate(point_list, coordinate):
    for point in point_list:
        if point.coordinate == coordinate:
            return point
    print('error in [get_point_by_coordinate]: Not found point ' + str(coordinate))


class CrossBlock:
    def __init__(self):
        # Save the cross point coordinates in the block
        # e.g. [(1, 1, 1), (2, 2, 2), ...]
        self.cross_voxel_list = []

        # Save the points (and their path) connected to the CrossBlock
        # e.g. [((3, 4, 5), 73), ((6, 7, 8), 79), ...]
        self.connection_list = []

        self.cross_point_list = []

        self.degree = 0
        self.size = 0

    def __str__(self):
        cbstr = 'Cross Block: size = ' + str(self.size) + '\tdegree = ' + str(self.degree) + '\n\t' \
                + 'cross voxels: ' + str(self.cross_voxel_list) + '\n\t' \
                + 'connections: ' + str(self.connection_list)
        return cbstr


class Skeleton:
    def __init__(self, vessel_label, tumor_label=None,
                 max_min_radis_dist=1,
                 max_vessel_tumor_dist=3,
                 thin_degree=0.33,
                 related_range_bias=2,
                 avg_radius=False,
                 part_img=None,
                 target=None
                 ):

        self.historian = {
            "generate_skele_point_list": False,
            "generate_radius_graph": False,
            "generate_path_graph": False,
            "generate_cross_block_list": False,
            "inject_path_id": False,
            "cut_branch": False,
            "generate_part_graph_normal": False,
            "generate_part_graph_simulator": False,
            "purify_path_graph": False,
            "generate_ordered_path": False,
            # "": False,
            # "": False,
            # "": False,
            # "": False,
            # "": False,
            # "": False,
            # "": False,
            # "": False,
        }

        # region <====== containers & overall settings ======>
        self.cross_marker = 114514

        self.vessel = vessel_label
        self.tumor = tumor_label
        self.shape = vessel_label.shape
        self.skele_point_list = []
        self.skele_point_list_cb = []
        self.radius_graph = np.zeros(self.shape)

        self.path_graph = np.zeros(self.shape)
        self.path_graph_cb = np.zeros(self.shape)

        self.part_graph = np.zeros(self.shape)

        self.cross_block_list = []
        self.end_voxel_list = []

        self.path_num = 0
        self.visited_graph = None

        self.ordered_path_point_list_dict = {}
        self.thin_point_list = []
        # endregion <====== container & overall settings ======>

        # region <====== PARAMETERS: thin analysis ======>
        self.part_img = part_img
        self.max_min_radis_dist = max_min_radis_dist
        self.max_vessel_tumor_dist = max_vessel_tumor_dist
        self.thin_degree = thin_degree
        self.related_range_bias = related_range_bias
        self.avg_radius = avg_radius
        self.save_mark = True
        self.thin_mark_graph = None
        self.target = target
        # endregion <====== PARAMETERS: thin analysis ======>

        # region <====== PARAMETERS: trunk part ======>
        self.trunk_angle_threshold = 50
        '''[self.trunk_angle_threshold] marks the max angle between adjacent two trunk paths'''
        self.trunk_path_graph = None
        '''[self.trunk_path_graph] merges all the trunk paths into one path, independent from [self.part_graph], 
           and must be generated depending on [self.part_graph]'''
        self.trunk_path_list = []
        '''[self.trunk_path_list] stores path_ids that belong to the trunk'''
        self.trunk_cross_list = []
        '''[self.trunk_cross_list] stores cross_blocks that belong to the trunk'''
        # endregion <====== PARAMETERS: trunk part ======>

        # region <====== PARAMETERS: vein part ======>
        self.y_path_graph = None
        self.y_cross_list = []
        self.y_path_list = []
        # endregion <====== PARAMETERS: vein part ======>

        # region <====== PARAMETERS: cut branch ======>
        self.branch_path_graph = None
        self.branch_cross_list = []
        self.branch_path_list = []
        # endregion <====== PARAMETERS: cut branch ======>

    def history_check(self, dependency_list):
        for dependency in dependency_list:
            assert dependency in self.historian.keys(), f"{dependency} do not exist in history"
            assert self.historian[dependency], f"{dependency} has not done yet"

    def history_update(self, dependency):
        assert dependency in self.historian.keys(), f"{dependency} should not exist in history"
        if self.historian[dependency]:
            tk3.print_color(f'Warning: {dependency} has been executed more than once', 'yellow')
        else:
            self.historian[dependency] = True
            tk3.print_color(f'Info: executing {dependency}...', 'yellow')

    def generate_skele_point_list(self):

        self.history_update('generate_skele_point_list')

        label = self.vessel.copy()
        label = ((label > 0) / 1.0).astype(np.uint8)

        origin_skeleton = kimimaro.skeletonize(label,
                                               teasar_params={
                                                   'scale': 1,
                                                   'const': 3,  # physical units
                                                   'pdrf_exponent': 4,
                                                   'pdrf_scale': 100000,
                                                   'soma_detection_threshold': 1100,  # physical units
                                                   'soma_acceptance_threshold': 3500,  # physical units
                                                   'soma_invalidation_scale': 1.0,
                                                   'soma_invalidation_const': 300,  # physical units
                                                   'max_paths': 1000,  # default None
                                               }, dust_threshold=10,
                                               # skip connected components with fewer than this many voxels
                                               anisotropy=(1, 1, 1),  # default True
                                               fix_branching=True,  # default True
                                               fix_borders=True,  # default True
                                               fill_holes=True,  # default False
                                               parallel_chunk_size=100,
                                               # how many skeletons to process before updating progress bar
                                               )
        vertices = None
        radius = None
        for i in origin_skeleton.keys():
            sk = origin_skeleton[i]
            vertices = sk.vertices
            radius = sk.radius

        for (v, r) in zip(vertices, radius):
            x, y, z = v[0], v[1], v[2]
            new_point = Point((x, y, z), r)
            self.skele_point_list.append(new_point)

        # print('skeleton total length: ' + str(len(self.skele_point_list)))

    def generate_radius_graph(self):
        """
        Every voxel equals to the radius of the skeleton point
        :param label:
        :param skele_point_list:
        :return:
        """
        self.history_check(['generate_skele_point_list'])
        self.history_update('generate_radius_graph')

        for skp in self.skele_point_list:
            self.radius_graph[skp.coordinate] = skp.radius

    def get_adjacent_info(self, curr_coordinate, graph=None):
        """
        :param skele_graph: visited_graph and radius_graph is all available here, just guarantee the skeleton voxel
                            value > 0 and background voxel value < 0
        :param curr_coordinate: target voxel coordinate
        :param graph:
        :return: degree - degree of the target voxel
                 adjacent_voxel_list - the coordinates of skeleton voxels in all 26 (max) adjacent voxels
        """
        if graph is None:
            self.history_check(['generate_radius_graph'])
            skele_graph = self.radius_graph
        else:
            skele_graph = graph
        surround_voxel_list = tk3.get_surround_voxel(curr_coordinate, skele_graph.shape)
        adjacent_voxel_list = []
        degree = 0
        for sur_v in surround_voxel_list:
            if skele_graph[sur_v] > 0:
                adjacent_voxel_list.append((sur_v, skele_graph[sur_v]))
                degree += 1
        return degree, adjacent_voxel_list

    def generate_path_graph(self):

        self.history_check(['generate_radius_graph'])
        self.history_update('generate_path_graph')

        # Part 1: Use DFS traversal to mark all the paths
        self.path_num = 1
        # In visited_graph, 1 - skeleton unvisited, 2 - skeleton visited, 0 - background
        self.visited_graph__ = np.where(self.radius_graph > 0, 1, 0)

        def inject(point_list):
            for point_coord in point_list:
                self.path_graph[point_coord] = self.path_num

        def DFS_traversal(curr_coordinate, is_first=False):

            # Visit this voxel
            self.visited_graph__[curr_coordinate] = 2

            degree, adjacent_voxel_list = self.get_adjacent_info(curr_coordinate, self.visited_graph__)

            # When meet the voxel degree = 1 again, it's the end, cease the DFS,
            # return list including itself
            if degree == 1:
                self.end_voxel_list.append(curr_coordinate)
                if is_first:
                    sur_v, visit_status = adjacent_voxel_list[0]
                    next_p_list = DFS_traversal(sur_v)  # Process starts here
                    # Collect the last partial result
                    for next_p_vox in next_p_list:
                        self.path_graph[next_p_vox] = self.path_num
                    self.path_graph[curr_coordinate] = self.path_num
                else:
                    self.path_num += 1  # When the path end, update the idx
                    p_list = [curr_coordinate, ]
                    return p_list
            # When meet the voxel degree = 2, it's a general path voxel, continue and keep one path,
            # return list including itself
            elif degree == 2:
                for sur_v, visit_status in adjacent_voxel_list:
                    if visit_status == 1:  # Visit the unvisited
                        p_list = DFS_traversal(sur_v)
                        p_list.append(curr_coordinate)
                        return p_list
                # If escape the cycle, that means found a circle, treat it as an end
                self.path_num += 1  # When the path end, update the idx
                p_list = [curr_coordinate, ]
                return p_list
            # When meet the voxel degree = 2, it's the cross, update path and spread,
            # don't return itself
            elif degree > 2:
                self.path_graph[curr_coordinate] = self.cross_marker
                # self.cross_voxel_list.append(curr_coordinate)
                for sur_v, visit_status in adjacent_voxel_list:
                    if visit_status == 1:  # Visit the unvisited
                        next_p_list = DFS_traversal(sur_v)
                        # Collect the partial result
                        for next_p_vox in next_p_list:
                            self.path_graph[next_p_vox] = self.path_num

                self.path_num += 1  # When the path end, update the idx
                this_p_list = []
                return this_p_list

        while True:
            unvisited = np.where(self.visited_graph__ == 1)

            # When every voxel has been visited, break
            if len(unvisited[0]) == 0:
                break

            # Find a start point (degree = 1)
            unvisited_voxel_list = tk3.tuple_to_list(unvisited)
            start_coordinate = None
            i = 1
            for unv_vox in unvisited_voxel_list:
                degree, _ = self.get_adjacent_info(unv_vox, self.visited_graph__)
                if degree == 1:
                    start_coordinate = unv_vox
                    break
            if start_coordinate is None:
                print("No start coordinate found")
                exit(1)
            # print('start_coordinate: ' + str(start_coordinate))

            DFS_traversal(start_coordinate, is_first=True)

        # Part 2: Remove the empty path_id (where the path with path_id has 0 length)
        corrected_path_graph = np.where(self.path_graph == self.cross_marker, self.cross_marker, 0)
        corrected_path_id = 0
        for path_id in range(1, self.path_num + 1):
            if np.sum(self.path_graph == path_id) > 0:
                if np.sum(self.path_graph == path_id) <= 2:
                    corrected_path_graph += np.where(self.path_graph == path_id, self.cross_marker, 0)
                else:
                    corrected_path_id += 1
                    corrected_path_graph += np.where(self.path_graph == path_id, corrected_path_id, 0)
        self.path_graph = corrected_path_graph
        self.path_num = corrected_path_id
        # print('path num: ' + str(self.path_num))
        # print('total skele length: ' + str(np.sum(np.where(self.path_graph > 0, 1, 0))))

    def generate_cross_block_list(self):

        self.history_check(['generate_path_graph'])
        self.history_update('generate_cross_block_list')

        # Get all the cross point
        cross_voxel_list = tk3.tuple_to_list(np.where(self.path_graph == self.cross_marker))

        while len(cross_voxel_list) > 0:
            cross_voxel = cross_voxel_list.pop()

            # Get the adjacent voxel and remove them from [cross_voxel_list]
            current_crvs = [cross_voxel, ]
            adjacent_list = []
            while True:
                # Cycle layer 1: Check if there are more cross point found in cascade
                round_new_add_list = []

                for current_crv in current_crvs:
                    # Cycle layer 2: Search the newly found cross point

                    # Find new cross point and save in [sub_round_new_add_list]
                    sub_round_new_add_list = []
                    for crv in cross_voxel_list:
                        # Cycle layer 3: Walk through cross_voxel_list
                        if 0 < tk3.get_distance(crv, current_crv) < 2:
                            adjacent_list.append(crv)
                            round_new_add_list.append(crv)
                            sub_round_new_add_list.append(crv)

                    # Clear the newly found cross point in [cross_voxel_list]
                    for crv in sub_round_new_add_list:
                        cross_voxel_list.remove(crv)

                if len(round_new_add_list) == 0:
                    break
                else:
                    current_crvs = round_new_add_list.copy()

            # Create CrossBlock
            cross_block = CrossBlock()

            # Inject current cross voxel and its adjacent cross voxels into [CrossBlock.cross_point_list]
            cross_block.cross_voxel_list.append(cross_voxel)
            cross_block.cross_voxel_list.extend(adjacent_list)
            cross_block.size = len(cross_block.cross_voxel_list)

            # Detect the connection and fill [CrossBlock.connection_list]
            distant_graph = np.ones(self.shape)
            for crv in cross_block.cross_voxel_list:
                distant_graph[crv] = 0
            distant_graph = ndimage.distance_transform_edt(distant_graph)
            detect_range = tk3.tuple_to_list(np.where(distant_graph < 2))
            for crv in cross_block.cross_voxel_list:
                if tk3.is_in_list(crv, detect_range):
                    detect_range.remove(crv)
            for detect_voxel in detect_range:
                path_id = int(self.path_graph[detect_voxel])
                if path_id > 0:
                    cross_block.degree += 1
                    new_connection = (detect_voxel, path_id)
                    cross_block.connection_list.append(new_connection)

            # Add this CrossBlock into [self.cross_block_list]
            self.cross_block_list.append(cross_block)

            # Link cross block and cross point
            for point in self.skele_point_list:
                if tk3.is_in_list(point.coordinate, cross_block.cross_voxel_list):
                    point.cross_voxel = True
                    point.cross_voxel = cross_block
                    cross_block.cross_point_list.append(point)

    def inject_path_id(self):
        """
        Write the path graph info into [self.skele_point_list]
        :return:
        """
        self.history_check(['generate_path_graph'])

        for skele_point in self.skele_point_list:
            path_id = int(self.path_graph[skele_point.coordinate])

            # cross path voxel
            if self.path_graph[skele_point.coordinate] == self.cross_marker:
                pass
            # normal path voxel
            elif self.path_graph[skele_point.coordinate] > 0:
                skele_point.path_id = path_id
                # end path voxel
                if tk3.is_in_list(skele_point.coordinate, self.end_voxel_list):
                    skele_point.is_end = True

            # error
            else:
                print('error in [inject_path_id]:' + str(skele_point.coordinate) + ' in [self.path_graph] is '
                      + str(self.path_graph[skele_point.coordinate]))
                print(self.path_graph[skele_point.coordinate])
                print(self.path_graph[skele_point.coordinate] == 0)
                exit(1)

    def cut_branch(self):
        """
        Abandoned method
        :return:
        """

        def get_path_average_radius():
            path_radius_dict = {}
            for skele_point in self.skele_point_list:
                if not skele_point.is_cross:
                    if tk3.is_in_list(str(skele_point.path_id), path_radius_dict.keys()):
                        path_radius_dict[str(skele_point.path_id)].append(skele_point.radius)
                    else:
                        path_radius_dict[str(skele_point.path_id)] = [skele_point.radius, ]
            path_avg_radius_dict = {}
            for path_id in path_radius_dict.keys():
                sum = 0
                for r in path_radius_dict[path_id]:
                    sum += r
                path_avg_radius_dict[path_id] = sum / len(path_radius_dict[path_id])
            return path_avg_radius_dict

        path_avg_radius_dict = get_path_average_radius()

        # Collect all the path point (except cross) in [self.skele_point_list_cb]
        for skele_point in self.skele_point_list:
            if not skele_point.is_cross:
                self.skele_point_list_cb.append(skele_point)

        # Strategy 2: Remove min radius branch
        for cb in self.cross_block_list:
            remove_path_id = -1
            min_radius = 10000
            for crv, path_id in cb.connection_list:
                avg_radius = path_avg_radius_dict[str(path_id)]
                if avg_radius < min_radius:
                    remove_path_id = path_id
                    min_radius = avg_radius
            if remove_path_id == -1 or min_radius == 0:
                print('big error, keep_path_id not found')
                exit(99)

            # Cut branch
            cut_range = np.ones(self.shape)
            for crv in cb.cross_voxel_list:
                cut_range[crv] = 0
            cut_range = ndimage.distance_transform_edt(cut_range)
            cut_range = np.where(cut_range <= min_radius * 10, 1, 0)

            skele_point_remove_list = []
            for point in self.skele_point_list_cb:
                if cut_range[point.coordinate] == 1 and self.path_graph[point.coordinate] == remove_path_id:
                    skele_point_remove_list.append(point)

            for point in skele_point_remove_list:
                self.skele_point_list_cb.remove(point)

    def generate_part_graph_normal(self, mode=0, path_graph=None):
        """
        Generate part graph in traditional way
        :param path_graph: use input path_graph
        :param mode: 0 - Normal with [self.skele_point_list]
                     1 - Normal with [self.skele_point_list_cb]
                     2 - Use [path_graph]
        :return:
        """
        self.history_check(['generate_skele_point_list'])
        vessel_coordinate_list = tk3.tuple_to_list(np.where(self.vessel > 0))
        distance_graph_list = []
        huge_number = self.shape[0] ** 2 + self.shape[1] ** 2 + self.shape[2] ** 2

        if mode == 0 or mode == 1:
            '''Use [self.skele_point_list]'''
            if mode == 0:
                skele_point_list = self.skele_point_list
            else:
                self.history_check(['cut_branch'])
                skele_point_list = self.skele_point_list_cb
            for skele_point in skele_point_list:
                # Cross skeleton voxels do not participate in softmax
                if skele_point.is_cross:
                    continue
                # weight = log(skele_point.radius + 1)
                weight = skele_point.radius ** (1 / 3)
                path_id = skele_point.path_id
                point_distance_graph = np.ones(self.shape)
                point_distance_graph[skele_point.coordinate] = 0
                point_distance_graph = ndimage.distance_transform_edt(point_distance_graph) / weight
                distance_graph_list.append((point_distance_graph * weight, path_id))

        elif mode == 2:
            if path_graph is None:
                logging.warning("No parameter [path_graph]")
                path_graph = self.path_graph
            voxel_list = tk3.tuple_to_list(np.where(np.where(path_graph < self.cross_marker, path_graph, 0) > 0))
            for voxel in voxel_list:
                path_id = path_graph[voxel]
                radius = self.radius_graph[voxel]
                weight = radius ** (1 / 3)
                point_distance_graph = np.ones(self.shape)
                point_distance_graph[voxel] = 0
                point_distance_graph = ndimage.distance_transform_edt(point_distance_graph) / weight
                distance_graph_list.append((point_distance_graph * weight, path_id))

        for vessel_coordinate in vessel_coordinate_list:
            min_distance = huge_number
            min_path_id = -1
            for distance_graph, path_id in distance_graph_list:
                distance = distance_graph[vessel_coordinate]
                if distance < min_distance:
                    min_distance = distance
                    min_path_id = path_id
            self.part_graph[vessel_coordinate] = min_path_id

    def generate_part_graph_simulator(self, path_graph):
        """
        Generate part graph in simulator way
        Strategy: Long path first, expand (dilate) path to simulate vessel, classify the rest voxels
        :param path_graph:
        :return:
        """
        self.history_check(['generate_radius_graph'])
        huge_number = self.shape[0] ** 2 + self.shape[1] ** 2 + self.shape[2] ** 2
        pure_path_graph = self.purify_path_graph(path_graph).astype(int)
        path_num = np.max(pure_path_graph)
        path_list = []
        # print('path_num: ' + str(path_num) + ' ' + str(type(path_num)))
        for path_id in range(1, path_num + 1):
            path_length = np.sum(path_graph == path_id)
            path_list.append((path_id, path_length))

        # Sort paths by their length (longest -> shortest)
        path_list = sorted(path_list, key=lambda path_tuple: path_tuple[1], reverse=True)

        simulate_graph = np.zeros(self.shape)
        dist_graph_dict = {}
        for path_id, _ in path_list:
            path_simulate_graph = np.zeros(self.shape)
            radius_list_dict = {}

            for pv in tk3.tuple_to_list(np.where(pure_path_graph == path_id)):
                radius = str(int(self.radius_graph[pv]) + 1)
                if radius not in radius_list_dict.keys():
                    radius_list_dict[radius] = []
                radius_list_dict[radius].append(pv)
            print('path ' + str(path_id) + ': ' + str(radius_list_dict))
            for radius in radius_list_dict.keys():
                single_voxel_graph = np.ones(self.shape)
                for pv in radius_list_dict[radius]:
                    single_voxel_graph[pv] = 0
                single_voxel_graph = np.where(ndimage.distance_transform_edt(single_voxel_graph) < int(radius), 1, 0)
                path_simulate_graph += single_voxel_graph
            path_simulate_graph = np.where(path_simulate_graph > 0, path_id, 0)
            dist_graph_dict[str(path_id)] = ndimage.distance_transform_edt(np.where(path_simulate_graph > 0, 0, 1))
            available_mask = np.where(simulate_graph == 0, 1, 0)
            path_simulate_graph = np.multiply(path_simulate_graph, available_mask)
            simulate_graph += path_simulate_graph
            simulate_graph = np.multiply(simulate_graph, self.vessel)

        skipped_mask = np.multiply(np.where(simulate_graph > 0, 0, 1), self.vessel)
        compensate_graph = np.zeros(self.shape)
        min_dist_graph = np.ones(self.shape) * huge_number
        for path_id in dist_graph_dict.keys():
            dist_graph = dist_graph_dict[path_id]
            path_id = int(path_id)
            compensate_graph = np.where(dist_graph < min_dist_graph, path_id, compensate_graph)
            min_dist_graph = np.where(dist_graph < min_dist_graph, dist_graph, min_dist_graph)

        compensate_graph = np.multiply(skipped_mask, compensate_graph)

        part_graph = compensate_graph + simulate_graph

        self.part_graph = part_graph

    def purify_path_graph(self, path_graph, keep_cross=False):
        """
        Get the purified path_graph:
            * Get rid of cross voxels (choice)
            * Remove empty path_id (compress path_id range)
        :param keep_cross: keep cross voxels ?
        :param path_graph:
        :return: purified_path_graph
        """
        if keep_cross:
            cross_graph = np.where(path_graph == self.cross_marker, self.cross_marker, 0)
        path_graph = np.where(path_graph < self.cross_marker, path_graph, 0)
        path_num = np.max(path_graph)
        forward_path_id = 1
        backward_path_id = path_num
        while backward_path_id > forward_path_id:
            if np.sum(path_graph == forward_path_id) == 0:
                if np.sum(path_graph == backward_path_id) == 0:
                    backward_path_id -= 1
                else:
                    path_graph = np.where(path_graph == backward_path_id, forward_path_id, path_graph)
                    print('switch ' + str(backward_path_id) + ' to ' + str(forward_path_id))
                    forward_path_id += 1
                    backward_path_id -= 1
            else:
                forward_path_id += 1
        if keep_cross:
            path_graph += cross_graph
        return path_graph

    def generate_ordered_path(self):
        self.history_check(['generate_cross_block_list',
                            'generate_path_graph'])
        self.history_update('generate_ordered_path')
        normal_path_length = 0
        for path_id in range(1, self.path_num + 1):
            p_len = int(np.sum(np.where(self.path_graph == path_id, 1, 0)))
            normal_path_length += p_len
        cross_length = 0
        for cb in self.cross_block_list:
            cross_length += cb.size

        for path_id in range(1, self.path_num + 1):
            inordered_path_voxel_list = tk3.tuple_to_list(np.where(self.path_graph == path_id))
            ordered_path_point_list = []

            # Part 1: Find the one of the ends of a path
            first_pv = None
            single_path_graph = np.where(self.path_graph == path_id, 1, 0)
            for pv in inordered_path_voxel_list:
                degree, _ = self.get_adjacent_info(pv, graph=single_path_graph)
                if degree == 1:
                    first_pv = pv
                    # print('path ' + str(path_id) + ' find: ' + str(first_pv))
                    break
            if first_pv is None:
                print('not find first_pv, maybe a cycle')
                print(inordered_path_voxel_list)
                exit(111)

            # Part 2: Order the path point
            pv = first_pv
            last_pv = None
            is_first = True
            while True:
                point = get_point_by_coordinate(self.skele_point_list, pv)
                ordered_path_point_list.append(point)
                degree, adjacent_voxel_list = self.get_adjacent_info(pv, graph=single_path_graph)
                if degree == 1:
                    if is_first:
                        last_pv = pv
                        pv = adjacent_voxel_list[0][0]
                        is_first = False
                    else:
                        break
                elif degree == 2:
                    if adjacent_voxel_list[0][0] == last_pv:
                        last_pv = pv
                        pv = adjacent_voxel_list[1][0]
                    elif adjacent_voxel_list[1][0] == last_pv:
                        last_pv = pv
                        pv = adjacent_voxel_list[0][0]
                    else:
                        exit(1213131)
                else:
                    exit(10201)
            self.ordered_path_point_list_dict[str(path_id)] = ordered_path_point_list

    def get_thin_point_list(self):

        self.history_check(['generate_ordered_path'])

        if self.save_mark:
            self.thin_mark_graph = np.zeros(self.shape)

        for path_id in self.ordered_path_point_list_dict.keys():

            path_point_list = self.ordered_path_point_list_dict[path_id]

            # 0927 测试改动：将最大半径替换为平均半径
            if self.avg_radius:
                avg_path_radius = 0.0
                for point in path_point_list:
                    avg_path_radius += point.radius
                avg_path_radius /= len(path_point_list)
                path_radius_standard = avg_path_radius
            else:
                max_path_radius = 0.0
                for point in path_point_list:
                    if point.radius > max_path_radius:
                        max_path_radius = point.radius
                path_radius_standard = max_path_radius

            radius_list = []
            for point in path_point_list:
                radius_list.append(point.radius)

            # rule 1: radius decrease -> increase

            minus_list = []
            for i in range(0, len(radius_list) - 1):
                minus = radius_list[i] - radius_list[i + 1]
                minus_list.append(minus)

            thin_indices = []
            for i in range(0, len(minus_list) - 1):
                if minus_list[i] < 0 and minus_list[i + 1] > 0:
                    thin_indices.append(i + 1)

            # rule 2: based on rule 1, must have: thin point radius <= path_radius_standard - self.max_min_radis_dist

            rough_thin_point_list = []
            for idx in thin_indices:
                if path_point_list[idx].radius <= path_radius_standard - self.max_min_radis_dist:
                    rough_thin_point_list.append(path_point_list[idx])

            # rule -1: too thin skeleton point

            for point in path_point_list:
                if point.radius <= path_radius_standard * self.thin_degree:
                    if point not in rough_thin_point_list:
                        rough_thin_point_list.append(point)

            # rule -2: for unconnected vessels

            for point in path_point_list:
                if point.is_end:
                    assert tk3.is_in_list(point.coordinate, self.end_voxel_list), "point is not in self.end_voxel_list"
                    rough_thin_point_list.append(point)

            # rule 3: based on rule 2, for every thin point, we cut its piece of vessel (related_vessel),
            # must have: distance [related_vessel -> tumor] < self.max_vessel_tumor_dist

            for point in rough_thin_point_list:
                related_vessel_mask = np.ones(self.shape)
                related_vessel_mask[point.coordinate] = 0
                related_vessel_mask = ndimage.distance_transform_edt(related_vessel_mask)
                if point.is_end:
                    related_vessel_mask = np.where(related_vessel_mask <= point.radius + 4, 1, 0)
                else:
                    # related_vessel_mask = np.where(related_vessel_mask <= point.radius + 2, 1, 0)
                    # related_vessel_mask = np.where(related_vessel_mask <= point.radius + 1, 1, 0)  # 0926(best for CA CHA)
                    related_vessel_mask = np.where(related_vessel_mask <= point.radius + self.related_range_bias, 1, 0)  # 0926测试改动，计划用于SMA

                related_vessel = np.multiply(self.vessel, related_vessel_mask)
                if self.save_mark:
                    self.thin_mark_graph += related_vessel
                vessel_tumor_dist = tk3.get_inter_distance(related_vessel, self.tumor)
                if vessel_tumor_dist < self.max_vessel_tumor_dist:
                    thin_point = ThinPoint(point)
                    thin_point.vessel_tumor_dist = vessel_tumor_dist
                    thin_point.radis_dist = path_radius_standard - thin_point.radius
                    self.thin_point_list.append(thin_point)

        if self.save_mark:
            self.thin_mark_graph = np.where(self.thin_mark_graph > 0, 1, 0)


    def get_thin_feature_old(self):

        self.history_check(['generate_ordered_path'])

        if self.save_mark:
            self.thin_mark_graph = np.zeros(self.shape)

        if self.target == 'artery':
            path_point_dict = {'CA': [], 'CHA': [], 'SMA': []}
            feature_dict = {'CA': {}, 'CHA': {}, 'SMA': {}}
            vessel_mapping = {'2': 'CA', '3': 'CHA', '6': 'SMA'}
        else:
            path_point_dict = {'PV': [], 'SMV': []}
            feature_dict = {'PV': {}, 'SMV': {}}
            vessel_mapping = {'1': 'PV', '2': 'SMV'}
        tumor_dist_transform = ndimage.distance_transform_edt(np.where(self.tumor > 0, 0, 1))
        rough_thin_point_list = []

        for path_id in self.ordered_path_point_list_dict.keys():

            path_point_list = self.ordered_path_point_list_dict[path_id]

            # 0927 测试改动：将最大半径替换为平均半径
            if self.avg_radius:
                avg_path_radius = 0.0
                for point in path_point_list:
                    avg_path_radius += point.radius
                avg_path_radius /= len(path_point_list)
                path_radius_standard = avg_path_radius
            else:
                max_path_radius = 0.0
                for point in path_point_list:
                    if point.radius > max_path_radius:
                        max_path_radius = point.radius
                path_radius_standard = max_path_radius

            radius_list = []
            for point in path_point_list:
                radius_list.append(point.radius)

            # rule 1: radius decrease -> increase

            minus_list = []
            for i in range(0, len(radius_list) - 1):
                minus = radius_list[i] - radius_list[i + 1]
                minus_list.append(minus)

            thin_indices = []
            for i in range(0, len(minus_list) - 1):
                if minus_list[i] < 0 and minus_list[i + 1] > 0:
                    thin_indices.append(i + 1)

            # rule 2: based on rule 1, must have: thin point radius <= path_radius_standard - self.max_min_radis_dist

            for idx in thin_indices:
                if path_point_list[idx].radius <= path_radius_standard - self.max_min_radis_dist:
                    rough_thin_point_list.append(path_point_list[idx])

            # rule -1: too thin skeleton point

            for point in path_point_list:
                if point.radius <= path_radius_standard * self.thin_degree:
                    if point not in rough_thin_point_list:
                        rough_thin_point_list.append(point)

            # rule -2: for unconnected vessels

            for point in path_point_list:
                if point.is_end:
                    assert tk3.is_in_list(point.coordinate, self.end_voxel_list), "point is not in self.end_voxel_list"
                    rough_thin_point_list.append(point)

            # rule 3: based on rule 2, for every thin point, we cut its piece of vessel (related_vessel),
            # must have: distance [related_vessel -> tumor] < self.max_vessel_tumor_dist

        for point in rough_thin_point_list:
            part_id = self.part_img[point.coordinate]
            if str(part_id) in vessel_mapping.keys():
                path_point_dict[vessel_mapping[str(part_id)]].append(point)



        for part_name in path_point_dict.keys():
            # 2.1. 直接特征
            radius_list = []
            tumor_dist_list = []
            syn_rt_list = []

            if len(path_point_dict[part_name]) == 0:
                feature_dict[part_name]['radius_min'] = -1
                feature_dict[part_name]['radius_max'] = -1
                feature_dict[part_name]['radius_mean'] = -1
                feature_dict[part_name]['radius_ptp'] = -1
                feature_dict[part_name]['radius_var'] = -1
                feature_dict[part_name]['radius_std'] = -1
                feature_dict[part_name]['syn_rt_min'] = -1
                feature_dict[part_name]['syn_rt_max'] = -1
                feature_dict[part_name]['syn_rt_mean'] = -1
                feature_dict[part_name]['syn_rt_ptp'] = -1
                feature_dict[part_name]['syn_rt_var'] = -1
                feature_dict[part_name]['syn_rt_std'] = -1
                feature_dict[part_name]['mean_ratio'] = -1
                feature_dict[part_name]['ptp_ratio'] = -1
                feature_dict[part_name]['var_ratio'] = -1
                feature_dict[part_name]['std_ratio'] = -1
                continue

            for point in path_point_dict[part_name]:
                radius_list.append(float(point.radius))
                tumor_dist_list.append(float(tumor_dist_transform[point.coordinate]))
            standard_radius = sum(radius_list) / len(radius_list)

            radius_min, radius_max, radius_mean, radius_ptp, radius_var, radius_std = tk3.basic_data_analysis(radius_list)
            print(f'radius_list: {radius_list}')
            print(
                f'\tmin = {radius_min:.2f}, max = {radius_max:.2f}, mean = {radius_mean:.2f}, ptp = {radius_ptp:.2f}, var = {radius_var:.2f}, std = {radius_std:.2f}')
            feature_dict[part_name]['radius_min'] = radius_min
            feature_dict[part_name]['radius_max'] = radius_max
            feature_dict[part_name]['radius_mean'] = radius_mean
            feature_dict[part_name]['radius_ptp'] = radius_ptp
            feature_dict[part_name]['radius_var'] = radius_var
            feature_dict[part_name]['radius_std'] = radius_std

            for radius, tumor_dist in zip(radius_list, tumor_dist_list):
                syn_rt_list.append(radius * tumor_dist)

            syn_rt_min, syn_rt_max, syn_rt_mean, syn_rt_ptp, syn_rt_var, syn_rt_std = tk3.basic_data_analysis(syn_rt_list)
            print(f'syn_rt_list: {syn_rt_list}')
            print(f'\tmin = {syn_rt_min:.2f}, max = {syn_rt_max:.2f}, mean = {syn_rt_mean:.2f}, ptp = {syn_rt_ptp:.2f}, var = {syn_rt_var:.2f}, std = {syn_rt_std:.2f}')
            feature_dict[part_name]['syn_rt_min'] = syn_rt_min
            feature_dict[part_name]['syn_rt_max'] = syn_rt_max
            feature_dict[part_name]['syn_rt_mean'] = syn_rt_mean
            feature_dict[part_name]['syn_rt_ptp'] = syn_rt_ptp
            feature_dict[part_name]['syn_rt_var'] = syn_rt_var
            feature_dict[part_name]['syn_rt_std'] = syn_rt_std
        return feature_dict



    def get_thin_feature(self):

        self.history_check(['generate_ordered_path'])

        min_tumor_dist_thresh = 5

        tumor_dist_transform = ndimage.distance_transform_edt(np.where(self.tumor > 0, 0, 1))

        if self.target == 'artery':
            path_dict = {'CA': [], 'CHA': [], 'SMA': []}
            feature_dict = {'CA': {}, 'CHA': {}, 'SMA': {}}
            vessel_mapping = {'2': 'CA', '3': 'CHA', '6': 'SMA'}
        else:
            path_dict = {'PV': [], 'SMV': []}
            feature_dict = {'PV': {}, 'SMV': {}}
            vessel_mapping = {'1': 'PV', '2': 'SMV'}

        # 1. 粗筛选：将单支 path 内，骨架与肿瘤最近距离不大于 min_tumor_dist_thresh 的 path 筛选出来，并按归属的 part 合并存储
        for path_id in self.ordered_path_point_list_dict.keys():

            path_point_list = self.ordered_path_point_list_dict[path_id]

            path_point_part_list = []
            tumor_dist_list = []
            for point in path_point_list:
                path_point_part_list.append(self.part_img[point.coordinate])
                tumor_dist_list.append(float(tumor_dist_transform[point.coordinate]))
            min_tumor_dist = min(tumor_dist_list)
            if min_tumor_dist >= min_tumor_dist_thresh:
                print(f'Path {path_id} is not near tumor, min_tumor_dist = {min_tumor_dist}')
                continue
            print(f'path_point_part_list: {path_point_part_list}')
            from scipy import stats
            path_part_id = stats.mode(path_point_part_list)[0]
            print(f'path {path_id} belongs to {path_part_id}')
            if str(path_part_id) in vessel_mapping.keys():
                path_dict[vessel_mapping[str(path_part_id)]].extend(path_point_list)

        # 2. 特征计算：对于一个 part 内的所有点，计算直接特征和比值特征
        for part_name in path_dict.keys():
            path_point_list = path_dict[part_name]
            if len(path_point_list) == 0:
                feature_dict[part_name]['radius_min'] = -1
                feature_dict[part_name]['radius_max'] = -1
                feature_dict[part_name]['radius_mean'] = -1
                feature_dict[part_name]['radius_ptp'] = -1
                feature_dict[part_name]['radius_var'] = -1
                feature_dict[part_name]['radius_std'] = -1
                feature_dict[part_name]['syn_rt_min'] = -1
                feature_dict[part_name]['syn_rt_max'] = -1
                feature_dict[part_name]['syn_rt_mean'] = -1
                feature_dict[part_name]['syn_rt_ptp'] = -1
                feature_dict[part_name]['syn_rt_var'] = -1
                feature_dict[part_name]['syn_rt_std'] = -1
                feature_dict[part_name]['mean_ratio'] = -1
                feature_dict[part_name]['ptp_ratio'] = -1
                feature_dict[part_name]['var_ratio'] = -1
                feature_dict[part_name]['std_ratio'] = -1
                continue

            # 2.1. 直接特征
            radius_list = []
            tumor_dist_list = []
            syn_rt_list = []
            for point in path_point_list:
                radius_list.append(float(point.radius))
                tumor_dist_list.append(float(tumor_dist_transform[point.coordinate]))
            standard_radius = sum(radius_list)/len(radius_list)

            radius_min, radius_max, radius_mean, radius_ptp, radius_var, radius_std = tk3.basic_data_analysis(
                radius_list)
            print(f'radius_list: {radius_list}')
            print(
                f'\tmin = {radius_min:.2f}, max = {radius_max:.2f}, mean = {radius_mean:.2f}, ptp = {radius_ptp:.2f}, var = {radius_var:.2f}, std = {radius_std:.2f}')
            feature_dict[part_name]['radius_min'] = radius_min
            feature_dict[part_name]['radius_max'] = radius_max
            feature_dict[part_name]['radius_mean'] = radius_mean
            feature_dict[part_name]['radius_ptp'] = radius_ptp
            feature_dict[part_name]['radius_var'] = radius_var
            feature_dict[part_name]['radius_std'] = radius_std

            for radius, tumor_dist in zip(radius_list, tumor_dist_list):
                syn_rt_list.append(radius * tumor_dist)

            syn_rt_min, syn_rt_max, syn_rt_mean, syn_rt_ptp, syn_rt_var, syn_rt_std = tk3.basic_data_analysis(
                syn_rt_list)
            print(f'syn_rt_list: {syn_rt_list}')
            print(
                f'\tmin = {syn_rt_min:.2f}, max = {syn_rt_max:.2f}, mean = {syn_rt_mean:.2f}, ptp = {syn_rt_ptp:.2f}, var = {syn_rt_var:.2f}, std = {syn_rt_std:.2f}')
            feature_dict[part_name]['syn_rt_min'] = syn_rt_min
            feature_dict[part_name]['syn_rt_max'] = syn_rt_max
            feature_dict[part_name]['syn_rt_mean'] = syn_rt_mean
            feature_dict[part_name]['syn_rt_ptp'] = syn_rt_ptp
            feature_dict[part_name]['syn_rt_var'] = syn_rt_var
            feature_dict[part_name]['syn_rt_std'] = syn_rt_std

            # 2.2. 比值特征
            # ROI 中的骨架点为：半径小于标准半径，且，与肿瘤距离小于阈值
            path_point_inROI_list = []
            radius_inROI_list = []
            tumor_dist_inROI_list = []
            path_point_outROI_list = []
            radius_outROI_list = []
            tumor_dist_outROI_list = []
            for i in range(0, len(path_point_list)):
                if radius_list[i] >= standard_radius or tumor_dist_list[i] >= min_tumor_dist_thresh:
                    path_point_outROI_list.append(path_point_list[i])
                    radius_outROI_list.append(radius_list[i])
                    tumor_dist_outROI_list.append(tumor_dist_list[i])
                else:
                    path_point_inROI_list.append(path_point_list[i])
                    radius_inROI_list.append(radius_list[i])
                    tumor_dist_inROI_list.append(tumor_dist_list[i])

            if len(path_point_inROI_list) == 0:
                feature_dict[part_name]['mean_ratio'] = -2
                feature_dict[part_name]['ptp_ratio'] = -2
                feature_dict[part_name]['var_ratio'] = -2
                feature_dict[part_name]['std_ratio'] = -2
                continue
            elif len(path_point_outROI_list) == 0:
                feature_dict[part_name]['mean_ratio'] = -3
                feature_dict[part_name]['ptp_ratio'] = -3
                feature_dict[part_name]['var_ratio'] = -3
                feature_dict[part_name]['std_ratio'] = -3
                continue

            print(f'In ROI:')
            radius_inROI_mean, radius_inROI_ptp, radius_inROI_var, radius_inROI_std = tk3.basic_data_analysis(radius_inROI_list)
            print(f'radius_list: {radius_inROI_list}')
            print(f'\tmean = {radius_inROI_mean:.2f}, ptp = {radius_inROI_ptp:.2f}, var = {radius_inROI_var:.2f}, std = {radius_inROI_std:.2f}')

            syn_rt_inROI_list = []
            for radius, tumor_dist in zip(radius_inROI_list, tumor_dist_inROI_list):
                syn_rt_inROI_list.append(radius * tumor_dist)

            syn_rt_inROI_mean, syn_rt_inROI_ptp, syn_rt_inROI_var, syn_rt_inROI_std = tk3.basic_data_analysis(syn_rt_inROI_list)
            print(f'syn_rt_list: {syn_rt_inROI_list}')
            print(f'\tmean = {syn_rt_inROI_mean:.2f}, ptp = {syn_rt_inROI_ptp:.2f}, var = {syn_rt_inROI_var:.2f}, std = {syn_rt_inROI_std:.2f}')

            print(f'Out ROI:')
            radius_outROI_mean, radius_outROI_ptp, radius_outROI_var, radius_outROI_std = tk3.basic_data_analysis(radius_outROI_list)
            print(f'radius_list: {radius_outROI_list}')
            print(f'\tmean = {radius_outROI_mean:.2f}, ptp = {radius_outROI_ptp:.2f}, var = {radius_outROI_var:.2f}, std = {radius_outROI_std:.2f}')

            syn_rt_outROI_list = []
            for radius, tumor_dist in zip(radius_outROI_list, tumor_dist_outROI_list):
                syn_rt_outROI_list.append(radius * tumor_dist)

            syn_rt_outROI_mean, syn_rt_outROI_ptp, syn_rt_outROI_var, syn_rt_outROI_std = tk3.basic_data_analysis(syn_rt_outROI_list)
            print(f'syn_rt_list: {syn_rt_outROI_list}')
            print(f'\tmean = {syn_rt_outROI_mean:.2f}, ptp = {syn_rt_outROI_ptp:.2f}, var = {syn_rt_outROI_var:.2f}, std = {syn_rt_outROI_std:.2f}')

            mean_ratio = syn_rt_outROI_mean / syn_rt_inROI_mean
            ptp_ratio = syn_rt_outROI_ptp / syn_rt_inROI_ptp
            var_ratio = syn_rt_outROI_var / syn_rt_inROI_var
            std_ratio = syn_rt_outROI_std / syn_rt_inROI_std

            print(f'Out/In Ratio:')
            print(f'\tmean = {mean_ratio:.2f}, ptp = {ptp_ratio:.2f}, var = {var_ratio:.2f}, std = {std_ratio:.2f}')
            print()

            feature_dict[part_name]['mean_ratio'] = mean_ratio
            feature_dict[part_name]['ptp_ratio'] = ptp_ratio
            feature_dict[part_name]['var_ratio'] = var_ratio
            feature_dict[part_name]['std_ratio'] = std_ratio

            print(f'feature_dict: {feature_dict[part_name]}')

        return feature_dict


        #     for point in rough_thin_point_list:
        #         related_vessel_mask = np.ones(self.shape)
        #         related_vessel_mask[point.coordinate] = 0
        #         related_vessel_mask = ndimage.distance_transform_edt(related_vessel_mask)
        #         if point.is_end:
        #             related_vessel_mask = np.where(related_vessel_mask <= point.radius + 4, 1, 0)
        #         else:
        #             # related_vessel_mask = np.where(related_vessel_mask <= point.radius + 2, 1, 0)
        #             # related_vessel_mask = np.where(related_vessel_mask <= point.radius + 1, 1, 0)  # 0926(best for CA CHA)
        #             related_vessel_mask = np.where(related_vessel_mask <= point.radius + self.related_range_bias, 1, 0)  # 0926测试改动，计划用于SMA
        #
        #     # 0927 测试改动：将最大半径替换为平均半径
        #     if self.avg_radius:
        #         avg_path_radius = 0.0
        #         for point in path_point_list:
        #             avg_path_radius += point.radius
        #         avg_path_radius /= len(path_point_list)
        #         path_radius_standard = avg_path_radius
        #     else:
        #         max_path_radius = 0.0
        #         for point in path_point_list:
        #             if point.radius > max_path_radius:
        #                 max_path_radius = point.radius
        #         path_radius_standard = max_path_radius
        #
        #     radius_list = []
        #     for point in path_point_list:
        #         radius_list.append(point.radius)
        #
        #     # rule 1: radius decrease -> increase
        #
        #     minus_list = []
        #     for i in range(0, len(radius_list) - 1):
        #         minus = radius_list[i] - radius_list[i + 1]
        #         minus_list.append(minus)
        #
        #     thin_indices = []
        #     for i in range(0, len(minus_list) - 1):
        #         if minus_list[i] < 0 and minus_list[i + 1] > 0:
        #             thin_indices.append(i + 1)
        #
        #     # rule 2: based on rule 1, must have: thin point radius <= path_radius_standard - self.max_min_radis_dist
        #
        #     rough_thin_point_list = []
        #     for idx in thin_indices:
        #         if path_point_list[idx].radius <= path_radius_standard - self.max_min_radis_dist:
        #             rough_thin_point_list.append(path_point_list[idx])
        #
        #     # rule -1: too thin skeleton point
        #
        #     for point in path_point_list:
        #         if point.radius <= path_radius_standard * self.thin_degree:
        #             if point not in rough_thin_point_list:
        #                 rough_thin_point_list.append(point)
        #
        #     # rule -2: for unconnected vessels
        #
        #     for point in path_point_list:
        #         if point.is_end:
        #             assert tk3.is_in_list(point.coordinate, self.end_voxel_list), "point is not in self.end_voxel_list"
        #             rough_thin_point_list.append(point)
        #
        #     # rule 3: based on rule 2, for every thin point, we cut its piece of vessel (related_vessel),
        #     # must have: distance [related_vessel -> tumor] < self.max_vessel_tumor_dist
        #
        #     for point in rough_thin_point_list:
        #         related_vessel_mask = np.ones(self.shape)
        #         related_vessel_mask[point.coordinate] = 0
        #         related_vessel_mask = ndimage.distance_transform_edt(related_vessel_mask)
        #         if point.is_end:
        #             related_vessel_mask = np.where(related_vessel_mask <= point.radius + 4, 1, 0)
        #         else:
        #             # related_vessel_mask = np.where(related_vessel_mask <= point.radius + 2, 1, 0)
        #             # related_vessel_mask = np.where(related_vessel_mask <= point.radius + 1, 1, 0)  # 0926(best for CA CHA)
        #             related_vessel_mask = np.where(related_vessel_mask <= point.radius + self.related_range_bias, 1, 0)  # 0926测试改动，计划用于SMA
        #
        #         related_vessel = np.multiply(self.vessel, related_vessel_mask)
        #         if self.save_mark:
        #             self.thin_mark_graph += related_vessel
        #         vessel_tumor_dist = tk3.get_inter_distance(related_vessel, self.tumor)
        #         if vessel_tumor_dist < self.max_vessel_tumor_dist:
        #             thin_point = ThinPoint(point)
        #             thin_point.vessel_tumor_dist = vessel_tumor_dist
        #             thin_point.radis_dist = path_radius_standard - thin_point.radius
        #             self.thin_point_list.append(thin_point)
        #
        # if self.save_mark:
        #     self.thin_mark_graph = np.where(self.thin_mark_graph > 0, 1, 0)


    @staticmethod
    def get_direction(start, end):
        if isinstance(start, Point) and isinstance(end, Point):
            start_vox = start.coordinate
            end_vox = end.coordinate
        elif isinstance(start, tuple) and isinstance(end, tuple):
            start_vox = start
            end_vox = end
        else:
            print("[get_direction] error")
            exit(-1)
        direction = (end_vox[0] - start_vox[0], end_vox[1] - start_vox[1], end_vox[2] - start_vox[2])
        if direction[0] < 0:
            direction = (-direction[0], -direction[1], -direction[2])
        direction_length = tk3.get_distance(start_vox, end_vox)
        direction = (direction[0] / direction_length, direction[1] / direction_length, direction[2] / direction_length)
        return direction

    def get_path_direction(self, path_id):
        """
        NEED: exec [self.generate_ordered_path] before
        :param path_id:
        :return:
        """
        self.history_check(['generate_ordered_path'])
        ordered_path_point_list = self.ordered_path_point_list_dict[str(path_id)]
        start_point = ordered_path_point_list[0]
        end_point = ordered_path_point_list[-1]

        # start_x, start_y, start_z = start_point.coordinate
        # end_x, end_y, end_z = end_point.coordinate
        # direction = (end_x - start_x, end_y - start_y, end_z - start_z)
        # if direction[0] < 0:
        #     direction = (-direction[0], -direction[1], -direction[2])
        # direction_length = tk3.get_distance(start_point.coordinate, end_point.coordinate)
        # direction = (direction[0] / direction_length, direction[1] / direction_length, direction[2] / direction_length)

        direction = self.get_direction(start_point, end_point)

        return direction

    def get_point_direction(self, voxel):
        self.history_check(['generate_ordered_path'])

        direction_length = 2

        path_id = self.path_graph[voxel]
        assert path_id != self.cross_marker, f"Voxel {voxel} is a cross point."
        path_id = str(path_id)
        assert path_id in self.ordered_path_point_list_dict.keys(), f"Voxel {voxel} is not a legal path point."
        path_point_list = self.ordered_path_point_list_dict[path_id]
        p_idx = -1
        for idx in range(0, len(path_point_list)):
            p = path_point_list[idx]
            if p.coordinate[0] == voxel[0] and p.coordinate[1] == voxel[1] and p.coordinate[2] == voxel[2]:
                p_idx = idx
                break
        assert p_idx >= 0, "error"
        start_idx, end_idx = max(p_idx - direction_length, 0), min(p_idx + direction_length, len(path_point_list) - 1)

        direction = self.get_direction(path_point_list[start_idx], path_point_list[end_idx])

        return direction

    def get_path_num(self, path_graph):
        path_num = np.max(np.where(path_graph < self.cross_marker, path_graph, 0))
        return path_num

    def remove_short_path(self, path_graph, min_length=10):
        path_num = self.get_path_num(path_graph)

        for path_id in range(1, path_num + 1):
            if np.sum(path_graph == path_id) <= min_length:
                path_graph = np.where(path_graph == path_id, 0, path_graph)

        path_graph = self.purify_path_graph(path_graph, keep_cross=True)

        return path_graph

    def get_path_average_radius(self, path_graph, path_id):
        voxel_list = tk3.tuple_to_list(np.where(path_graph == path_id))
        sum_radius = 0
        for voxel in voxel_list:
            sum_radius += self.radius_graph[voxel]
        return sum_radius / len(voxel_list)

    def generate_y_path_graph(self, cross_position=None):

        def get_cross_block_score(cross_block):
            path_list = []
            cross_list = []

            def score_iterate(father_path_id):
                next_cb = None
                for cb in self.cross_block_list:
                    if tk3.is_in_list(cb, cross_list):
                        continue
                    find_cb = False
                    for _, path_id in cb.connection_list:
                        if path_id == father_path_id:
                            find_cb = True
                            break
                    if find_cb:
                        cross_list.append(cb)
                        next_cb = cb

                if next_cb is not None:
                    for _, son_path_id in next_cb.connection_list:
                        if not tk3.is_in_list(son_path_id, path_list):
                            path_list.append(son_path_id)
                            score_iterate(son_path_id)

            path_score_list = []
            for _, father_path_id in cross_block.connection_list:
                path_list = []
                cross_list = [cross_block]
                score_iterate(father_path_id)
                score_graph = np.zeros(self.shape)
                for path_id in path_list:
                    score_graph += np.where(self.path_graph == path_id, 1, 0)
                score_graph = np.multiply(score_graph, self.radius_graph)
                path_score = np.sum(score_graph)
                print('path_list: ' + str(path_list) + '  ' + str(len(path_list)) + '  length = ' + str(path_score))
                path_score_list.append(path_score)
            print('path_length_list: ' + str(path_score_list))
            score = 0
            for i in range(0, len(path_score_list)):
                score += abs(path_score_list[i] - path_score_list[i - 1])
            return score

        def trunk_iterate(father_path_id):
            next_cb = None
            for cb in self.cross_block_list:
                if tk3.is_in_list(cb, self.y_cross_list):
                    continue
                find_cb = False
                for _, path_id in cb.connection_list:
                    if path_id == father_path_id:
                        find_cb = True
                        break
                if find_cb:
                    self.y_cross_list.append(cb)
                    next_cb = cb

            if next_cb is not None:
                for _, son_path_id in next_cb.connection_list:
                    if not tk3.is_in_list(son_path_id, self.y_path_list):
                        self.y_path_list.append(son_path_id)
                        trunk_iterate(son_path_id)

        if cross_position is None:

            min_score_cb = None
            min_score = 114514
            for cb in self.cross_block_list:
                if cb.degree == 3:
                    score = get_cross_block_score(cb)
                    print('score = ' + str(score))
                    if score < min_score:
                        min_score = score
                        min_score_cb = cb
            print('final score = ' + str(min_score))
            if min_score_cb is None:
                return
            target_cb = min_score_cb
        else:
            target_cb = None
            min_distance = 114514
            for cb in self.cross_block_list:
                distance = tk3.get_distance(cb.cross_voxel_list[0], cross_position)
                if distance < min_distance:
                    min_distance = distance
                    target_cb = cb
        # max_radius_cb = None
        # max_radius = 0
        # for cb in self.cross_block_list:
        #     if cb.degree == 3:
        #         radius = 0
        #         count = 0
        #         for cv in cb.cross_voxel_list:
        #             radius += self.radius_graph[cv]
        #             count += 1
        #         radius /= count
        #
        #         if radius > max_radius:
        #             max_radius = radius
        #             max_radius_cb = cb
        # if max_radius_cb is None:
        #     return

        self.y_path_graph = np.zeros(self.shape)
        new_path_id = 1
        for _, path_id in target_cb.connection_list:
            temp_path_graph = np.zeros(self.shape)
            print('path_id = ' + str(new_path_id), end='  ')
            self.y_cross_list = []
            self.y_cross_list.append(target_cb)
            self.y_path_list.append(path_id)
            trunk_iterate(path_id)
            self.y_cross_list.remove(target_cb)

            for cb in self.y_cross_list:
                for voxel in cb.cross_voxel_list:
                    temp_path_graph[voxel] = 1
            for path_id in self.y_path_list:
                temp_path_graph += np.where(self.path_graph == path_id, 1, 0)
            new_path_id += 1

            self.y_path_graph += np.multiply(np.where(temp_path_graph > 0, new_path_id, 0),
                                             np.where(self.y_path_graph > 0, 0, 1))
            print('path_num = ' + str(self.get_path_num(self.y_path_graph)))

    def generate_artery_path_graph(self):
        # self.branch_path_graph = None
        # self.branch_cross_list = []
        # self.branch_path_list = []

        def trunk_iterate(father_path_id, cross_list, path_list):

            # Calculate father path direction (3D vector, length = 1)
            print('------> father path:' + str(father_path_id))
            father_direction = self.get_path_direction(father_path_id)

            # father_path  --find-->  Cross_block
            for cb in self.cross_block_list:
                find_cb = False
                for path_voxel, path_id in cb.connection_list:
                    if path_id == father_path_id and not tk3.is_in_list(cb, cross_list):
                        find_cb = True
                        break
                if find_cb:
                    cross_list.append(cb)
                    print('find cross block:' + str(cb))

                    # Cross_block  --find-->  son_path
                    min_angle = self.trunk_angle_threshold
                    min_path_id = -1
                    for son_path_voxel, son_path_id in cb.connection_list:
                        if not tk3.is_in_list(son_path_id, path_list):
                            son_path_direction = self.get_path_direction(son_path_id)
                            angle = tk3.get_angle(father_direction, son_path_direction)
                            print(
                                'Angle between ' + str(father_path_id) + ' and ' + str(son_path_id) + ': ' + str(angle))
                            if angle <= self.trunk_angle_threshold:
                                if angle < min_angle:
                                    min_angle = angle
                                    min_path_id = son_path_id
                    if min_path_id > 0:
                        path_list.append(min_path_id)
                        trunk_iterate(min_path_id, cross_list, path_list)

        # Stage 1: Get the path point with max radius, confirm it as start point, and start path

        radius_graph_path_only = np.multiply(np.where(self.path_graph > 0, 1, 0),
                                             np.where(self.path_graph < self.cross_marker, 1, 0))
        radius_graph_path_only = np.multiply(radius_graph_path_only, self.radius_graph)

        max_radius_voxel = np.unravel_index(radius_graph_path_only.argmax(), self.shape)
        max_radius = self.radius_graph[max_radius_voxel]

        # Stage 2: Search trunk paths and fill [self.trunk_path_list]

        start_path_id = self.path_graph[max_radius_voxel]
        self.branch_path_list.append(start_path_id)
        trunk_iterate(start_path_id, self.branch_cross_list, self.branch_path_list)

        # Stage 3: Generate [self.trunk_path_graph] with [self.trunk_path_list]
        # The trunk path contains paths and cross_blocks between adjacent paths
        self.branch_path_graph = self.path_graph.copy()

        sub_branch_path_list = []
        for trunk_path in self.branch_path_list:
            self.branch_path_graph = np.where(self.branch_path_graph == trunk_path, start_path_id,
                                              self.branch_path_graph)
        for cb in self.branch_cross_list:
            for cross_voxel in cb.cross_voxel_list:
                self.branch_path_graph[cross_voxel] = start_path_id
            for _, path_id in cb.connection_list:
                if not tk3.is_in_list(path_id, self.branch_path_list):
                    sub_branch_path_list.append((path_id, cb))

        for start_sub_path_id, start_cb in sub_branch_path_list:
            sub_cross_list = [start_cb]
            sub_path_list = []
            trunk_iterate(start_sub_path_id, sub_cross_list, sub_path_list)

            for sub_path in sub_path_list:
                self.branch_path_graph = np.where(self.branch_path_graph == sub_path, start_sub_path_id,
                                                  self.branch_path_graph)
            for cb in sub_cross_list:
                for cross_voxel in cb.cross_voxel_list:
                    self.branch_path_graph[cross_voxel] = start_sub_path_id

        self.branch_path_graph = self.purify_path_graph(self.branch_path_graph, keep_cross=False)
        branch_cut_list = []
        for i in range(1, np.max(self.branch_path_graph).astype(np.int) + 1):
            ar = self.get_path_average_radius(self.branch_path_graph, i)
            branch_cut_list.append((i, ar))

        branch_cut_list = sorted(branch_cut_list, key=lambda x: x[1], reverse=True)

        for path_id, avg_radius in branch_cut_list:
            dist_mask = np.where(self.branch_path_graph == path_id, 0, 1)
            dist_mask = ndimage.distance_transform_edt(dist_mask)
            dist_mask = np.where(dist_mask < max_radius * 1, 0, 1)
            self.branch_path_graph = np.multiply(dist_mask, self.branch_path_graph) + \
                                     np.where(self.branch_path_graph == path_id, path_id, 0)

        self.branch_path_graph = self.remove_short_path(self.branch_path_graph)

    def generate_trunk_path_graph(self, ignore_list, contain_list, start_vox=None):

        def trunk_iterate(father_path_id):

            # Calculate father path direction (3D vector, length = 1)
            print('------> father path:' + str(father_path_id))
            father_direction = self.get_path_direction(father_path_id)

            # father_path  --find-->  Cross_block
            for cb in self.cross_block_list:
                find_cb = False
                for path_voxel, path_id in cb.connection_list:
                    if path_id == father_path_id and not tk3.is_in_list(cb, self.trunk_cross_list):
                        find_cb = True
                        break
                if find_cb:
                    self.trunk_cross_list.append(cb)
                    print('find cross block:' + str(cb))

                    # Cross_block  --find-->  son_path
                    min_angle = self.trunk_angle_threshold
                    min_path_id = -1
                    for son_path_voxel, son_path_id in cb.connection_list:
                        if tk3.is_in_list(son_path_id, ignore_path_list):
                            continue
                        if not tk3.is_in_list(son_path_id, self.trunk_path_list):
                            son_path_direction = self.get_path_direction(son_path_id)
                            angle = tk3.get_angle(father_direction, son_path_direction)
                            print(f"Angle between {father_path_id} and {son_path_id}: {angle}")
                            if angle <= self.trunk_angle_threshold:
                                if angle < min_angle:
                                    min_angle = angle
                                    min_path_id = son_path_id
                    if min_path_id > 0:
                        self.trunk_path_list.append(min_path_id)
                        trunk_iterate(min_path_id)

        def get_nearest_path(voxel, path_graph):
            dist_graph = np.ones_like(path_graph)
            dist_graph[voxel] = 0
            dist_graph = ndimage.distance_transform_edt(dist_graph)
            dist_graph = np.multiply(dist_graph,
                                     np.multiply(
                                         np.where(path_graph > 0, 1, 0),
                                         np.where(path_graph < self.cross_marker, 1, 0)
                                     ))
            dist_graph = np.where(dist_graph == 0, 1000, dist_graph)
            min_voxel = np.unravel_index(np.argmin(dist_graph), dist_graph.shape)
            return path_graph[min_voxel]

        ignore_path_list = []
        if ignore_list is not None:
            for ignore_voxel in ignore_list:
                ignore_path_id = get_nearest_path(ignore_voxel, self.path_graph)
                ignore_path_list.append(ignore_path_id)

        contain_path_list = []
        if contain_list is not None:
            for contain_voxel in contain_list:
                contain_path_id = get_nearest_path(contain_voxel, self.path_graph)
                contain_path_list.append(contain_path_id)

        # Stage 1: Get the path point with max radius, confirm it as start point, and start path

        radius_graph_path_only = np.multiply(np.where(self.path_graph > 0, 1, 0),
                                             np.where(self.path_graph < self.cross_marker, 1, 0))
        radius_graph_path_only = np.multiply(radius_graph_path_only, self.radius_graph)

        max_radius_voxel = np.unravel_index(radius_graph_path_only.argmax(), self.shape)
        max_radius = self.radius_graph[max_radius_voxel]

        if start_vox is None:
            start_path_id = self.path_graph[max_radius_voxel]
        else:
            start_path_id = get_nearest_path(start_vox, self.path_graph)
            print(f"start_path_id = {start_path_id}")

        # Stage 2: Search trunk paths and fill [self.trunk_path_list]

        self.trunk_path_list.append(start_path_id)
        self.trunk_path_list.extend(contain_path_list)
        trunk_iterate(start_path_id)

        # Stage 3: Generate [self.trunk_path_graph] with [self.trunk_path_list]
        # The trunk path contains paths and cross_blocks between adjacent paths
        self.trunk_path_graph = self.path_graph.copy()

        for trunk_path in self.trunk_path_list:
            self.trunk_path_graph = np.where(self.trunk_path_graph == trunk_path, start_path_id, self.trunk_path_graph)
        for cb in self.trunk_cross_list:
            for cross_voxel in cb.cross_voxel_list:
                self.trunk_path_graph[cross_voxel] = start_path_id

        dist_mask = np.where(self.trunk_path_graph == start_path_id, 0, 1)
        dist_mask = ndimage.distance_transform_edt(dist_mask)
        dist_mask = np.where(dist_mask < max_radius, 0, 1)
        self.trunk_path_graph = np.multiply(dist_mask, self.trunk_path_graph) + \
                                np.where(self.trunk_path_graph == start_path_id, start_path_id, 0)

        # self.trunk_path_graph = self.remove_short_path(self.trunk_path_graph)

    def process_vein_part(self, cross_position=None):

        self.generate_skele_point_list()

        self.generate_radius_graph()

        self.generate_path_graph()

        self.generate_cross_block_list()

        self.inject_path_id()

        self.generate_ordered_path()

        self.generate_y_path_graph(cross_position)

        if self.y_path_graph is not None:
            self.generate_part_graph_simulator(self.y_path_graph)

    def process_artery_part(self):

        self.generate_skele_point_list()

        self.generate_radius_graph()

        self.generate_path_graph()

        self.generate_cross_block_list()

        self.inject_path_id()

        self.generate_ordered_path()

        self.generate_artery_path_graph()

        self.generate_part_graph_simulator(self.branch_path_graph)

    def process_trunk_part(self, ignore_list=None, contain_list=None):

        self.generate_skele_point_list()

        self.generate_radius_graph()

        self.generate_path_graph()

        self.generate_cross_block_list()

        self.inject_path_id()

        self.generate_ordered_path()

        self.generate_trunk_path_graph(ignore_list=ignore_list, contain_list=contain_list)

        self.generate_part_graph_simulator(self.trunk_path_graph)

    def process_normal_part(self):

        self.generate_skele_point_list()

        self.generate_radius_graph()

        self.generate_path_graph()

        self.generate_cross_block_list()

        self.inject_path_id()

        self.generate_ordered_path()

        self.generate_trunk_path_graph()

        self.generate_part_graph_simulator(self.path_graph)

    def process_thin_analysis(self):
        self.generate_skele_point_list()

        self.generate_radius_graph()

        self.generate_path_graph()

        self.generate_cross_block_list()

        self.inject_path_id()

        self.generate_ordered_path()

        # return self.get_thin_feature()

        return self.get_thin_feature_old()

    def process_contact_length(self, posi_dist_list, part_graph=None, target=None):
        assert self.tumor is not None, 'Skeleton need the tumor image'
        if part_graph is None:
            assert target is not None, 'Contact length function need the target (artery/vein)'
            assert target in ['artery', 'vein'], f'Target "{target}" is not a target'
        self.generate_skele_point_list()
        self.generate_radius_graph()

        contour_vessel = tk3.get_3D_contour(self.vessel, contour_thickness=1.5)
        contact_vessel = tk3.get_3D_contact(self.tumor, contour_vessel, contact_range=2)
        contact_dist_graph = ndimage.distance_transform_edt(np.where(contact_vessel > 0, 0, 1))
        radius_graph_plus = np.where(self.radius_graph > 0, self.radius_graph + 0.3, 0)
        contact_skele_graph = np.where(radius_graph_plus - contact_dist_graph > 0, 1, 0)

        skele_vox_list = tk3.tuple_to_list(np.where(contact_skele_graph > 0))

        result_skele_graph = np.where(self.radius_graph > 0, 1, 0) + contact_skele_graph

        if part_graph is None:
            contact_length = 0
            for skele_vox in skele_vox_list:
                for posi_dist in posi_dist_list:
                    posi, dist = posi_dist
                    adj_vox = (skele_vox[0] + posi[0], skele_vox[1] + posi[1], skele_vox[2] + posi[2])
                    if 0 <= adj_vox[0] < self.shape[0] and 0 <= adj_vox[1] < self.shape[1] and 0 <= adj_vox[2] < \
                            self.shape[2]:
                        if contact_skele_graph[adj_vox] > 0:
                            contact_length += dist
                            print("contact_length + " + str(dist))
                contact_skele_graph[skele_vox] = 0
            return contact_length, result_skele_graph
        else:
            if target == 'artery':
                part_length_dict = {'2': 0, '3': 0, '6': 0}
                part_vox_dict = {'2': [], '3': [], '6': []}
                part_vox_graph_dict = {'2': np.zeros(self.shape), '3': np.zeros(self.shape), '6': np.zeros(self.shape)}
                part_contact_dict = {'2': 0, '3': 0, '6': 0}

            else:
                part_length_dict = {'1': 0, '2': 0}
                part_vox_dict = {'1': [], '2': []}
                part_vox_graph_dict = {'1': np.zeros(self.shape), '2': np.zeros(self.shape)}
                part_contact_dict = {'1': 0, '2': 0}

            for part_id in part_contact_dict.keys():
                part_contact_dict[part_id] = int(np.sum(np.multiply(np.where(contact_vessel == 1, 1, 0),
                                                                    np.where(part_graph == int(part_id), 1, 0)
                                                                    )))

            for skele_vox in skele_vox_list:
                part_id = str(part_graph[skele_vox])
                if part_id in part_vox_dict.keys():
                    part_vox_dict[part_id].append(skele_vox)
                    part_vox_graph_dict[part_id][skele_vox] = 1

            for part_id in part_length_dict.keys():
                for skele_vox in part_vox_dict[part_id]:
                    for posi_dist in posi_dist_list:
                        posi, dist = posi_dist
                        adj_vox = (skele_vox[0] + posi[0], skele_vox[1] + posi[1], skele_vox[2] + posi[2])
                        if 0 <= adj_vox[0] < self.shape[0] and 0 <= adj_vox[1] < self.shape[1] and 0 <= adj_vox[2] < \
                                self.shape[2]:
                            if part_vox_graph_dict[part_id][adj_vox] > 0:
                                part_length_dict[part_id] += dist
                                print(f"\ttarget {target} part {part_id} contact_length += {dist}")
                    part_vox_graph_dict[part_id][skele_vox] = 0
            return part_length_dict, part_contact_dict

    def process_base(self):

        self.generate_skele_point_list()

        self.generate_radius_graph()

        self.generate_path_graph()

        self.generate_cross_block_list()

        self.inject_path_id()

        self.generate_ordered_path()

    def process(self):

        self.generate_skele_point_list()

        self.generate_radius_graph()

        self.generate_path_graph()

        self.generate_cross_block_list()

        self.inject_path_id()

        self.cut_branch()

        print(len(self.skele_point_list))
        print(len(self.skele_point_list_cb))

        # exit(100)

        self.generate_part_graph_normal()
        tk3.save_nii(self.part_graph, 'test.nii.gz', image_info)

        # for skele_point in skele_point_list:
        #     print(skele_point)

        # kernel = morphology.ball(1)
        # img_dilation = morphology.dilation(path_graph, kernel)
        # tk3.save_nii(img_dilation, 'test.nii.gz', image_info)


if __name__ == '__main__':
    # image_dict = tk3.get_nii('data_p2/33_seg.nii.gz')
    # image = image_dict['artery']
    # image_info = image_dict['info']
    # skeleton = Skeleton(image)
    # # skeleton.process_skeleton_analysis()
    # skeleton.process_trunk_part()
    # tk3.save_nii(skeleton.part_graph, "test_part.nii.gz", image_info)

    image_dict = tk3.get_nii('data_p2.2_preprocess/58_pre.nii.gz')
    vessel = image_dict['artery']
    tumor = image_dict['tumor']
    image_info = image_dict['info']

    # skeleton = Skeleton(vessel, tumor)
    # skeleton.process_thin_analysis()

    # skeleton = Skeleton(vessel)
    # skeleton.process_vein_part()
    # tk3.save_nii(skeleton.part_graph * 3 + tumor * 2, "test_part.nii.gz", image_info)

    skeleton = Skeleton(vessel)
    skeleton.process_artery_part()
    tk3.save_nii(skeleton.part_graph * 3 + tumor * 2, "test_part.nii.gz", image_info)
