import time

import SimpleITK as sitk
import cc3d
import kimimaro
import numpy as np
from skimage import morphology
import toolkit_3D as tk3


def get_vector_angle(point, point_next):
    x = np.array(point)
    y = np.array(point_next)
    # mod for array
    mod_x = np.sqrt(x.dot(x))
    mod_y = np.sqrt(y.dot(y))
    p_xy = x.dot(y)
    cos_ = p_xy / (mod_x * mod_y + 1e-4)
    angle_hu = np.arccos(cos_)
    angle_d = angle_hu * 180 / np.pi
    return angle_d


class Node:
    """docstring for Nodel"""

    def __init__(self, instance_segmentation):
        """
        Get the base information about shape, instance_segmentation, skeleton and segmentation
        :param instance_segmentation:
        """

        self.x_size, self.y_size, self.z_size = instance_segmentation.shape  # 256,256,90
        self.instance_segmentation = instance_segmentation.copy()

        # num > 0 -> 1, num <= 0 -> 0
        self.segmentation = ((instance_segmentation > 0) / 1.0).astype(np.uint8)

        # self.skeleton = morphology.skeletonize_3d(self.segmentation).astype(np.uint32)

        self.skeleton_0 = kimimaro.skeletonize(self.segmentation,
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
        self.skeleton = np.zeros((self.x_size, self.y_size, self.z_size)).astype(np.uint8)

        for idx in self.skeleton_0.keys():
            for vox_list in self.skeleton_0[idx].vertices: # xxxxxxx
                vox_list = list(vox_list.astype(np.uint8))
                self.skeleton[vox_list[0], vox_list[1], vox_list[2]] = 1

        self.connect_skeleton = self.skeleton.copy()
        self.remove_small_skeleton = self.skeleton.copy()
        self.region = np.zeros((self.x_size, self.y_size, self.z_size)).astype(np.uint8)

        self.connection = np.zeros((self.x_size, self.y_size, self.z_size)).astype(np.uint32)
        self.connection_const = np.zeros((self.x_size, self.y_size, self.z_size)).astype(np.uint32)
        self.connection_region_const = np.zeros((self.x_size, self.y_size, self.z_size)).astype(np.uint32)

        # key_path:
        #   key is point from cluster
        #   value is many list for cross point to near branch, cross and end point path
        # paths is all path for the key cross, branch and end point
        # self.key_path = {}
        # self.key_filter_path = {}
        self.region_path = {}
        self.key_region_path = {}
        self.key_regionmerge_path = {}
        self.cross_branchpoint = {}

        # get keypoint different seg path direction :
        #   key is the key point,
        #   value is path_dict(key -> path end point and value -> mean direction)
        self.points = {}
        self.keypoint_segdirection = {}
        # gt cross point lilst
        self.real_cross_point = []
        self.real_branch_point = []
        self.graph = {}
        self.key_regionmerge_C_path = {}

        self.degree3_cross_point = []
        self.degree3_branch_point = []
        self.degree4_cross_point = []
        self.degree4_branch_point = []
        self.degreebig4_cross_point = []
        self.degreebig4_branch_point = []
        self.pointlist = []

    # Used in compute_graph - 1
    def compute_connection(self):
        """
        Compute the number of a voxel's nearby skeleton voxel
        and save it in the corresponding position in [self.connection]
        :return:
        """

        """
        skeleton and segmentation: (x,y,z)
		calaute point's cross () 
		connection will be changed
		connection_const save the connect num constly
        """
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                for z in range(0, self.z_size):
                    if self.skeleton[x, y, z] != 1:
                        continue
                    # calculate how many neighbor near pix
                    crop = self.skeleton[max(x - 1, 0):min(x + 2, self.x_size), max(y - 1, 0):min(y + 2, self.y_size),
                           max(z - 1, 0):min(z + 2, self.z_size)]
                    sum = np.sum(crop) - 1
                    self.connection[x, y, z] = sum
                    self.connection_const[x, y, z] = sum

    def compute_connection_skeleton(self):
        """
        Compute the number of nearby voxel in the segmented skeleton
        (just like [compute_connection])
        and save in [self.connection_region_const]
        :return:
        """
        binary_region = ((self.region > 0) / 1.0).astype(np.uint8)
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                for z in range(0, self.z_size):
                    if binary_region[x, y, z] != 1:
                        continue
                    crop = binary_region[max(x - 1, 0):min(x + 2, self.x_size), max(y - 1, 0):min(y + 2, self.y_size),
                           max(z - 1, 0):min(z + 2, self.z_size)]
                    sum = np.sum(crop) - 1
                    self.connection_region_const[x, y, z] = sum

    def solve_dfs(self, point, visited, clustered_points, num):
        """ get cross points"""

        # Collect this point
        clustered_points.append(point)

        # Set this point visited
        x_cur, y_cur, z_cur = point
        visited[x_cur, y_cur, z_cur] = 1

        # mark this point in [self.connection] with [num]
        self.connection[x_cur, y_cur, z_cur] = num

        # Iterate all the nearby voxel to collect more cluster point
        for x in range(max(0, x_cur - 1), min(self.x_size, x_cur + 2)):
            for y in range(max(0, y_cur - 1), min(self.y_size, y_cur + 2)):
                for z in range(max(0, z_cur - 1), min(self.z_size, z_cur + 2)):
                    if [x, y, z] == [x_cur, y_cur, z_cur]:
                        continue
                    if self.connection[x, y, z] != 0 and self.connection[x, y, z] != 2 and visited[x, y, z] == 0:
                        self.solve_dfs((x, y, z), visited, clustered_points, num)

    # Used in compute_graph - 2
    def cluster(self):
        """
        Key point for cluster
        Get the cluster points and store them in [self.points] with index > 10000
        :return:
        """
        visited = np.zeros((self.x_size, self.y_size, self.z_size))
        num = 10000

        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                for z in range(0, self.z_size):
                    # key_point

                    # The nearby voxel is not 0 or 2 (not an island or a part of skeleton)
                    # and never visited
                    if self.connection[x, y, z] != 0 and self.connection[x, y, z] != 2 and visited[x, y, z] == 0:
                        # The collection of cluster point
                        clustered_points = []

                        # Collect this cluster point and all the nearby cluster points
                        self.solve_dfs((x, y, z), visited, clustered_points, num)

                        arr = np.array(clustered_points)
                        center = np.mean(arr, axis=0)
                        dist = [np.linalg.norm(np.array(vec) - center) for vec in clustered_points]
                        idx = np.argmin(dist)

                        # points dict is for the cross and branch points
                        self.points[num] = {'coordinate': clustered_points[idx],
                                            'neighbour': set(),
                                            'cluster': clustered_points
                                            }
                        num += 1

    # Used in compute_graph - 3
    def make_connect_skeleton(self):
        """
        Remove the cluster points from skeleton
        :return:
        """
        for x in range(0, self.x_size):
            for y in range(0, self.y_size):
                for z in range(0, self.z_size):
                    # Only cluster points, make them in [self.connect_skeleton] to 0
                    if self.connection[x, y, z] > 10000:
                        self.connect_skeleton[x, y, z] = 0

    # Used in compute_graph - 4
    def make_skeleton_segments(self):
        """
        Mark the segmented skeleton,
        and compute the nearby skeleton voxel of every skeleton voxel
        :return:
        """
        binary = ((self.connect_skeleton > 0) / 1.0).astype(np.uint8)
        # remove small region, mark the connected region
        self.remove_small_skeleton = binary
        self.region = cc3d.connected_components(self.remove_small_skeleton)
        self.compute_connection_skeleton()

    def solve_paths(self, point, used, paths, flag, flag_const, start):
        x_cur, y_cur, z_cur = point
        used[x_cur, y_cur, z_cur] = 1
        paths[-1].append((x_cur, y_cur, z_cur))

        # ==2 connectpoint and ==num same clust->end point
        if self.connection_region_const[x_cur, y_cur, z_cur] == 1 and flag[x_cur, y_cur, z_cur] != start:
            paths.append([])
            return

        for x in range(max(0, x_cur - 1), min(self.x_size, x_cur + 2)):
            for y in range(max(0, y_cur - 1), min(self.y_size, y_cur + 2)):
                for z in range(max(0, z_cur - 1), min(self.z_size, z_cur + 2)):
                    if [x, y, z] == [x_cur, y_cur, z_cur]:
                        continue
                    if used[x, y, z] == 0 and self.connection_region_const[x, y, z] > 0:
                        self.solve_paths((x, y, z), used, paths, flag, flag_const, start)

    # Used in compute_graph - 5
    def make_new_graph_from_segment(self):
        """
        Find region path, and save in [self.region_path]
        :return:
        """

        visited = np.zeros((self.x_size, self.y_size, self.z_size))

        max_label = np.max(self.region)
        for idx in range(1, max_label + 1):
            # Go through all the region labels
            paths = [[]]
            used = np.zeros((self.x_size, self.y_size, self.z_size))

            # Get single region [binary_region]
            binary_region = (self.region == idx).astype(np.uint8)

            # Only keep [self.connection_region_const] in this region
            flag = np.where(binary_region > 0, self.connection_region_const, 0)
            flag_const = flag.copy()

            # End point is where only exist one voxel nearby
            # Mark these end points in [flag] with end_idx + 6
            end_point = np.where(flag_const == 1)
            for end_idx in range(0, len(end_point[0])):
                x0 = end_point[0][end_idx]
                y0 = end_point[1][end_idx]
                z0 = end_point[2][end_idx]
                flag[x0, y0, z0] = end_idx + 6

            for end_idx in range(0, len(end_point[0])):
                x0 = end_point[0][end_idx]
                y0 = end_point[1][end_idx]
                z0 = end_point[2][end_idx]
                start = flag[x0, y0, z0]
                if used[x0, y0, z0] == 0:
                    self.solve_paths((x0, y0, z0), used, paths, flag, flag_const, start)
                    paths.pop(-1)
                    self.region_path[idx] = paths

    # key point connection and path between key point
    def to_nearest_key_pointv3(self, pred_crosspoint):
        crosspoint_arr = np.array(pred_crosspoint)
        if len(self.key_regionmerge_C_path.keys()) != 0:
            flag = True
            dist = [np.linalg.norm(crosspoint_arr - np.array(key_point)) for key_point in
                    self.key_regionmerge_C_path.keys()]
            keypoint_list = [key_point for key_point in self.key_regionmerge_C_path.keys()]
            idx = np.argmin(dist)
            neareset_keypoint = keypoint_list[idx]
            return (dist[idx], neareset_keypoint, flag)
        else:
            flag = False
            return (0, (0, 0, 0), flag)

    def nearest_point_cluster(self, point):
        point_arr = np.array(point)
        min_distance = 1000
        min_point = 0
        for num in self.points.keys():
            for cluster_point in self.points[num]['cluster']:
                dist = np.linalg.norm(point_arr - np.array(cluster_point))
                if dist < min_distance:
                    min_distance = dist
                    min_point = self.points[num]['coordinate']
        return min_point

    def point_cluster(self, point_list):
        arr = np.array(point_list)
        center = np.mean(arr, axis=0)
        print("point_list", point_list)
        flag = True
        dist = [np.linalg.norm(np.array(vec) - center) for vec in point_list]
        if max(dist) < 20:
            print("dist", dist)
            idx = np.argmin(dist)
            center_point = point_list[idx]
        else:
            center_point = center
            flag = False
        return center_point, flag

    # Used in compute_graph - 6
    def make_segment_final_keypoint(self):
        """
        Make the graph with cluster points and region paths
        :return:
        """
        for idx in self.region_path.keys():
            paths = self.region_path[idx]
            for path in paths:
                start_point = path[0]
                end_point = path[-1]
                start_point = self.nearest_point_cluster(start_point)
                end_point = self.nearest_point_cluster(end_point)
                path[0] = start_point
                path[-1] = end_point
                base_path = path.copy()

                if path[0] not in self.key_region_path.keys():
                    list_path = []
                    list_path.append(path)
                    self.key_region_path[path[0]] = list_path
                else:
                    list_path_old = self.key_region_path[path[0]]
                    list_path_old.append(path)
                    self.key_region_path[path[0]] = list_path_old
                # save end point too
                if path[-1] not in self.key_region_path.keys():
                    list_path_reverse = []
                    base_path.reverse()
                    list_path_reverse.append(base_path)
                    self.key_region_path[path[-1]] = list_path_reverse
                else:
                    list_path_reverse_old = self.key_region_path[path[-1]]
                    base_path.reverse()
                    list_path_reverse_old.append(base_path)
                    self.key_region_path[path[-1]] = list_path_reverse_old

    def check_local_direction(self, nodei, nodej, path_ij):
        vector_ij = np.array(path_ij[-1]) - np.array(path_ij[0])
        paths_i = self.key_regionmerge_C_path_0[nodei]
        paths_j = self.key_regionmerge_C_path_0[nodej]
        flag_i = True
        for path_i in paths_i:
            vector = np.array(path_i[-1]) - np.array(path_i[0])
            angle = get_vector_angle(vector_ij, vector)
            if (angle < 85 or angle > 95) and path_i[-1] != nodej:
                flag_i = False

        flag_j = True
        for path_j in paths_j:
            vector = np.array(path_j[-1]) - np.array(path_j[0])
            angle = get_vector_angle(vector_ij, vector)
            if (angle < 85 or angle > 95) and path_j[-1] != nodei:
                flag_j = False
        flag = flag_i and flag_j
        return flag

    # Used in compute_graph - 8
    def refine_graph(self):
        """remove the cirle"""
        self.graph = self.key_region_path.copy()
        count = 1
        while count != 0:
            count = 0
            key_list = []
            for key_point in self.graph.keys():
                paths = self.graph[key_point]
                row = len(paths)
                # delet the degree==1 point
                if row <= 1:
                    key_list.append(key_point)
                    count += 1
            for key_point in key_list:
                self.graph.pop(key_point)
            for key_point in self.graph.keys():
                paths = self.graph[key_point]
                newpath = []
                for path in paths:
                    if path[-1] not in key_list:
                        newpath.append(path)
                self.graph[key_point] = newpath

    def graph_delet(self, graph_circle):
        graph = graph_circle.copy()
        count = 1
        while count != 0:
            count = 0
            key_list = []
            for key_point in graph.keys():
                paths = graph[key_point]
                row = len(paths)
                if row <= 1:
                    key_list.append(key_point)
                    count += 1
            for key_point in key_list:
                graph.pop(key_point)
            for key_point in graph.keys():
                paths = graph[key_point]
                newpath = []
                for path in paths:
                    if path[-1] not in key_list:
                        newpath.append(path)
                graph[key_point] = newpath
        return graph

    def make_graph_list(self):
        graph_remove = self.graph.copy()
        graph_list = []
        while graph_remove != {}:
            graph_new = {}
            pop_obj = graph_remove.popitem()
            key = pop_obj[0]
            for key_point in graph_remove.keys():
                paths = graph_remove[key_point]
                newpath = []
                for path in paths:
                    if path[-1] != key:
                        newpath.append(path)
                graph_remove[key_point] = newpath

            graph_done = self.graph_delet(graph_remove)
            for point in graph_remove.keys():
                if point not in graph_done.keys():
                    graph_new[point] = graph_remove[point]
            graph_new[pop_obj[0]] = pop_obj[1]
            graph_remove = graph_done.copy()
            graph_list.append(graph_new)
        return graph_list

    # Used in compute_graph - 9
    def merge_circle(self):
        self.key_regionmerge_C_path_0 = self.key_region_path.copy()
        # get the circle graph
        graph_list = self.make_graph_list()
        for graph in graph_list:
            newpaths = []
            point_list = []
            for circle_point in graph.keys():
                point_list.append(circle_point)
            if point_list != []:
                center_point, flag = self.point_cluster(point_list)
                if flag:
                    # merge circle
                    for circle_point in graph.keys():
                        paths = self.key_region_path[circle_point]
                        for path in paths:
                            if path[-1] not in graph.keys():
                                newpaths.append(path)
                        self.key_regionmerge_C_path_0.pop(circle_point)
                    self.key_regionmerge_C_path_0[center_point] = newpaths
                    # delet the circle center path
                    for keypoint in self.key_regionmerge_C_path_0.keys():
                        delet_path = []
                        paths = self.key_regionmerge_C_path_0[keypoint]
                        for path in paths:
                            if path[-1] not in graph.keys():
                                delet_path.append(path)
                            else:
                                path[-1] = center_point
                                delet_path.append(path)
                        self.key_regionmerge_C_path_0[keypoint] = delet_path

    # Used in compute_graph - 10
    def merge_close(self, min_length=10):
        """
		Merge the close point
		"""

        for keypoint in self.key_regionmerge_C_path_0.keys():
            x, y, z = keypoint
            if self.connection_const[x, y, z] > 2 or self.connection_const[x, y, z] == 1:
                self.cross_branchpoint[keypoint] = self.key_regionmerge_C_path_0[keypoint]
            # self.key_regionmerge_path[keypoint] = self.key_region_path[keypoint]

        node_visited = set()
        false_link = set()
        # for key_i in self.cross_branchpoint.keys():
        key_newpath_dict = {}
        key_delet_dict = {}
        for key_i in self.key_regionmerge_C_path_0.keys():
            if key_i not in node_visited:
                paths = self.key_regionmerge_C_path_0[key_i]
                node_visited.add(key_i)
                node_deleted = set()
                new_paths = []
                for path in paths:
                    lenght = len(path)
                    endpoint = path[-1]
                    if endpoint in self.key_regionmerge_C_path_0.keys():
                        flag = self.check_local_direction(key_i, endpoint, path)
                        degree = len(paths)
                        flag_m = not flag
                        # deal false link
                        if lenght < min_length and flag:
                            false_link.add(endpoint)
                        elif lenght < min_length and flag_m:
                            node_visited.add(endpoint)
                            node_deleted.add(endpoint)
                        elif lenght > min_length:
                            new_paths.append(path)
                key_delet_dict[key_i] = node_deleted

                for delet_point in node_deleted:
                    if delet_point in self.key_regionmerge_C_path_0.keys():
                        paths_delet = self.key_regionmerge_C_path_0[delet_point]
                        for path_delet in paths_delet:
                            # delet path
                            if path_delet[-1] != key_i:
                                # path_delet[0] = key_i
                                new_delet_path = path_delet.copy()
                                new_delet_path[0] = key_i
                                new_paths.append(new_delet_path)
                        # reverse connect
                        for path_delet_out in paths_delet:
                            if path_delet_out[-1] != key_i \
                                    and path_delet_out[-1] in self.key_regionmerge_C_path_0.keys():
                                # other connect point
                                paths_connect = self.key_regionmerge_C_path_0[path_delet_out[-1]]
                                for path_to_point in paths_connect:
                                    # if path_to_point[-1]==path_delet_out[-1]:
                                    if path_to_point[-1] == delet_point:
                                        path_to_point[-1] = key_i
                                self.key_regionmerge_C_path_0[path_delet_out[-1]] = paths_connect
                if new_paths != []:
                    self.key_regionmerge_C_path[key_i] = new_paths

        # for key_i in self.key_regionmerge_C_path.keys():
        # 	paths = self.key_regionmerge_C_path[key_i]
        # 	deletnode_set = key_delet_dict[key_i]
        # 	if deletnode_set!=():
        # 		for delet_point in deletnode_set:
        # 			paths_delet = self.key_regionmerge_C_path_0[delet_point]
        # 			for path_delet_out in paths_delet:
        # 				if path_delet_out[-1]!=key_i and path_delet_out[-1] in self.key_regionmerge_C_path_0.keys():
        # 					#other connect point
        # 					paths_connect = self.key_regionmerge_C_path_0[path_delet_out[-1]]
        # 					for path_to_point in paths_connect:
        # 						#if path_to_point[-1]==path_delet_out[-1]:
        # 						if path_to_point[-1]==delet_point:
        # 							path_to_point[-1] = key_i
        # 					self.key_regionmerge_C_path[path_delet_out[-1]] = paths_connect

    def compute_graph(self):
        time1 = time.time()
        self.compute_connection()
        time2 = time.time()
        # print("compute_connection : ", time2 - time1)

        self.cluster()
        time3 = time.time()
        # print("cluster : ", time3 - time2)

        self.make_connect_skeleton()
        time4 = time.time()
        # print("make_connect_skeleton : ", time4 - time3)

        self.make_skeleton_segments()
        time5 = time.time()
        # print("make_skeleton_segments : ", time5 - time4)

        self.make_new_graph_from_segment()
        time6 = time.time()
        # print("make_new_graph_from_segment : ", time6 - time5)

        self.make_segment_final_keypoint()
        time7 = time.time()
        # print("make_segment_final_keypoint : ", time7 - time6)

        self.gt_cross()
        time8 = time.time()
        # print("gt_cross : ", time8 - time7)

        self.refine_graph()
        time9 = time.time()
        # print("refine_graph : ", time9 - time8)

        self.merge_circle()
        time10 = time.time()
        # print("merge_circle : ", time10 - time9)

        self.merge_close()
        time11 = time.time()
        # print("merge_close : ", time11 - time10)

        self.sum_result()

        print("Skeletonization completed")

    def get_direction(self, point, point_next):
        x = np.array(point)
        y = np.array(point_next)
        # mod for array
        mod_x = np.sqrt(x.dot(x))
        mod_y = np.sqrt(y.dot(y))
        p_xy = x.dot(y)
        cos_ = p_xy / (mod_x * mod_y + 1e-4)
        angle_hu = np.arccos(cos_)
        angle_d = angle_hu * 180 / np.pi
        return angle_d

    def get_direction_feature(self, point, point_next):
        x = np.array(point)
        y = np.array(point_next)
        vector = y - x
        x_vector = [1, 0, 0]
        y_vector = [0, 1, 0]
        z_vector = [0, 0, 1]
        x_angle = self.get_direction(vector, x_vector)
        y_angle = self.get_direction(vector, y_vector)
        z_angle = self.get_direction(vector, z_vector)
        angle_d = np.array([x_angle, y_angle, z_angle])
        return angle_d

    def get_vector_feature(self, point, point_next):
        x = np.array(point)
        y = np.array(point_next)
        vector = y - x
        return vector

    def keypoint_seg_meandirection(self, keypoint, box_size=20):
        # get the near point
        dist, keypoint, flag = self.to_nearest_key_pointv3(keypoint)
        if flag == True:
            # paths = self.cross_branchpoint[keypoint]
            paths = self.key_regionmerge_C_path[keypoint]
            step_n = 9
            # each step_n pixel  to discribe a direction and get the mean direction for path
            each_seg = {}
            for path in paths:
                end_point = []
                seg_distance = []
                mean_direction = 0
                for point in path[::step_n]:
                    cur_idx = path.index(point)
                    if cur_idx + step_n > len(path) - 1:
                        point_direction = self.get_direction_feature(point, path[len(path) - 1])
                    else:
                        point_direction = self.get_direction_feature(point, path[min(cur_idx + step_n, len(path) - 1)])
                    mean_direction += point_direction
                # local feature
                local_len = 0
                for point_local in path[:min(len(path), box_size)]:
                    mean_direction_local = 0
                    cur_idx = path.index(point_local)
                    local_len += 1
                    if cur_idx + 1 > len(path) - 1:
                        point_direction = self.get_direction_feature(point_local, path[len(path) - 1])
                    else:
                        point_direction = self.get_direction_feature(point_local,
                                                                     path[min(cur_idx + step_n, len(path) - 1)])
                    mean_direction_local += point_direction
                mean_direction_local = mean_direction_local / local_len
                mean_direction = mean_direction / (len(path) - 1)
                seg_distance.append(mean_direction)
                end_point.append(path[-1])
                vector_feature = self.get_vector_feature(path[0], path[-1])
                vector_local_feature = self.get_vector_feature(path[0], path[min(len(path) - 1, box_size)])
                each_seg[path[-1]] = {"seg_direction": mean_direction, "seg_vector": vector_feature,
                                      "seg_local_direction": mean_direction_local,
                                      "seg_local_vector": vector_local_feature, "seg_length": len(path)}
            # self.keypoint_segdirection[path[0]]= {'seg_distance_dict':each_seg}
            self.keypoint_segdirection[keypoint] = {'seg_distance_dict': each_seg}
        else:
            self.keypoint_segdirection = {}

    def get_GT_crosspoint_test(self):
        visited = np.zeros((self.x_size, self.y_size, self.z_size))
        for key_point in self.key_regionmerge_C_path.keys():
            paths = self.key_regionmerge_C_path[key_point]
            if len(paths) > 1:
                for path in paths:
                    length = int(0.5 * len(path))
                    x_s, y_s, z_s = key_point
                    x_e, y_e, z_e = path[length]
                    if (visited[x_s, y_s, z_s] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] != self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_cross = (x_s, y_s, z_s)
                        self.real_cross_point.append(real_cross)
                        visited[x_s, y_s, z_s] = 1
                    elif (visited[x_s, y_s, z_s] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] == self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_branch = (x_s, y_s, z_s)
                        self.real_branch_point.append(real_branch)
                        visited[x_s, y_s, z_s] = 1

    # Used in compute_graph - 7
    def gt_cross(self):
        self.get_GT_crosspoint_test()

    # Used in compute_graph - 11
    def sum_result(self):
        # visited = np.zeros((self.x_size,self.y_size,self.z_size))
        for key_point in self.key_regionmerge_C_path.keys():
            visited = np.zeros((self.x_size, self.y_size, self.z_size))
            paths = self.key_regionmerge_C_path[key_point]
            self.pointlist.append(key_point)
            if len(paths) == 3:
                flag = False
                for path in paths:
                    length = int(0.5 * len(path))
                    x_s, y_s, z_s = key_point
                    x_e, y_e, z_e = path[length]
                    if (visited[x_e, y_e, z_e] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] != self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_cross = (x_s, y_s, z_s)
                        flag = True
                        # self.degree3_cross_point.append(real_cross)
                        visited[x_e, y_e, z_e] = 1
                    elif (visited[x_e, y_e, z_e] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] == self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_branch = (x_s, y_s, z_s)
                        # self.degree3_branch_point.append(real_branch)
                        visited[x_e, y_e, z_e] = 1
                if flag == True:
                    self.degree3_cross_point.append(key_point)
                else:
                    self.degree3_branch_point.append(key_point)
            elif len(paths) == 4:
                flag = False
                for path in paths:
                    length = int(0.5 * len(path))
                    x_s, y_s, z_s = key_point
                    x_e, y_e, z_e = path[length]
                    if (visited[x_e, y_e, z_e] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] != self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_cross = (x_s, y_s, z_s)
                        flag = True
                        # self.degree4_cross_point.append(real_cross)
                        visited[x_e, y_e, z_e] = 1
                    elif (visited[x_e, y_e, z_e] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] == self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_branch = (x_s, y_s, z_s)
                        # self.degree4_branch_point.append(real_branch)
                        visited[x_e, y_e, z_e] = 1
                if flag == True:
                    self.degree3_cross_point.append(key_point)
                else:
                    self.degree3_branch_point.append(key_point)

            elif len(paths) > 4:
                flag = False
                for path in paths:
                    length = int(0.5 * len(path))
                    x_s, y_s, z_s = key_point
                    x_e, y_e, z_e = path[length]
                    if (visited[x_e, y_e, z_e] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] != self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_cross = (x_s, y_s, z_s)
                        flag = True
                        # self.degreebig4_cross_point.append(real_cross)
                        visited[x_e, y_e, z_e] = 1
                    elif (visited[x_e, y_e, z_e] == 0) and (
                            self.instance_segmentation[x_s, y_s, z_s] == self.instance_segmentation[
                        x_e, y_e, z_e]) and (self.instance_segmentation[x_s, y_s, z_s] != 0) and (
                            self.instance_segmentation[x_e, y_e, z_e] != 0):
                        real_branch = (x_s, y_s, z_s)
                        # self.degreebig4_branch_point.append(real_branch)
                        visited[x_e, y_e, z_e] = 1
                if flag == True:
                    self.degree3_cross_point.append(key_point)
                else:
                    self.degree3_branch_point.append(key_point)


def get_vector_direction(point, point_next):
    x = np.array(point)
    y = np.array(point_next)
    # mod for array
    mod_x = np.sqrt(x.dot(x))
    mod_y = np.sqrt(y.dot(y))
    p_xy = x.dot(y)
    cos_ = p_xy / (mod_x * mod_y + 1e-4)
    angle_hu = np.arccos(cos_)
    angle_d = angle_hu * 180 / np.pi
    return angle_d


def get_skeleton_summary(target_only_img, print_info=False):
    def remove_same_path(path_dict_list):
        result_path_list = []
        for idx in range(0, len(path_dict_list)):
            if idx == len(path_dict_list) - 1:
                continue
            if path_dict_list[idx]["flag"] is True:
                for idx1 in range(idx + 1, len(path_dict_list)):
                    if is_same_path(path_dict_list[idx], path_dict_list[idx1]):
                        path_dict_list[idx]["flag"] = False
                        break
        for pd in path_dict_list:
            if pd["flag"] is True:
                path_new_dict = {
                    "id": 0,
                    "length": pd["length"],
                    "start_point": pd["start_point"],
                    "end_point": pd["end_point"],
                    "path": pd["path"],
                    "weight": -1
                }
                result_path_list.append(path_new_dict)

        result_path_list = sorted(result_path_list, key=lambda x: x["length"])

        id = 1
        for pd in result_path_list:
            pd["id"] = id
            id += 1

        return result_path_list

    def is_same_path(path_dict_1, path_dict_2):
        if path_dict_1["length"] != path_dict_2["length"]:
            return False

        path1 = path_dict_1["path"]
        path2 = path_dict_2["path"]
        length = path_dict_1["length"]

        sum = 0
        for p in path1:
            if p in path2:
                sum += 1
        if sum == length:
            return True

    image_graph = Node(target_only_img)
    image_graph.compute_graph()

    path_dict_list = []
    point_list = []

    for point in image_graph.key_regionmerge_C_path.keys():
        paths = image_graph.key_regionmerge_C_path[point]
        point_list.append(point)
        for path in paths:
            path_dict = {
                "length": len(path),
                "start_point": path[0],
                "end_point": path[-1],
                "path": path,
                "flag": True
            }
            path_dict_list.append(path_dict)

    path_list = remove_same_path(path_dict_list)

    if print_info:
        print("paths: ")
        for path in path_list:
            print("path " + str(path["id"]) + ": ", end="")
            print(path["path"])


    return point_list, path_list


if __name__ == '__main__':

    image_dict = tk3.get_nii('data_p2/33_seg.nii.gz')
    image = image_dict['artery']
    image_info = image_dict['info']

    skele_image = np.zeros(image.shape)

    print("input labels shape : " + str(image.shape))
    image_graph = Node(image)

    for idx in image_graph.skeleton_0.keys():
        print('idx: ' + str(idx))
        for point in image_graph.skeleton_0[idx].vertices:
            skele_image[int(point[0]), int(point[1]), int(point[2])] = 1
        print("vertices:")
        print(len(image_graph.skeleton_0[idx].vertices))
        print(image_graph.skeleton_0[idx].vertices)
        print("radius:")
        print(len(image_graph.skeleton_0[idx].radius))
        print(image_graph.skeleton_0[idx].radius)
        for vox_list in image_graph.skeleton_0[idx].vertices:  # xxxxxxx
            vox_list = list(vox_list.astype(np.uint8))
            image_graph.skeleton[vox_list[0], vox_list[1], vox_list[2]] = 1

    kernel = morphology.ball(1)
    img_dilation = morphology.dilation(skele_image, kernel)
    tk3.save_nii(img_dilation, 'test.nii.gz', image_info)

    # image_graph.compute_graph()
    #
    # segmentation = image_graph.instance_segmentation.copy()
    #
    # # A sphere-shaped region to dilate the path
    # kernel = morphology.ball(1)
    #
    # # counter to make different path color
    # segments_num = 1
    #
    # # The result image to contain paths and points
    # image_instance = np.zeros(segmentation.shape)
    #
    # # key_regionmerge_C_path - list of paths
    # for point in image_graph.key_regionmerge_C_path.keys():
    #     paths = image_graph.key_regionmerge_C_path[point]
    #
    #     # Mark points in a path by 1
    #     for path in paths:
    #         image = np.zeros(segmentation.shape)
    #         for p in path:
    #             x, y, z = p
    #             image[x, y, z] = 1
    #
    #         # make paths dilation by 1, to make them visible
    #         img_dilation = morphology.dilation(image, kernel)
    #
    #         # color different paths and inject into [image_instance]
    #         image_instance[img_dilation > 0] = img_dilation[img_dilation > 0] * segments_num
    #         segments_num += 1
    #
    # # point - the location of the connection of different regions
    # for point in image_graph.key_regionmerge_C_path.keys():
    #     x, y, z = point
    #     paths = image_graph.key_regionmerge_C_path[point]
    #     image_instance[
    #     max(x - 3, 0): min(x + 3, image_instance.shape[0]),
    #     max(y - 3, 0): min(y + 3, image_instance.shape[1]),
    #     max(z - 3, 0): min(z + 3, image_instance.shape[2])
    #     ] = 66
    #
    # # write_image_to_niigz(image_instance,"./ME195840_x13824_14080y6912_7168z4140_4230_instance_map_skele_point.nii.gz")
    # print("image_instance shape : " + str(image_instance.shape))
    #
    # # new_nii = sitk.GetImageFromArray(image_instance.transpose(2, 1, 0).astype(np.uint8))
    #
    # tk3.save_nii(image_instance, "test.nii.gz", image_info)
    #
    # print("out labels shape : " + str(image_instance.shape))

    '''
	for point in image_graph.key_regionmerge_C_path.keys():
		if len(image_graph.key_regionmerge_C_path[point])!=0:
			x,y,z = point
			segmentation3[max(x-3,0):min(x+3,segmentation3.shape[0]),max(y-3,0):min(y+3,segmentation3.shape[1]),max(z-3,0):min(z+3,segmentation3.shape[2])]=66
	write_image_to_niigz(segmentation3,"./ME195840_x13824_14080y6912_7168z4140_4230_instance_map_graph_points.nii.gz")
	#print("image_graph.key_regionmerge_C_path",image_graph.key_regionmerge_C_path)
	np.save('./ME195840_x13824_14080y6912_7168z4140_4230_instance_map_keypoint_paths.npy', image_graph.key_regionmerge_C_path)
	read_dictionary = np.load('./ME195840_x13824_14080y6912_7168z4140_4230_instance_map_keypoint_paths.npy', allow_pickle=True).item()
	print("image_graph.key_regionmerge_C_path",read_dictionary)
	for key in read_dictionary.keys():
		paths = read_dictionary[key]
		print("keypoint",key)
		for path in paths:
			print("endpoint:",path[-1])

	kernel = morphology.ball(1)
	img_dialtion = morphology.dilation(image_graph.skeleton, kernel)
	write_image_to_niigz(img_dialtion,"./ME195840_x13824_14080y6912_7168z4140_4230_instance_map_skeleton.nii.gz")
	seg_region = morphology.dilation(image_graph.remove_small_skeleton, kernel)
	write_image_to_niigz(seg_region,"./ME195840_x13824_14080y6912_7168z4140_4230_instance_map_skeleton_rm.nii.gz")	
	'''
