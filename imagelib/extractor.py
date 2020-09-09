"""
Image lib
@Author Chanwoo Kwon, Yonsei Univ. Researcher, 2020.05~
"""

import cv2
import numpy as np
import math


class Extractor:
    """
    vector extractor
    extract vector value from graph image
    """

    def extract(self, image_path, is_left: bool):
        """
        extract graph only (external interface)
        :param image_path: graph image's path
        :param is_left: is left ear's graph
        :return: graph list from image
        """
        image = cv2.imread(image_path)
        return self.extract_image(image, is_left)

    def extract_image(self, image, is_left: bool):
        """
        extract graph only (external interface)
        :param image: graph image's object(opencv)
        :param is_left: is left ear's graph
        :return: graph list from image
        """
        crop = self.__crop_by_axis(image)
        if is_left:
            res = self.__graph_left(crop)[0]
        else:
            res = self.__graph_right(crop)[0]

        return sorted(res, key=lambda v: v[0], reverse=True)

    def extract_image_with_peak(self, image, is_left: bool):
        """
        extract graph and peak value from image (external interface)
        :param image: image object (opencv)
        :param is_left:  is left ear's graph
        :return: graph and peak list from image
        """
        return self.__get_vector_and_peak(image, is_left)

    def extract_with_peak(self, image_path, is_left: bool):
        """
        extract graph and peak value from image (external interface)
        :param image_path: graph image's path
        :param is_left: is left ear's graph
        :return: graph and peak list from image
        """
        image_mat = cv2.imread(image_path)
        return self.extract_image_with_peak(image_mat, is_left)

    def __crop_by_axis(self, img):
        """
        detect x, y axis and crop the image
        :param img: image matrix
        :return: crop image matrix
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        row = cv2.reduce(gray, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32F)
        column = cv2.reduce(gray, 1, cv2.REDUCE_SUM, dtype=cv2.CV_32F)

        x = np.argmax(row)  # start point x-axis
        y = np.argmax(column)  # start point y-axis

        crop = img[:y, x + 1:]

        return crop

    def __get_vector_and_peak(self, img, is_left: bool):
        """
        extract graph and peak value from image (internal method)
        :param img: image matrix read by opencv lib
        :param is_left: is left ear's graph
        :return: image y-values vector
        """
        # Read the image and create a blank mask

        crop = self.__crop_by_axis(img)

        graph_list, end_of_x, draw_graph, mask = self.__graph_left(crop) if is_left else self.__graph_right(crop)

        peak_candidate = self.__peak(crop, draw_graph, mask)
        peak_list = []

        for candidate in peak_candidate:
            x, y = self.__peak_extractor(candidate, crop.shape[0])
            if end_of_x - 1 <= x <= end_of_x + 1:
                continue
            peak_list.append((x, y))

        result = []
        for line in graph_list:
            peak_point_list = []
            for x, y in peak_list:
                y2 = line[x]
                # find peak in graph
                distance = Extractor.__calculate_distance(x, y, x, y2)
                if distance < 10:
                    peak_point_list.append((x, y))

            result.append({"graph": line, "peak": sorted(peak_point_list, key=lambda v: v[0])})

        return sorted(result, key=lambda v: v["graph"][0], reverse=True)

    def __graph(self, crop, lower_color, upper_color):
        """
        detect graph from image
        :param crop: cropped image
        :param lower_color: lower value of color range
        :param upper_color: upper value of color range
        :return: graph list, end of x-value from graphs, image drown graph's contours
        """
        # detect range of color
        mask = cv2.inRange(crop, lower_color, upper_color)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        merge = crop & mask_rgb  # extract
        gray = cv2.cvtColor(merge, cv2.COLOR_BGR2GRAY)

        # contours (means vector of graph's values)
        graph_lines, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        blank1 = np.zeros(crop.shape, np.uint8)
        cv2.drawContours(blank1, graph_lines, -1, (255, 255, 255))
        blank1 = cv2.dilate(blank1, np.ones((3, 3), np.uint8), iterations=1)

        graph_lines, _ = cv2.findContours(cv2.cvtColor(blank1, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE,
                                          cv2.CHAIN_APPROX_NONE)  # find graph's values

        graph_list = []
        for graph_line in graph_lines:
            # extract graph line from graph contour
            graph_list.append(self.__line_extractor(graph_line, crop.shape[1], crop.shape[0]))

        # check end of x-value from graphs
        end_of_x = 0
        for graph_line in graph_list:
            one_of_end = graph_line["st_end"]["end"]
            if end_of_x == 0 or end_of_x < one_of_end:
                end_of_x = one_of_end

        # reconnect graph when they're broken
        graph_list = self.__reconnect_line(graph_list, end_of_x)
        blank1 = np.zeros(crop.shape, np.uint8)

        for graph in graph_list:
            for i in range(len(graph)):
                y = crop.shape[0] - int(graph[i]) - 1
                x = i

                blank1[y, x] = [255, 255, 255]
                if y > 1:
                    blank1[y - 1, x] = [255, 255, 255]

                if y < crop.shape[0] - 1:
                    blank1[y + 1, x] = [255, 255, 255]

                if x > 0:
                    blank1[y, x - 1] = [255, 255, 255]

                if x < crop.shape[1] - 1:
                    blank1[y, x + 1] = [255, 255, 255]

        return graph_list, end_of_x, blank1, mask_rgb

    def __graph_left(self, crop):
        """
        extract graph when it's left ear's
        :param crop: cropped image
        :return: same to __graph
        """
        lower_color = (200, 0, 0)
        upper_color = (255, 100, 100)

        return self.__graph(crop, lower_color, upper_color)

    def __graph_right(self, crop):
        """
        extract graph when it's right ear's
        :param crop: cropped image
        :return: same to __graph
        """
        lower_color = (0, 0, 200)
        upper_color = (10, 0, 255)

        return self.__graph(crop, lower_color, upper_color)

    def __peak(self, crop, draw_graph, mask):
        """
        extract peak from image
        :param crop: cropped image
        :param draw_graph: image drown graph's contour
        :param mask: first mask of graph image
        :return: peak candidates
        """
        # do threshold and reverse image
        _, threshold = cv2.threshold(crop, 80, 255, cv2.THRESH_BINARY_INV)
        threshold = threshold & (255 - mask)
        black = threshold[:, :, 0]

        # extend shape of image to remove character in image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grad = cv2.morphologyEx(black, cv2.MORPH_GRADIENT, kernel)
        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        line_candidates, hierarchy = cv2.findContours(connected, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        blank2 = np.zeros(crop.shape, np.uint8)

        for i in range(len(line_candidates)):
            cnt = line_candidates[i]
            area = cv2.contourArea(cnt)
            if 0 < area <= 22:  # remove character candidate (peak's marks are same size and most occurred)
                cv2.drawContours(blank2, line_candidates, i, (0, 255, 0))
        intersection = draw_graph & blank2  # peak's mark is overlap with graph line

        inter_black = cv2.cvtColor(intersection, cv2.COLOR_BGR2GRAY)
        inter_black = cv2.dilate(inter_black, np.ones((5, 5), np.uint8), iterations=1)

        peak_candidate, _ = cv2.findContours(inter_black, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return peak_candidate

    def __line_extractor(self, graph_line, x_limit, y_limit):
        """
        extract line of graph from graph's contour
        :param graph_line: graph's contour
        :param x_limit: end of x-value in image for initialize an array
        :param y_limit: end of y-value in image for reverse to y-value
        :return: list of real value in graph
        """
        # get all y-values of x-values
        value_of_contour_list = [[] for x in range(x_limit)]
        for array_item in graph_line:
            for array_item_2 in array_item:
                x = array_item_2[0]
                y = array_item_2[1]
                value_of_contour_list[x].append(y)

        # calculate real y-value of x-values
        real_value_list = []
        for i in range(len(value_of_contour_list)):
            item = value_of_contour_list[i]
            if len(item) != 0:
                real_value_list.append(y_limit - sum(item) / len(item))
            else:
                real_value_list.append(0)

        return {"value": real_value_list, "st_end": self.__find_start_end_in_line(real_value_list)}

    @staticmethod
    def __calculate_distance(x1, y1, x2, y2):
        """
        calculate distance between two points
        :param x1: point1's x-value
        :param y1: point1's y-value
        :param x2: point2's x-value
        :param y2: point2's y-value
        :return: distance between two points
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def __reconnect_line(self, extracted_lines, end_of_x):
        """
        reconnect lines when unconnected
        :param extracted_lines: real lines of graph
        :param end_of_x: end of x-value
        :return: list of reconnected line
        """
        new_graph_list = []
        need_reconnect = []

        # check the line need to connect
        for graph_line in extracted_lines:
            start = graph_line["st_end"]["start"]
            end = graph_line["st_end"]["end"]
            if start != 0 or end != end_of_x:
                need_reconnect.append({"graph": graph_line, "already_connected": False})
            else:
                new_graph_list.append(graph_line["value"][:end_of_x])

        # save information of reconnection
        chain = []
        for i in range(len(need_reconnect)):
            need_graph_line = need_reconnect[i]
            already_connected = need_graph_line["already_connected"]
            if already_connected is True:
                continue

            graph_line = need_graph_line["graph"]
            end = graph_line["st_end"]["end"]
            graph = graph_line["value"]

            if end != 0:
                for j in range(len(need_reconnect)):
                    need_graph_line_2 = need_reconnect[j]
                    already_connected_2 = need_graph_line_2["already_connected"]
                    if already_connected_2 is True:
                        continue

                    graph_line_2 = need_graph_line_2["graph"]
                    one_of_start = graph_line_2["st_end"]["start"]
                    graph_2 = graph_line_2["value"]
                    distance = Extractor.__calculate_distance(end, graph[end], one_of_start, graph_2[one_of_start])

                    if distance < 20:
                        # need_graph_line_2["already_connected"] = True
                        chain.append((i, j))

        # reconnect
        def combine(first_idx, second_idx: int):
            if need_reconnect[first] == -1 or need_reconnect[second] == -1:
                return

            first_graph = need_reconnect[first_idx]["graph"]["value"]
            second_graph = need_reconnect[second_idx]["graph"]["value"]
            first_end = need_reconnect[first_idx]["graph"]["st_end"]["end"]
            second_st = need_reconnect[second_idx]["graph"]["st_end"]["start"]
            second_end = need_reconnect[second_idx]["graph"]["st_end"]["end"]
            first_graph[second_st:second_end] = second_graph[second_st:second_end]
            first_graph[first_end:second_st] = [first_graph[first_end] for x in range(first_end, second_st)]

            need_reconnect[first_idx]["graph"]["st_end"]["end"] = second_end

            need_reconnect[second_idx] = -1

        for i in range(len(chain)):
            first, second = chain[i]
            for j in range(i, len(chain)):
                x, y = chain[j]

                if x == second:
                    combine(x, y)

            combine(first, second)

        # remove rest
        for line in need_reconnect:
            if line != -1:
                new_graph_list.append(line["graph"]["value"][:end_of_x])

        return new_graph_list

    def __find_start_end_in_line(self, extracted_graph_line):
        """
        detect start end end points in graph
        :param extracted_graph_line: real line of graph
        :return: start point's x-value, end point's x-value
        """
        st = 0
        end = len(extracted_graph_line) - 1

        for i in range(len(extracted_graph_line)):
            if extracted_graph_line[i] != 0:
                st = i
                break

        for j in range(len(extracted_graph_line) - 1, -1, -1):
            if extracted_graph_line[j] != 0:
                end = j
                break

        return {"start": st, "end": end}

    def __peak_extractor(self, peak_point, y_limit):
        """
        extract peak point from peak's contour
        :param peak_point:
        :param y_limit:
        :return: peak point
        """
        x_list = []
        values = {}
        for array_item in peak_point:
            for array_item_2 in array_item:
                x = array_item_2[0]
                y = y_limit - array_item_2[1]
                if x not in x_list:
                    x_list.append(x)
                    if x not in values or values[x] > y:
                        values[x] = y

        if len(x_list) == 0:
            return -1

        x = int(sum(x_list) / len(x_list))
        y = values[x]

        return x, y
