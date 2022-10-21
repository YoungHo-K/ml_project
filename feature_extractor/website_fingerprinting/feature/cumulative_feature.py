"""
Reference: https://github.com/lsvih/CUMUL
"""

import numpy as np


class CumulativeFeature:
    def __init__(self, number_of_features=100):
        self.number_of_features = number_of_features

    def generate(self, packet_list):
        number_of_incoming_packet = 0
        number_of_outgoing_packet = 0
        size_of_total_incoming_packet = 0
        size_of_total_outgoing_packet = 0
        cumulative_packet_size_list = list()
        abs_cumulative_packet_size_list = list()

        packet_list.sort(key=lambda x: x.timestamp)
        for packet in packet_list:
            if packet.direction > 0:
                number_of_incoming_packet += 1
                size_of_total_incoming_packet += packet.packet_size
            else:
                number_of_outgoing_packet += 1
                size_of_total_outgoing_packet += packet.packet_size

            if len(cumulative_packet_size_list) == 0:
                cumulative_packet_size_list.append(packet.packet_size * packet.direction)
                abs_cumulative_packet_size_list.append(packet.packet_size)

            else:
                cumulative_packet_size_list.append(cumulative_packet_size_list[-1] + packet.packet_size * packet.direction)
                abs_cumulative_packet_size_list.append(abs_cumulative_packet_size_list[-1] + packet.packet_size)

        features = list()
        features.append(number_of_incoming_packet)
        features.append(number_of_outgoing_packet)
        features.append(size_of_total_incoming_packet)
        features.append(size_of_total_outgoing_packet)

        features.extend(self._get_interpolated_cumulative_packet_size(cumulative_packet_size_list, abs_cumulative_packet_size_list))

        return features


    def _get_interpolated_cumulative_packet_size(self, cumulative_packet_size_list, abs_cumulative_packet_size_list):
        interpolated_cumulative_packet_size_list = np.interp(np.linspace(abs_cumulative_packet_size_list[0], abs_cumulative_packet_size_list[-1], self.number_of_features),
                                                             abs_cumulative_packet_size_list,
                                                             cumulative_packet_size_list)

        return list(interpolated_cumulative_packet_size_list)