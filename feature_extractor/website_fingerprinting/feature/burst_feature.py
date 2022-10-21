
from utils.statistics import get_max_in_list, get_mean_in_list


class BurstFeature:
    def generate(self, packet_list):
        incoming_burst_list, outgoing_burst_list = self._get_burst(packet_list)
        if (len(incoming_burst_list) == 0) or (len(outgoing_burst_list) == 0):
            return None

        features = list()
        features.append(get_max_in_list(incoming_burst_list))
        features.append(get_max_in_list(outgoing_burst_list))
        features.append(get_mean_in_list(incoming_burst_list))
        features.append(get_mean_in_list(outgoing_burst_list))
        features.append(len(incoming_burst_list))
        features.append(len(outgoing_burst_list))
        features.append(len(list(filter(lambda x: x > 5, incoming_burst_list))))
        features.append(len(list(filter(lambda x: x > 5, outgoing_burst_list))))
        features.append(len(list(filter(lambda x: x > 10, incoming_burst_list))))
        features.append(len(list(filter(lambda x: x > 10, outgoing_burst_list))))
        features.append(len(list(filter(lambda x: x > 15, outgoing_burst_list))))
        features.append(len(list(filter(lambda x: x > 15, outgoing_burst_list))))

        return features

    @staticmethod
    def _get_burst(packet_list):
        incoming_burst_list = list()
        outgoing_burst_list = list()
        number_of_incoming_packet = 0
        number_of_outgoing_packet = 0

        packet_list.sort(key=lambda x: x.timestamp)
        for packet in packet_list:
            if packet.direction > 0:
                if number_of_outgoing_packet != 0:
                    outgoing_burst_list.append(number_of_outgoing_packet)

                    number_of_outgoing_packet = 0

                number_of_incoming_packet += 1

                continue

            if number_of_incoming_packet != 0:
                incoming_burst_list.append(number_of_incoming_packet)

                number_of_incoming_packet = 0

            number_of_outgoing_packet += 1

        incoming_burst_list.append(number_of_incoming_packet) if number_of_incoming_packet != 0 else None
        outgoing_burst_list.append(number_of_outgoing_packet) if number_of_outgoing_packet != 0 else None

        return incoming_burst_list, outgoing_burst_list