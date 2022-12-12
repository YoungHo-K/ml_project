import numpy as np

from utils.statistics import get_statistics


class BurstTimeIntervalFeature:
    def generate(self, packet_list):
        incoming_burst_time_interval_list, outgoing_burst_time_interval_list = self._get_burst_time_intervals(packet_list)
        if len(incoming_burst_time_interval_list) == 0:
            incoming_burst_time_interval_list = [-1]

        if len(outgoing_burst_time_interval_list) == 0:
            incoming_burst_time_interval_list = [-1]

        if (len(incoming_burst_time_interval_list) == 0) or (len(outgoing_burst_time_interval_list) == 0):
            return None

        features = list()
        features.extend(get_statistics(incoming_burst_time_interval_list))
        features.extend(get_statistics(outgoing_burst_time_interval_list))

        return features

    @staticmethod
    def _get_burst_time_intervals(packet_list):
        incoming_burst_list = list()
        outgoing_burst_list = list()
        incoming_burst = list()
        outgoing_burst = list()

        packet_list.sort(key=lambda x: x.timestamp)
        for index, packet in enumerate(packet_list):
            if packet.direction > 0:
                if len(outgoing_burst) > 0:
                    outgoing_burst_list.append(outgoing_burst)
                    outgoing_burst = list()

                incoming_burst.append(packet.timestamp)

                continue

            if len(incoming_burst) > 0:
                incoming_burst_list.append(incoming_burst)
                incoming_burst = list()

            outgoing_burst.append(packet.timestamp)

        incoming_burst_list.append(incoming_burst) if len(incoming_burst) != 0 else None
        outgoing_burst_list.append(outgoing_burst) if len(outgoing_burst) != 0 else None

        incoming_burst_time_interval_list = np.diff(list(map(lambda x: x[0], incoming_burst_list)))
        outgoing_burst_time_interval_list = np.diff(list(map(lambda x: x[0], outgoing_burst_list)))

        return incoming_burst_time_interval_list, outgoing_burst_time_interval_list


