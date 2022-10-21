import dpkt
import socket

import numpy as np

from utils.utils import get_extension


class Packet:
    def __init__(self):
        self.packet_size = None
        self.timestamp = None
        self.direction = None

    def __str__(self):
        np.set_printoptions(suppress=True)

        msg = "-------------- Packet --------------"
        msg += f"\n Timestamp:               {self.timestamp}"
        msg += f"\n Direction:               {self.direction}"
        msg += f"\n Packet size:             {self.packet_size}"
        msg += "\n\n"

        return msg


class TorPacketParser:
    def __init__(self, entry_node_list=None):
        if entry_node_list is None:
            raise Exception("[ERROR] Invalid entry node list.")

        self.entry_node_list = entry_node_list

    def parse(self, client_ip, file_path):
        tor_packet_list = list()

        packet_list = self._get_packets(file_path)
        for timestamp, bytes_data in packet_list:
            try:
                eth = dpkt.ethernet.Ethernet(bytes_data)
                if not self._is_valid_packet(client_ip, eth):
                    continue

                packet = Packet()
                packet.timestamp = timestamp
                packet.direction = self._get_direction(client_ip, eth)
                packet.packet_size = self._get_packet_size(eth)

                tor_packet_list.append(packet)

            except Exception:
                continue

        if len(tor_packet_list) == 0:
            return None

        return tor_packet_list

    def _is_valid_packet(self, client_ip, eth):
        ip = eth.data
        tcp = ip.data
        packet_size = len(tcp.data)
        source_ip = socket.inet_ntoa(ip.src)
        dst_ip = socket.inet_ntoa(ip.dst)

        if eth.type != dpkt.ethernet.ETH_TYPE_IP:
            return False

        if ip.p != dpkt.ip.IP_PROTO_TCP:
            return False

        if (tcp.sport == 80) or (tcp.dport == 80):
            return False

        if packet_size == 0:
            return False

        if not ((source_ip == client_ip) and (dst_ip in self.entry_node_list) or
                ((source_ip in self.entry_node_list) and (dst_ip == client_ip))):
            return False

        return True

    def _get_direction(self, client_ip, eth):
        ip = eth.data
        source_ip = socket.inet_ntoa(ip.src)
        dst_ip = socket.inet_ntoa(ip.dst)

        direction = -1 if (source_ip == client_ip) and (dst_ip in self.entry_node_list) else 1

        return direction

    @staticmethod
    def _get_packet_size(eth):
        ip = eth.data
        tcp = ip.data

        return len(tcp.data)

    @staticmethod
    def _get_packets(file_path):
        with open(file_path, "rb") as file_descriptor:
            extension = get_extension(file_path)

            if extension == ".pcap":
                packet_list = dpkt.pcap.Reader(file_descriptor)

            elif extension == ".pcapng":
                packet_list = dpkt.pcapng.Reader(file_descriptor)

            else:
                raise Exception("[ERROR] Invalid pcap file type.")

        return packet_list