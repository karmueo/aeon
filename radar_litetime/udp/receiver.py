"""UDP 组播接收器。

从 UDP 组播地址接收雷达航迹报文。
"""

import socket


class MulticastReceiver:
    """UDP 组播接收器。

    从指定组播地址和端口接收数据。
    """

    def __init__(
        self,
        group: str,
        port: int,
        iface: str = "0.0.0.0",
        bind_ip: str = "",
        timeout_s: float = 2.0,
    ):
        """初始化接收器。

        Args:
            group: 组播地址。
            port: 组播端口。
            iface: 网卡 IP 地址。
            bind_ip: 绑定 IP，默认为空（绑定到端口）。
            timeout_s: 接收超时秒数。
        """
        self.group = group
        self.port = port
        self.iface = iface
        self.bind_ip = bind_ip
        self.timeout_s = timeout_s
        self.sock = None

    def open(self) -> "MulticastReceiver":
        """打开组播 socket。

        Returns:
            self，支持链式调用。
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if self.bind_ip:
            sock.bind((self.bind_ip, self.port))
        else:
            sock.bind(("", self.port))
        mreq = socket.inet_aton(self.group) + socket.inet_aton(self.iface)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(self.timeout_s)
        self.sock = sock
        return self

    def recv(self, bufsize: int = 1024 * 64) -> tuple[bytes, tuple[str, int]]:
        """接收组播数据。

        Args:
            bufsize: 接收缓冲区大小。

        Returns:
            (data, address) 元组，data 为接收到的数据，address 为发送方地址。
        """
        if self.sock is None:
            raise RuntimeError("socket 未初始化，请先调用 open()")
        return self.sock.recvfrom(bufsize)

    def close(self) -> None:
        """关闭 socket。"""
        if self.sock is not None:
            self.sock.close()
            self.sock = None
