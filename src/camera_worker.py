# -*- coding: utf-8 -*-
"""
camera_worker.py — Этап 3. Поток-читатель одной камеры.

Каждая камера = отдельный поток. Задача потока:
  - открыть RTSP (FFmpeg/TCP), читать кадры с frame-skip (~target_fps);
  - класть кадры в ОБЩУЮ очередь (не блокируясь: при переполнении дропать);
  - при обрыве — переподключаться с экспоненциальным backoff, НЕ роняя остальные камеры;
  - вести статистику: FPS, задержка, дропы, число реконнектов.

Сами кадры НЕ обрабатываются здесь — детекция/распознавание в одном
inference-потоке (см. inference_worker.py), который дёргает общий GPU-движок.
"""
import os
import time
import queue
import socket
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse

import cv2

# RTSP через FFmpeg: TCP + таймауты (ставить до VideoCapture).
# timeout (мкс) — таймаут СОКЕТА, включая connect: критично, иначе мёртвая камера
# подвешивает open() и держит глобальный мьютекс FFmpeg, блокируя другие камеры.
# stimeout — устаревший алиас для совместимости.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|timeout;5000000|stimeout;5000000|max_delay;5000000|buffer_size;1024000",
)


@dataclass
class FrameItem:
    """Единица в очереди на распознавание."""
    cam_id: str
    zone: str
    frame: "cv2.typing.MatLike"
    capture_ts: float          # время захвата (для сквозной задержки)


@dataclass
class CamStats:
    """Живая статистика по камере (читается потоком статистики)."""
    cam_id: str
    zone: str
    connected: bool = False
    frames_read: int = 0
    frames_enqueued: int = 0
    drops: int = 0
    reconnects: int = 0
    fps_ema: float = 0.0
    last_error: str = ""
    _t_prev: float = field(default=0.0, repr=False)

    def note_processed(self, now: float):
        if self._t_prev:
            inst = 1.0 / max(1e-6, now - self._t_prev)
            self.fps_ema = inst if self.fps_ema == 0 else 0.8 * self.fps_ema + 0.2 * inst
        self._t_prev = now


def _is_network_source(src) -> bool:
    """RTSP/HTTP-источник (а не файл/вебка)?"""
    return isinstance(src, str) and "://" in src


def tcp_reachable(url: str, timeout: float = 3.0) -> bool:
    """
    Быстрая проверка доступности камеры по TCP СВОИМИ средствами (свой таймаут).
    Критично: недостижимый RTSP, поданный сразу в cv2.VideoCapture, подвешивает
    open() на десятки секунд и держит глобальный мьютекс FFmpeg в OpenCV,
    блокируя открытие ДРУГИХ камер. Поэтому сначала пробуем подключиться сами.
    """
    try:
        p = urlparse(url)
        host = p.hostname
        if not host:
            return True                      # не распарсили — пусть пробует cv2
        port = p.port or (554 if p.scheme.startswith("rtsp") else 80)
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _open_capture(source: str) -> cv2.VideoCapture:
    if str(source).isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap


class CameraWorker(threading.Thread):
    def __init__(self, cam: dict, frame_queue: "queue.Queue[FrameItem]",
                 settings: dict, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"cam-{cam['id']}")
        self.cam = cam
        self.q = frame_queue
        self.stop_event = stop_event
        self.target_fps = settings["recognition"]["target_fps"]
        self.stats = CamStats(cam_id=cam["id"], zone=cam.get("zone", ""))

    # ----- обработка одного открытого источника -----
    def _read_loop(self, cap: cv2.VideoCapture) -> str:
        """Читать кадры пока поток жив. Возвращает причину выхода ('eof'|'disconnect'|'stop')."""
        is_file = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) > 0
        min_interval = 1.0 / max(0.1, self.target_fps)
        last_proc = 0.0
        fail = 0

        while not self.stop_event.is_set():
            if not cap.grab():
                if is_file:
                    # файл закончился -> зациклить (удобно для теста «как поток»)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                fail += 1
                if fail > 50:
                    return "disconnect"
                time.sleep(0.02)
                continue
            fail = 0
            self.stats.frames_read += 1

            now = time.time()
            if now - last_proc < min_interval:
                continue                     # frame skip
            last_proc = now

            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            item = FrameItem(self.cam["id"], self.cam.get("zone", ""), frame, now)
            try:
                self.q.put_nowait(item)
                self.stats.frames_enqueued += 1
            except queue.Full:
                # очередь забита (GPU не успевает) — дропаем самый старый, кладём свежий
                try:
                    self.q.get_nowait()
                    self.q.put_nowait(item)
                except queue.Empty:
                    pass
                self.stats.drops += 1
            self.stats.note_processed(now)

        return "stop"

    def run(self):
        backoff = 1.0
        backoff_max = 30.0
        src = self.cam["rtsp"]
        while not self.stop_event.is_set():
            # Пре-флайт: для сетевых камер проверяем доступность сами, чтобы
            # зависший connect не блокировал другие камеры через мьютекс OpenCV.
            if _is_network_source(src) and not tcp_reachable(src, timeout=3.0):
                self.stats.connected = False
                self.stats.last_error = "unreachable"
                self.stats.reconnects += 1
                self._sleep_backoff(backoff)
                backoff = min(backoff_max, backoff * 2)
                continue

            cap = _open_capture(src)
            if not cap.isOpened():
                self.stats.connected = False
                self.stats.last_error = "open failed"
                self.stats.reconnects += 1
                self._sleep_backoff(backoff)
                backoff = min(backoff_max, backoff * 2)
                continue

            self.stats.connected = True
            self.stats.last_error = ""
            backoff = 1.0                      # успешное подключение -> сброс backoff

            reason = self._read_loop(cap)
            cap.release()

            if reason == "stop":
                break
            # disconnect -> реконнект с backoff
            self.stats.connected = False
            self.stats.last_error = "disconnect"
            self.stats.reconnects += 1
            self._sleep_backoff(backoff)
            backoff = min(backoff_max, backoff * 2)

    def _sleep_backoff(self, seconds: float):
        """Спать с backoff, но прерываемо по stop_event."""
        end = time.time() + seconds
        while time.time() < end and not self.stop_event.is_set():
            time.sleep(0.1)
