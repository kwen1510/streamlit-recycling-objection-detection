import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

# For object detection
import torch

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

# Initialise an empty model as a global variable.
model = None


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("Streamlit Recycling Object Detection")
    
    # Run object detection
    app_object_detection()

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_loopback():
    """ Simple video loopback """
    webrtc_streamer(key="loopback")
    

def app_object_detection():

    MODEL_URL = "https://github.com/kwen1510/streamlit-webrtc/raw/main/model/best.pt"
    MODEL_LOCAL_PATH = HERE / "./models/best.pt"

    # Download model file from YOLOv5
    download_file(MODEL_URL, MODEL_LOCAL_PATH)
    
    # Load model file
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_LOCAL_PATH, force_reload=True)

    DEFAULT_CONFIDENCE_THRESHOLD = 0.5

    class Detection(NamedTuple):
        name: str
        prob: float

    class MobileNetSSDVideoProcessor(VideoProcessorBase):
        confidence_threshold: float
        result_queue: "queue.Queue[List[Detection]]"

#         def __init__(self) -> None:
#             self.result_queue = queue.Queue()
            
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            
            image = frame.to_ndarray(format="bgr24")
            
            results = model(image)
            
            model.conf = 0.3
            
            
            
#             image = frame.to_ndarray(format="bgr24")
#             blob = cv2.dnn.blobFromImage(
#                 cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
#             )
#             self._net.setInput(blob)
#             detections = self._net.forward()
#             annotated_image, result = self._annotate_image(image, detections)

            # NOTE: This `recv` method is called in another thread,
            # so it must be thread-safe.
#             self.result_queue.put(result)

            return av.VideoFrame.from_ndarray(np.squeeze(results.render()), format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=MobileNetSSDVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )



if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
