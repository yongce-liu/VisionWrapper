import time
from vuer import Vuer
from vuer.schemas import ImageBackground, Hands
from multiprocessing import Array, Process, shared_memory
import numpy as np
import asyncio
import cv2

from multiprocessing import context

Value = context._default_context.Value


class TeleVision:
    def __init__(
        self,
        binocular: bool,
        img_shape: tuple,
        img_shm_name: str,
        cert_file: str,
        key_file: str,
        ngrok: bool,
    ):
        self.binocular = binocular

        if ngrok:
            self.vuer = Vuer(host="0.0.0.0", queries=dict(grid=False), queue_len=3)
        else:
            self.vuer = Vuer(
                host="0.0.0.0",
                cert=cert_file,
                key=key_file,
                queries=dict(grid=False),
                queue_len=3,
            )

        self.vuer.add_handler("HAND_MOVE")(self.on_hand_move)
        self.vuer.add_handler("CAMERA_MOVE")(self.on_cam_move)

        if not img_shape is None:
            self.img_height = img_shape[0]
            if binocular:
                self.img_width = img_shape[1] // 2
            else:
                self.img_width = img_shape[1]

            existing_shm = shared_memory.SharedMemory(name=img_shm_name)
            self.img_array = np.ndarray(
                img_shape, dtype=np.uint8, buffer=existing_shm.buf
            )

            if binocular:
                self.vuer.spawn(start=False)(self.main_image_binocular)
            else:
                self.vuer.spawn(start=False)(self.main_image_monocular)

        self.left_hand_shared = Array("d", 16, lock=True)
        self.right_hand_shared = Array("d", 16, lock=True)
        self.left_landmarks_shared = Array("d", 75, lock=True)
        self.right_landmarks_shared = Array("d", 75, lock=True)

        self.head_matrix_shared = Array("d", 16, lock=True)
        self.aspect_shared = Value("d", 1.0, lock=True)

        self.process = Process(target=self.vuer_run)
        self.process.daemon = True
        self.process.start()

    def vuer_run(self):
        self.vuer.run()

    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value["camera"]["aspect"]
        except:
            pass

    async def on_hand_move(self, event, session, fps=60):
        try:
            left_hand_data = event.value["left"]
            right_hand_data = event.value["right"]
            # left_hand_state = event.value["leftState"]
            # right_hand_state = event.value["rightState"]

            self.extract_hand_poses(
                left_hand_data, self.left_hand_shared, self.left_landmarks_shared
            )
            self.extract_hand_poses(
                right_hand_data, self.right_hand_shared, self.right_landmarks_shared
            )
            # extract_hand_states(left_hand_state, "left")
            # extract_hand_states(right_hand_state, "right")

        except:
            pass

    async def main_image_binocular(self, session, fps=60):
        session.upsert @ Hands(
            fps=fps, stream=True, key="hands", showLeft=False, showRight=False
        )
        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image[:, : self.img_width],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        # The underlying rendering engine supported a layer binary bitmask for both objects and the camera.
                        # Below we set the two image planes, left and right, to layers=1 and layers=2.
                        # Note that these two masks are associated with left eye’s camera and the right eye’s camera.
                        layers=1,
                        format="jpeg",
                        quality=50,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        display_image[:, self.img_width :],
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=50,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            # 'jpeg' encoding should give you about 30fps with a 16ms wait in-between.
            await asyncio.sleep(0.016 * 2)

    async def main_image_monocular(self, session, fps=60):
        session.upsert @ Hands(
            fps=fps, stream=True, key="hands", showLeft=False, showRight=False
        )
        while True:
            display_image = cv2.cvtColor(self.img_array, cv2.COLOR_BGR2RGB)
            # aspect_ratio = self.img_width / self.img_height
            session.upsert(
                [
                    ImageBackground(
                        display_image,
                        aspect=1.778,
                        height=1,
                        distanceToCamera=1,
                        format="jpeg",
                        quality=50,
                        key="background-mono",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(0.016)

    @property
    def left_hand(self):
        return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def right_hand(self):
        return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")

    @property
    def left_landmarks(self):
        return np.array(self.left_landmarks_shared[:]).reshape(25, 3)

    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        return float(self.aspect_shared.value)

    @staticmethod
    def extract_hand_poses(
        hand_data,
        arm_pose_shared=None,
        hand_position_shared=None,
        hand_orientation_shared=None,
    ):
        if arm_pose_shared is not None:
            with arm_pose_shared.get_lock():
                arm_pose_shared[:] = hand_data[0:16]

        if hand_position_shared is not None:
            with hand_position_shared.get_lock():
                for i in range(25):
                    base = i * 16
                    hand_position_shared[i * 3 : i * 3 + 3] = [
                        hand_data[base + 12],
                        hand_data[base + 13],
                        hand_data[base + 14],
                    ]

        if hand_orientation_shared is not None:
            with hand_orientation_shared.get_lock():
                for i in range(25):
                    base = i * 16
                    hand_orientation_shared[i * 9 : i * 9 + 9] = [
                        hand_data[base + 0],
                        hand_data[base + 1],
                        hand_data[base + 2],
                        hand_data[base + 4],
                        hand_data[base + 5],
                        hand_data[base + 6],
                        hand_data[base + 8],
                        hand_data[base + 9],
                        hand_data[base + 10],
                    ]

    # def extract_hand_states(state_dict, prefix):
    #     # pinch
    #     with getattr(self, f"{prefix}_pinch_state_shared").get_lock():
    #         getattr(self, f"{prefix}_pinch_state_shared").value = bool(state_dict.get("pinch", False))
    #     with getattr(self, f"{prefix}_pinch_value_shared").get_lock():
    #         getattr(self, f"{prefix}_pinch_value_shared").value = float(state_dict.get("pinchValue", 0.0))
    #     # squeeze
    #     with getattr(self, f"{prefix}_squeeze_state_shared").get_lock():
    #         getattr(self, f"{prefix}_squeeze_state_shared").value = bool(state_dict.get("squeeze", False))
    #     with getattr(self, f"{prefix}_squeeze_value_shared").get_lock():
    #         getattr(self, f"{prefix}_squeeze_value_shared").value = float(state_dict.get("squeezeValue", 0.0))


if __name__ == "__main__":
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    import threading
    from image_server.image_client import ImageClient

    # image
    img_shape = (480, 640 * 2, 3)
    img_shm = shared_memory.SharedMemory(
        create=True, size=np.prod(img_shape) * np.uint8().itemsize
    )
    img_array = np.ndarray(img_shape, dtype=np.uint8, buffer=img_shm.buf)
    img_client = ImageClient(tv_img_shape=img_shape, tv_img_shm_name=img_shm.name)
    image_receive_thread = threading.Thread(
        target=img_client.receive_process, daemon=True
    )
    image_receive_thread.start()

    # television
    tv = TeleVision(True, img_shape, img_shm.name)
    print("vuer unit test program running...")
    print("you can press ^C to interrupt program.")
    while True:
        time.sleep(0.03)
