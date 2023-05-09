from SoccerNet.Downloader import SoccerNetDownloader
from dask import delayed, compute
import json
from glob import glob
import os
import numpy as np
from SoccerNet.DataLoader import FrameCV
from dask.distributed import Client
from config import password, DOWNLOAD_PATH, GENERATED_IMAGES_PATH, RESOLUTION


PATH = os.path.join(DOWNLOAD_PATH, "england_epl/**/1_*.mkv")


def download_data():
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory=DOWNLOAD_PATH)

    mySoccerNetDownloader.password = password
    mySoccerNetDownloader.downloadGames(
        files=[f"1_{RESOLUTION}.mkv", f"2_{RESOLUTION}.mkv"],
        split=["train", "valid", "test", "challenge"],
    )

    mySoccerNetDownloader.downloadGames(
        files=[
            "1_player_boundingbox_maskrcnn.json",
            "2_player_boundingbox_maskrcnn.json",
        ],
        split=["train", "valid", "test", "challenge"],
    )


def _generate_images_from_video(video_path, bbox_path, step_size=10):
    fps = 2
    start = None
    duration = None

    video_loader = FrameCV(
        video_path, FPS=fps, transform=None, start=start, duration=duration
    )

    frame_width = video_loader.frames.shape[2]
    frame_height = video_loader.frames.shape[1]

    labels = json.load(open(bbox_path))
    predictions = labels["predictions"]
    ratio_width = labels["size"][2] / frame_width
    ratio_height = labels["size"][1] / frame_height

    frames = []
    new_bboxes = []

    for idx, (frame, bboxes) in enumerate(zip(video_loader.frames, predictions)):
        if idx % step_size != 0:
            continue
        frames.append(frame)
        bboxes_ = []

        for rect in bboxes["bboxes"]:
            x_top_left = min(max(0, int(rect[0] / ratio_width)), frame_width - 1)
            x_bottom_right = min(max(0, int(rect[2] / ratio_width)), frame_width - 1)
            y_top_left = min(max(0, int(rect[1] / ratio_height)), frame_height - 1)
            y_bottom_right = min(max(0, int(rect[3] / ratio_height)), frame_height - 1)
            bboxes_.append([x_top_left, y_top_left, x_bottom_right, y_bottom_right])
        new_bboxes.append(bboxes_)

    return frames, new_bboxes


def _generate_images_wrapper(video_path, bbox_path):
    frames, new_bboxes = _generate_images_from_video(video_path, bbox_path)
    return {"frames": frames, "new_bboxes": new_bboxes}


def _save_output(result, directory, f_video):
    np.save(
        os.path.join(directory, f"{f_video}.npy"),
        np.array(result["frames"]),
        allow_pickle=True,
    )

    json_object = json.dumps({"bboxes": result["new_bboxes"]})
    with open(os.path.join(directory, f"{f_video}_bbox.json"), "w") as f:
        json.dump(json_object, f)


def prepare_data():

    client = Client(n_workers=2, threads_per_worker=1)

    paths = glob(PATH, recursive=True)
    paths = [os.path.dirname(path) for path in paths]

    delayed_tasks = []
    for path in paths:
        for (f_video, f_bbox) in zip(
            [f"1_{RESOLUTION}.mkv", f"2_{RESOLUTION}.mkv"],
            [
                "1_player_boundingbox_maskrcnn.json",
                "2_player_boundingbox_maskrcnn.json",
            ],
        ):
            video_path = os.path.join(path, f_video)
            bbox_path = os.path.join(path, f_bbox)
            directory = os.path.join(GENERATED_IMAGES_PATH, path)
            os.makedirs(directory, exist_ok=True)

            delayed_result = delayed(_generate_images_wrapper)(video_path, bbox_path)
            delayed_tasks.append(
                delayed(_save_output)(delayed_result, directory, f_video)
            )

    compute(*delayed_tasks)

    client.close()


if __name__ == "__main__":
    # download_data()
    prepare_data()
