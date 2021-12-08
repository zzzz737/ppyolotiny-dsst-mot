import glob
import os

import cv2

from multi_tracker import MultiTracker


def images2mp4(images_dir, output_path):
    """
    :param images_dir: 图片序列文件夹路径
    :param output_path: 图片所合成视频的保存路径
    """
    assert output_path.endswith('.mp4')
    dir_path = os.path.join(os.path.dirname(__file__), *output_path.split(os.sep)[:-1])
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    video = cv2.VideoWriter(
        filename=output_path,
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=10,
        frameSize=(512, 512))

    for img_path in sorted(glob.glob(os.path.join(images_dir, '*.jpg'))):
        video.write(cv2.imread(img_path))

    video.release()


def predict_stream(stream_path, multi_tracker, save_dir):
    """
    :param stream_path: 视频流文件的地址
    :param multi_tracker: 多目标跟踪器类
    :param save_dir: 每帧绘制结果图片的保存文件夹地址
    """
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    video = cv2.VideoCapture(stream_path)

    while True:
        _, frame = video.read()
        if frame is None:
            video.release()
            break

        plotted_frame = multi_tracker.update(frame)

        save_path = os.path.join(save_dir, '%03d.jpg' % (multi_tracker.frame_num - 1))
        cv2.imwrite(save_path, plotted_frame)


if __name__ == '__main__':
    images2mp4(
        images_dir='DIC-C2DH-HeLa/Test',
        output_path=f'output{os.sep}test.mp4')

    multi_tracker = MultiTracker(
        det_model_path='cell_det_model',
        det_threshold=0.25,
        stride=2)
    predict_stream(
        stream_path='output/test.mp4',
        multi_tracker=multi_tracker,
        save_dir='output/track_result')

    images2mp4(
        images_dir='output/track_result',
        output_path=f'output{os.sep}test_result.mp4')
