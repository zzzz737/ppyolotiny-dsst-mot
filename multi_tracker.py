import cv2
import dlib
import numpy as np
from scipy.optimize import linear_sum_assignment


class SingleTracker(dlib.correlation_tracker):
    def __init__(self,
                 tracker_id,
                 image,
                 bbox,
                 category,
                 min_score=6.5,
                 max_lost_cnt=4):
        """
            初始化单目标跟踪器。
        :param tracker_id: 跟踪器ID
        :param image: 起始帧图像
        :param bbox: 跟踪目标边框 (x_min, y_min, x_max, y_max)
        :param category: 跟踪目标类别
        :param min_score: 跟踪器默认目标丢失的阈值
        :param max_lost_cnt: 可容忍的最大跟踪失败次数
        """
        super(SingleTracker, self).__init__()

        self.id = int(tracker_id)
        self.category = str(category)
        self.min_score = float(min_score)
        self.lost_cnt = 0
        self.max_lost_cnt = max_lost_cnt
        self.failed = False
        self.bbox_color = (
            np.random.randint(50, 255),
            np.random.randint(50, 255),
            np.random.randint(50, 255))

        self.bbox = bbox
        self.start_track(image, dlib.rectangle(*bbox))

    def update_bbox(self, image):
        """
            在当前帧中更新目标所在区域。
        :param image: 当前帧图像
        """
        score = self.update(image)
        self.lost_cnt = self.lost_cnt + 1 if score < self.min_score else 0
        if self.lost_cnt > self.max_lost_cnt:
            self.failed = True

        pos = self.get_position()
        self.bbox = (int(pos.left()), int(pos.top()),
                     int(pos.right()), int(pos.bottom()))


class MultiTracker:
    def __init__(self,
                 det_model_path,
                 det_threshold=0.25,
                 stride=2):
        """
        :param det_model_path: 检测器路径
        :param det_threshold: 检测器预测结果阈值
        :param stride: 检测器预测间隔
        """
        self.det_threshold = det_threshold
        self.stride = stride
        try:
            from paddlex import load_model
            self.model = load_model(det_model_path)
        except Exception as e:
            raise e

        self.frame_num = 0  # 帧数统计
        self.tracker_num = 0  # 跟踪器ID统计
        self.trackers = []  # 跟踪器实例存储列表

    def _update_trackers(self, image):
        """
            对于每个已有单目标跟踪器，更新观测区域，判断目标是否丢失并删除该实例。
        """
        del_idx = []
        for i in range(len(self.trackers)):
            self.trackers[i].update_bbox(image)
            if self.trackers[i].failed:
                del_idx.append(i)

        self.trackers = [self.trackers[i] for i in range(len(self.trackers)) if i not in del_idx]

    def get_det_result(self, image):
        """
            返回检测器预测结果：([x_min, y_min, x_max, y_max], category)
        """
        results = self.model.predict(image)
        selected_result = []
        for result in results:
            if result['score'] < self.det_threshold:
                continue

            x_min, y_min, w, h = np.int32(result['bbox'])
            selected_result.append((
                [x_min, y_min, x_min + w, y_min + h],
                result['category']))

        return selected_result

    @staticmethod
    def get_IoU(_bbox1, _bbox2):
        """
            输入边框的对角线端点(x_min, y_min, x_max, y_max)，计算两个矩形的交并比IoU。
        """
        x1min, y1min, x1max, y1max = _bbox1
        x2min, y2min, x2max, y2max = _bbox2

        s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
        s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

        x_min, y_min = max(x1min, x2min), max(y1min, y2min)
        x_max, y_max = min(x1max, x2max), min(y1max, y2max)
        inter_w, inter_h = max(y_max - y_min + 1., 0.), max(x_max - x_min + 1., 0.)

        intersection = inter_h * inter_w
        union = s1 + s2 - intersection

        return intersection / union

    def _add_tracker(self, image, bbox: list or tuple, category):
        """
            初始化单目标跟踪器，观测指定区域。
        """
        tracker = SingleTracker(
            tracker_id=self.tracker_num,
            image=image,
            bbox=bbox,
            category=category)
        self.trackers.append(tracker)
        self.tracker_num += 1

    def _add_trackers(self, image):
        """
            将预测框和观测框根据交并比距离进行级联匹配，未匹配上的视为新目标，对其创建跟踪器。
        """
        if self.frame_num % self.stride == 0:
            return

        # 生成检测器预测框列表
        predict_result = self.get_det_result(image)
        predict_bboxes = [bbox for bbox, _ in predict_result]
        # 生成跟踪器观测框列表
        tracker_bboxes = [tracker.bbox for tracker in self.trackers]

        # 生成交并比距离矩阵
        cost_matrix = np.zeros(shape=(len(tracker_bboxes), len(predict_bboxes)), dtype='float32')
        for i in range(len(tracker_bboxes)):
            for j in range(len(predict_bboxes)):
                cost_matrix[i, j] = self.get_IoU(tracker_bboxes[i], predict_bboxes[j])

        # 获取配对结果 (row_i, col_i)
        row, col = linear_sum_assignment(cost_matrix, maximize=True)

        # 生成为配对结果列表，分配单目标跟踪器
        unused_idx = [i for i in range(len(predict_result)) if i not in col]
        for idx in unused_idx:
            bbox, category = predict_result[idx]
            self._add_tracker(image, bbox, category)

    def plot(self, image):
        """
            将各个已有的单目标跟踪器的边框等信息在当前帧上绘制并返回。
        """
        thickness = round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        for tracker in self.trackers:
            # 获取目标边框的 (x_min, y_min) 和 (x_max, y_max)，绘制矩形
            pt1, pt2 = (tracker.bbox[0], tracker.bbox[1]), (tracker.bbox[2], tracker.bbox[3])
            cv2.rectangle(image,
                          pt1=pt1, pt2=pt2,
                          color=tracker.bbox_color,
                          thickness=thickness,
                          lineType=cv2.LINE_AA)

            # 获取类别文字绘制所需的宽和高，得到两个顶点 (x_min, y_min) 和 (x_max, y_max)
            w, h = cv2.getTextSize(text=tracker.category,
                                   fontFace=0,
                                   fontScale=thickness / 3,
                                   thickness=max(thickness - 1, 1))[0]
            font_pt1, font_pt2 = pt1, (pt1[0] + w, pt1[1] + h)
            # 填充类别文字框的背景色
            cv2.rectangle(image,
                          pt1=font_pt1, pt2=font_pt2,
                          color=tracker.bbox_color,
                          thickness=-1,
                          lineType=cv2.LINE_AA)
            # 将字符绘制在文字框内
            cv2.putText(image, '{}({})'.format(tracker.category, tracker.id),
                        org=(font_pt1[0], font_pt2[1]),
                        fontFace=0,
                        fontScale=thickness / 3,
                        color=(225, 255, 255),
                        thickness=max(thickness - 1, 1),
                        lineType=cv2.LINE_AA)

        return image

    def update(self, image):
        self.frame_num = (self.frame_num + 1) % 864000

        self._update_trackers(image)
        self._add_trackers(image)

        return self.plot(image)
