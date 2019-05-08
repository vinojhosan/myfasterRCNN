import numpy as np
import cv2 as cv

from synthetic_dataset import ShapeDataset

class Anchors:

    def __init__(self, image_size):
        self.anchors = self.get_anchor_rect(image_size)

    def get_anchor_rect(self, image_size):
        """
        Total anchors will be produced for 600 x 800 image is 38*50*9 = 17100
        :param image_size:
        :return:
        """
        anchor_ratio = [[1, 1], [1, 2], [2, 1]]
        anchor_size = [128, 256, 512]
        stride = 16
        anchors = []
        for r in range(0, img.shape[0], stride):
            for c in range(0, img.shape[1], stride):
                for ratio in anchor_ratio:
                    r_ratio = ratio[0]
                    c_ratio = ratio[1]

                    for a_s in anchor_size:
                        r_start = r-int(a_s*r_ratio/2)
                        r_end = r+int(a_s*r_ratio/2)
                        c_start = c-int(a_s*c_ratio/2)
                        c_end = c+int(a_s*c_ratio/2)

                        if r_start < 0:
                            r_start = 0
                        if r_end > image_size[0]-1:
                            r_end = image_size[0]-1

                        if c_start < 0:
                            c_start = 0
                        if c_end > image_size[1]-1:
                            c_end = image_size[1]-1

                        anchors.append([r_start, c_start, r_end, c_end])

        return anchors

    def anchors_drawing(self, anchors_op, img):
        for anc in anchors_op:
            cv.rectangle(img, pt1=(anc[1], anc[0]), pt2=(anc[3], anc[2]), color=[255, 0, 0], thickness=2)
        cv.imwrite('anchors.png', img)

    def intersection_over_union(self, boxA, boxB):

        # Area of Intersection rectangle of two boxes
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Areas of individual boxes
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # IOU calculation
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def rectangle_anchor_match(self, rectangle):
        foreground_anchors = []
        background_anchors = []
        for anc in self.anchors:
            iou = self.intersection_over_union(rectangle, [anc[1], anc[0], anc[3], anc[2]])

            if iou >= 0.5:
                foreground_anchors.append(anc)
            else:
                background_anchors.append(anc)

        return foreground_anchors, background_anchors


if __name__ == '__main__':
    shapers = ShapeDataset()

    img, shape_rect = shapers.generate_image([600, 800], 10)
    anchors = Anchors(img.shape)
    print(anchors.anchors)
    print('length: ', len(anchors.anchors))

    for rect in shape_rect:
        fa, ba = anchors.rectangle_anchor_match(rect[1])
        print(len(fa), len(ba))

        if len(fa) == 0:
            cv.imwrite('img.png', img)
        # print(fa)