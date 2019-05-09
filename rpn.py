import numpy as np
import cv2 as cv

from synthetic_dataset import ShapeDataset

class Anchors:

    def __init__(self, image_size):
        self.anchors = self.get_anchor_rect(image_size)

    def get_anchor_rect(self, image_size):
        """
        Total anchors will be produced for 600 x 800 image is 38*50*9 = 17100
        Anchors are like [y, x, h, w, y_pos, x_pos, anchortype]
        x_pos, y_pos - the position at the feature
        :param image_size:
        :return:
        """
        anchor_ratio = [[1, 1], [1, 2], [2, 1]]
        anchor_size = [128, 256, 512]
        stride = 16
        anchors = []
        for r in range(0, image_size[0], stride):
            for c in range(0, image_size[1], stride):
                anchor_cnt = 0
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

                        anchors.append([r_start, c_start, r_end, c_end, float(r/stride), float(c/stride), anchor_cnt])
                        anchor_cnt += 1

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

    def bbox_correction(self, rect, anc):

        # Rectangle's center and height and width
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        x = rect[0] + w / 2
        y = rect[1] + h / 2

        # anchors's center and height and width
        wa = anc[3] - anc[1]
        ha = anc[2] - anc[0]
        xa = anc[1] + wa / 2
        ya = anc[0] + ha / 2

        tx = (x - xa)/wa
        ty = (y - ya)/ha
        tw = np.log(w/wa)
        th = np.log(h/ha)

        return [tx, ty, tw, th]

    def anchor_prespective_rects(self, rect_list):
        modified_anchors = []
        for anc in self.anchors:
            is_rect_in_anchor = False
            corrected = [0, 0, 0, 0]
            for rect in rect_list:
                iou = self.intersection_over_union(rect[1], [anc[1], anc[0], anc[3], anc[2]])
                if iou >= 0.5:
                    is_rect_in_anchor = True
                    corrected = self.bbox_correction(rect[1], anc)
                    break

            if is_rect_in_anchor == True:
                modified_anchors.append([1, 0] + corrected)
            else:
                modified_anchors.append([0, 1] + corrected)

        return modified_anchors

    def rectangle_anchor_match(self, rectangle):
        foreground_anchors = []
        background_anchors = []
        non_fg_bg_anchors = []
        for anc in self.anchors:
            iou = self.intersection_over_union(rectangle, [anc[1], anc[0], anc[3], anc[2]])
            anc = self.bbox_correction(rectangle, anc)
            if iou >= 0.7:
                foreground_anchors.append(anc)
            elif iou <= 0.3:
                background_anchors.append(anc)
            else:
                non_fg_bg_anchors.append(anc)

        return foreground_anchors, background_anchors, non_fg_bg_anchors

def preprocessing_img(img):

    img /= 255.0
    img -= 0.5

    img = np.expand_dims(img, axis=0)
    return img

def rpn_generator():

    img_size = [600, 800, 3]
    shaper = ShapeDataset()
    anchors = Anchors(img_size)

    k = 256
    feature_y = int(np.round(600 / 16))
    feature_x = int(np.round(800 / 16))
    rpn_objectness = np.zeros([feature_y, feature_x, 2*k], np.float)

    rpn_bbox = np.zeros([feature_y, feature_x, 4*k])
    while True:
        img, rects = shaper.generate_image([600, 800], 15)

        modified_anchors = anchors.anchor_prespective_rects(rects)
        modified_anchors = np.array(modified_anchors, np.float)

        img = preprocessing_img(img)

        objectness = np.expand_dims(modified_anchors[:, 0:2], axis=0)
        bbox_out = np.expand_dims(modified_anchors[:, 2:], axis=0)

        # print('objectness.shape: ', objectness.shape)
        # print('bbox_out.shape: ', bbox_out.shape)

        yield img, [objectness, bbox_out]

        # fg_list = []
        # bg_list = []
        # for rect in rects:
        #     fa, ba, na = anchors.rectangle_anchor_match(rect[1])
        #     fg_list = fg_list + fa
        #     bg_list= bg_list + ba
        #
        # max_value = 256
        # """**************Foreground anchors********************"""
        # if len(fg_list) > max_value:
        #     fg_list = fg_list[np.random.randint(fg_list.shape[0], size=max_value-2), ::]
        # fg_list = np.c_[np.ones(len(fg_list)), np.zeros(len(fg_list)), fg_list]
        #
        # print('fg_list:', fg_list.shape)
        #
        # """**************Background anchors********************"""
        # bg_list_selected = np.array(bg_list)
        # bg_list_selected = bg_list_selected[
        #                    np.random.randint(
        #                        bg_list_selected.shape[0], size=max_value - len(fg_list)), ::]
        # bg_list_selected = np.c_[np.zeros(len(bg_list_selected)), bg_list_selected]
        # print('bg_list:', bg_list_selected.shape)
        #
        # """**************Mixed and shuffled anchors********************"""
        # rpn_list = np.concatenate([fg_list, bg_list_selected], axis=0)
        # np.random.shuffle(rpn_list)
        # print('rpn_list:', rpn_list.shape)
        #
        # img = preprocessing_img(img)
        #
        # yield img, []
        # break

if __name__ == '__main__':

    for rpn in rpn_generator():
        print(rpn[0].shape)
