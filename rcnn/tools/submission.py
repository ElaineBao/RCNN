"""
usage:
python -m rcnn.tools.submission --pklfileIn imagenet__test_detections_merge.pkl --txtfileOut detections_submission.txt
"""
from __future__ import print_function
import argparse
import cPickle

SUBMIT_FORMAT = '{img_index} {DET_CLS_ID} {confidence:f} {xmin:f} {ymin:f} {xmax:f} {ymax:f}\n'

def parse_args():
    parser = argparse.ArgumentParser(description='Imagenet Challenge Submission')

    parser.add_argument('--pklfileIn', help='bbox results in pkl format', type=str)
    parser.add_argument('--txtfileOut', help='bbox results in txt format', default='detections_submission.txt', type=str)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)

    det_file = args.pklfileIn
    print("loading detection result from {}".format(det_file))
    with open(det_file, 'r') as f:
        all_boxes = cPickle.load(f)

    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    print("num_classes:{}, num_images:{}".format(num_classes, num_images))

    fileOut = args.txtfileOut
    print("writing detection result to {}".format(fileOut))
    with open(fileOut, 'wt') as f:
        for i in xrange(num_images):
            for j in xrange(1, num_classes):
                box_img_cls = all_boxes[j][i]
                if box_img_cls.shape[0] == 0:
                    continue
                else:
                    for k in range(box_img_cls.shape[0]):
                        line = SUBMIT_FORMAT.format(img_index=i+1, DET_CLS_ID=j,
                                                         confidence=box_img_cls[k, -1],
                                                         xmin=box_img_cls[k, 0], ymin=box_img_cls[k, 1],
                                                         xmax=box_img_cls[k, 2], ymax=box_img_cls[k, 3])
                        print(line)
                        f.write(line)

if __name__ == '__main__':
    main()

