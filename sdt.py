import cv2
import numpy as np

def four_point_transform(image, pts):
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    # or use this function
    # M = cv2.findHomography(pts, dst)[0]

    print("angle of rotation: {}".format(np.arctan2(-M[1, 0], M[0, 0]) * 180 / np.pi))

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

# define 4 points for ROI
def selectROI(event, x, y, flags, param):
    global imagetmp, roiPts

    if event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        print(x, y)
        cv2.circle(imagetmp, (x, y), 2, (0, 255, 0), -1)
        if len(roiPts) < 1:
            cv2.line(imagetmp, roiPts[-2], (x, y), (0, 255, 0), 2)
        if len(roiPts) == 4:
            cv2.line(imagetmp, roiPts[0], (x, y), (0, 255, 0), 2)

        cv2.imshow("image", imagetmp)
        print("select ROI")


def main():
    global imagetmp, roiPts
    roiPts = []

    image = cv2.imread("images/image2.png")
    imagetmp = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", selectROI)

    while len(roiPts) < 4:
        cv2.imshow("image", imagetmp)
        cv2.waitKey(500)

    roiPts = np.array(roiPts, dtype=np.float32)

    warped = four_point_transform(image, roiPts)

    cv2.imshow("Warped", warped)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()