import numpy as np
import cv2
import os

class ModelParams():
    mean = [104, 117, 123]
    scale = 1.0
    in_width = 300
    in_height = 300

class Paths():
    folder_path = '../visuals'
    facial_landmarks_model_path = '../model/lbfmodel.yaml'
    model_path = '../model/deploy.prototxt'
    model_config_path = '../model/res10_300x300_ssd_iter_140000.caffemodel'

class FaceSwapper:
    __image = None
    __file  = None
    __model_params = ModelParams()
    __paths = Paths()

    __face_detector = None
    __landmark_detector_obj = None

    def __init__(self):
        self.__face_detector = cv2.dnn.readNetFromCaffe(self.__paths.model_path, self.__paths.model_config_path)
        self.__landmark_detector_obj = cv2.face.createFacemarkLBF()
        self.__landmark_detector_obj.loadModel(self.__paths.facial_landmarks_model_path)

    def run(self):
        self.__loadImage()
        if (self.__image.any()):
            blob = cv2.dnn.blobFromImage(self.__image, scalefactor=self.__model_params.scale, size=(self.__model_params.in_width, self.__model_params.in_height),
                            mean=self.__model_params.mean, swapRB=False, crop=False)
            self.__face_detector.setInput(blob)
            detections = self.__face_detector.forward()
            bbox = self.__getFaceBoundingBox(self.__image, detections)

            if (len(bbox) == 2):
                image_dst = self.__image.copy()
                retval, landmarks_list = self.__landmark_detector_obj.fit(image_dst, bbox)

                self.__warpSrcToDstFace(self.__image, image_dst, landmarks_list[0][0], landmarks_list[1][0])
                self.__warpSrcToDstFace(self.__image, image_dst, landmarks_list[1][0], landmarks_list[0][0])

                cv2.imshow('Final frame, press any key to save the results', image_dst)
                cv2.waitKey(0)

                self.__saveResult(image_dst, self.__file)
            else:
                print('[ERROR] Not enough faces detected across the image. Only 2 are needed to proceed')
        else:
            print('[WARINING] No image loaded. Finishing the program.')

        cv2.destroyAllWindows()

    def __warpSrcToDstFace(self, src_img, dst_img, src_face_landmarks, dst_face_landmarks):
        face_1_dst_landmarks = []
        face_2_dst_landmarks = []

        hullIndex = cv2.convexHull(np.array(src_face_landmarks), returnPoints=False)

        for i in range(0, len(hullIndex)):
            face_1_dst_landmarks.append(src_face_landmarks[hullIndex[i][0]])
            face_2_dst_landmarks.append(dst_face_landmarks[hullIndex[i][0]])

        rect = (0, 0, dst_img.shape[1], dst_img.shape[0])

        dt = self.__calculateDelaunayTriangles(rect, face_2_dst_landmarks)
        if len(dt) == 0:
            print("No Delanauy triangles calculated")

        for i in range(0, len(dt)):
            t1 = []
            t2 = []

            for j in range(0, 3):
                t1.append(face_1_dst_landmarks[dt[i][j]])
                t2.append(face_2_dst_landmarks[dt[i][j]])

            self.__warpTriangle(src_img, dst_img, t1, t2)

    def __rectContains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True

    def __applyAffineTransform(self, src, srcTri, dstTri, size):
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        return dst
        
    def __warpTriangle(self, img1, img2, t1, t2):
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        t1Rect = []
        t2Rect = []
        t2RectInt = []

        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

        mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

        img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

        size = (r2[2], r2[3])

        img2Rect = self.__applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

        img2Rect = img2Rect * mask

        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
        img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect

    def __calculateDelaunayTriangles(self, rect, points):
        subdiv = cv2.Subdiv2D(rect)

        for p in points:
            if p[0] > self.__image.shape[1]:
                p[0] = self.__image.shape[1] - 1

            if p[1] > self.__image.shape[0]:
                p[1] = self.__image.shape[0] - 1

            subdiv.insert((p[0], p[1]))

        triangleList = subdiv.getTriangleList()

        delaunayTri = []

        for t in triangleList:
            pt = []
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))

            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            if self.__rectContains(rect, pt1) and self.__rectContains(rect, pt2) and self.__rectContains(rect, pt3):
                ind = []
                for j in range(0, 3):
                    for k in range(0, len(points)):
                        if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)
                if len(ind) == 3:
                    delaunayTri.append((ind[0], ind[1], ind[2]))

        return delaunayTri

    def __getFaceBoundingBox(self, image, detections, detection_threshold=0.90):
        height, width = image.shape[:2]
        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if (confidence >= detection_threshold):
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x1, y1, x2, y2) = box.astype("int")

                face_width = x2 - x1
                face_height = y2 - y1

                faces.append([x1, y1, face_width, face_height])
    
        return np.array(faces).astype(int)

    def __loadImage(self):
        try:
            for file in os.listdir(self.__paths.folder_path):
                self.__file = file
                self.__image = cv2.imread(self.__paths.folder_path + '/' + file, cv2.IMREAD_COLOR)
            print('[INFO] Loaded image')
        except:
            print('[ERROR] An error occured while reading the images')

    def __saveResult(self, image, image_name):
        try:
            filename = self.__paths.folder_path + '/' + 'swapped-' + image_name
            cv2.imwrite(filename, image)
            print('[INFO] Correctly saved image as:', filename)
        except:
            print('[ERROR] An error occurred while saving the image')



if __name__ == "__main__":
    FaceSwapper_obj = FaceSwapper()
    FaceSwapper_obj.run()