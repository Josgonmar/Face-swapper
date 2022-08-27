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

            if (len(bbox) > 1):
                image_dst = self.__image.copy()
                retval, landmarks_list = self.__landmark_detector_obj.fit(image_dst, bbox)

                face_1_dst_landmarks = landmarks_list[0][0][0:17] # Get only the external facial landmarks
                face_2_dst_landmarks = landmarks_list[1][0][0:17]
                
                face_1_src_img = self.__image[bbox[0][1]:bbox[0][1]+bbox[0][3], bbox[0][0]:bbox[0][0]+bbox[0][2], :]
                face_2_src_img = self.__image[bbox[1][1]:bbox[1][1]+bbox[1][3], bbox[1][0]:bbox[1][0]+bbox[1][2], :]

                face_2_src_bbox = []
                face_2_src_bbox.append([0,0,face_2_src_img.shape[1],face_2_src_img.shape[0]])
                retval, face_2_src_landmarks = self.__landmark_detector_obj.fit(face_2_src_img, np.array(face_2_src_bbox).astype(int))

                h_f2_to_f1, mask = cv2.findHomography(face_2_src_landmarks[0][0][0:17], face_1_dst_landmarks, cv2.RANSAC)

                warped_image = cv2.warpPerspective(face_2_src_img, h_f2_to_f1, (image_dst.shape[1], image_dst.shape[0]))
                mask = np.zeros([image_dst.shape[0], image_dst.shape[1]], dtype=np.uint8)

                cv2.fillConvexPoly(mask, np.int32([face_1_dst_landmarks]), (255, 255, 255), cv2.LINE_AA)
                
                warped_image = warped_image.astype(float)
                mask3 = np.zeros_like(warped_image)
                for i in range(0, 3):
                    mask3[:, :, i] = mask / 255
                mask3_inv = 1 - mask3

                frame_masked = cv2.multiply(image_dst.astype(float), mask3_inv)
                frame_out = cv2.add(warped_image.astype(np.uint8), frame_masked.astype(np.uint8))

                cv2.imshow('Final frame', frame_out)
                cv2.waitKey(0)
            else:
                print('[ERROR] Not enough faces detected across the image. Only 2 are needed to proceed')
        else:
            print('[WARINING] No image loaded. Finishing the program.')

        cv2.destroyAllWindows()
    
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
                self.__image = cv2.imread(self.__paths.folder_path + '/' + file, cv2.IMREAD_COLOR)
            print('[INFO] Loaded image')
        except:
            print('[ERROR] An error occured while reading the images')
    
    def __saveResult(self, image, image_name):
        try:
            filename = 'swapped' + image_name
            cv2.imwrite(filename, image)
            print('[INFO] Correctly saved image as:', filename)
        except:
            print('[ERROR] An error occurred while saving the image')

        

if __name__ == "__main__":
    FaceSwapper_obj = FaceSwapper()
    FaceSwapper_obj.run()