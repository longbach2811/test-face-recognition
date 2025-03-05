import onnxruntime as ort
import cv2
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import cosine_distances
# from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

def letterbox(im, new_shape=(512, 512), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

class InferenceOnnx:
    def __init__(self, model_dir, size=(112, 112)):
        self.size = size
        self.ort_session = self.load_onnx_model(model_dir)
    
    def load_onnx_model(self, model_dir):
        providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        ort_model = ort.InferenceSession(model_dir, providers=providers)
        return ort_model
    
    def inference_onnx_model(self, image_path):
        image = cv2.imread(image_path)
        input_data = self.preprocess(image)
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        result = torch.Tensor(self.ort_session.run([output_name], {input_name: input_data})[0])
        # result = self.l2_norm(result, axis=1)
        return result[0]
       
    def l2_norm(self, result, axis=1):
        norm = torch.norm(result, 2, axis, True)
        output = torch.div(result, norm)
        return output

    
    def preprocess(self, image):
        image, _, _ = letterbox(image, new_shape=self.size, color=(0, 0, 0))
        image = cv2.resize(image, dsize=self.size)
        cv2.imwrite("image.jpg", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.divide(image, 255.0)
        image = np.subtract(image, 0.5)
        image = np.divide(image, 0.5)
        image_data = np.transpose(image, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data


class TestFaceRecognition:
    def __init__(self, model_dir, data_dir, ratio=0.5):
        self.model = InferenceOnnx(model_dir)
        self.model = InferenceOnnx(model_dir)
        self.data_dir = data_dir
        self.train_image_paths, train_labels, self.val_image_paths, val_labels = self._make_dataset(self.data_dir, ratio)
        print(f"Number of image for reference: {len(self.train_image_paths)}")
        print(f"Number of image for validation: {len(self.val_image_paths)}")
        overall_label = train_labels + val_labels
 
        unique_labels = sorted(set(overall_label))
        self.label_to_idx = {label: index for index, label in enumerate(unique_labels)}

        self.train_labels = [self.label_to_idx[label] for label in train_labels]
        self.val_labels = [self.label_to_idx[label] for label in val_labels]
    
    def _make_dataset(self, data_dir, ratio):
        train_image_paths, train_labels = [], []
        val_image_paths, val_labels = [], []
        
        list_of_person = os.listdir(data_dir)
        for person in list_of_person:
            person_path = os.path.join(data_dir, person)
            item_list =  os.listdir(person_path)
            split_number = int(len(item_list) * ratio)
            
            # handle the case if the # item equal or less than 1:
            if len(item_list) <= 1:
                train_image_paths.extend([os.path.join(person_path, element) for element in item_list])
                train_labels.extend([person] * len(item_list))
                continue

            for element in item_list[:split_number]:
                image_path = os.path.join(person_path, element)
                train_image_paths.append(image_path)
                train_labels.append(person)
                
            for element in item_list[split_number:]:
                image_path = os.path.join(person_path, element)
                val_image_paths.append(image_path)
                val_labels.append(person)
            
        return train_image_paths, train_labels, val_image_paths, val_labels
    
    def _get_embeddings(self, image_paths):
        embeddings = []
        for image_path in tqdm(image_paths, desc="Compute embedded", leave=False):
            embedding = self.model.inference_onnx_model(image_path)
            embeddings.append(embedding.cpu().numpy())
        return embeddings
    
    def validate(self):
        train_embeddings = self._get_embeddings(self.train_image_paths)
        val_embeddings = self._get_embeddings(self.val_image_paths)
        
        correct_predictions = 0
        for val_embedding, val_label in tqdm(zip(val_embeddings, self.val_labels), desc="Validating", leave=False):
            val_embedding = val_embedding.reshape(1, -1)
            similarities = cosine_similarity(val_embedding, train_embeddings)[0]
            predicted_label = self.train_labels[np.argmax(similarities)]
            if predicted_label == val_label:
                correct_predictions += 1
        
        accuracy = correct_predictions / len(self.val_labels) if self.val_labels else 0
        print(f"Validation accuracy: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":
    test = TestFaceRecognition(
        model_dir=r"D:\longbh\FaceRecognition\test-face-recognition\model_zoo\ms1m_megaface_r50_pfc.onnx",
        data_dir=r"D:\longbh\FaceRecognition\test-face-recognition\lfw-deepfunneled_crop",
        ratio=0.5
    )
    
    test.validate()
