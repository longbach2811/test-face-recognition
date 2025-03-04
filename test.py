import onnxruntime as ort
import cv2
import numpy as np
import torch
import os
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

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
        result = self.l2_norm(result, axis=1)  # Apply correct normalization
        return result[0]
       
    def l2_norm(self, result, axis=1):
        norm = torch.norm(result, 2, axis, True)
        output = torch.div(result, norm)
        return output

    
    def preprocess(self, image):
        # image, _, _ = letterbox(image, new_shape=self.size, color=(0, 0, 0))
        image = cv2.resize(image, dsize=self.size)
        cv2.imwrite("image.jpg", image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = np.array(image) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data


class TestFaceRecognition:
    def __init__(self, model_dir, data_dir):
        self.model = InferenceOnnx(model_dir)
        self.data_dir = data_dir
        self.train_image_paths, self.train_labels, self.val_image_paths, self.val_labels = self._make_dataset(self.data_dir)
    
    def _make_dataset(self, data_dir):
        train_image_paths, train_labels = [], []
        val_image_paths, val_labels = [], []
        for root, _, files in os.walk(data_dir):
            if len(files) < 2:
                continue 
            train_image_path = os.path.join(root, files[0])
            train_image_paths.append(train_image_path)
            train_label = int(files[0].split("_")[0])
            train_labels.append(train_label)
            
            val_image_path = os.path.join(root, files[1])
            val_image_paths.append(val_image_path)
            val_label = int(files[1].split("_")[0])
            val_labels.append(val_label)
            
        train_image_paths, train_labels = zip(*sorted(zip(train_image_paths, train_labels), key=lambda x: x[1]))
        val_image_paths, val_labels = zip(*sorted(zip(val_image_paths, val_labels), key=lambda x: x[1]))
        return train_image_paths, train_labels, val_image_paths, val_labels
    
    def _get_embeddings(self, image_paths):
        embeddings = []
        for image_path in tqdm(image_paths, desc="Compute  ", leave=False):
            embedding = self.model.inference_onnx_model(image_path)
            embeddings.append(embedding.cpu().numpy())
        return embeddings
    
    def validate(self):
        train_embeddings = self._get_embeddings(self.train_image_paths)
        val_embeddings = self._get_embeddings(self.val_image_paths)
        
        correct_predictions = 0
        for val_embedding, val_label in tqdm(zip(val_embeddings, self.val_labels), desc="Validating", leave=False):
            val_embedding = val_embedding.reshape(1, -1)  # Ensure correct shape for cosine similarity
            similarities = cosine_similarity(val_embedding, train_embeddings)[0]
            predicted_label = self.train_labels[np.argmax(similarities)]
            if predicted_label == val_label:
                correct_predictions += 1
            # break
        
        accuracy = correct_predictions / len(self.val_labels) if self.val_labels else 0
        print(f"Validation accuracy: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":
    test = TestFaceRecognition(
        model_dir=r"D:\longbh\FaceRecognition\test-face-recognition\model_zoo\R100.onnx",
        data_dir=r"D:\longbh\FaceRecognition\test-face-recognition\test_sets\lfw_2nd_cut\gen"
    )
    
    test.validate()
