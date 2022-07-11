# !pip install git+https://github.com/openai/CLIP.git
# !wget https://github.com/haltakov/natural-language-image-search/releases/download/1.0.0/photo_ids.csv \
#     -O ~/unsplash-dataset/photo_ids.csv
# !wget https://github.com/haltakov/natural-language-image-search/releases/download/1.0.0/features.npy \
#     -O ~/unsplash-dataset/features.npy
import requests
from io import BytesIO
from PIL import Image
import clip
import numpy as np
import pandas as pd
import torch


class CLIPImageFinder:
    def __init__(self, photo_ids_path, photo_features_path, device='cpu'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        photo_ids = pd.read_csv(photo_ids_path)[:50000]
        self.photo_ids = list(photo_ids['photo_id'])

        photo_features = np.load(photo_features_path)[:50000]
        self.photo_features = torch.from_numpy(photo_features).to(self.device)

    def encode_search_query(self, search_query):
        with torch.no_grad():
            # Encode and normalize the search query using CLIP
            text_encoded = self.model.encode_text(clip.tokenize(search_query).to(self.device))[0]
            text_encoded /= text_encoded.norm()
        # Retrieve the feature vector
        return text_encoded

    def find_best_matches(self, text_features, photo_features, photo_ids, results_count=3):
        # Compute the similarity between the search query and each photo using the Cosine similarity
        similarities = photo_features @ text_features

        # Sort the photos by their similarity score
        best_photo_idx = (-similarities).argsort()

        # Return the photo IDs of the best matches
        best_photos = []
        best_scores = []
        for idx in best_photo_idx:
            if len(best_photos) == results_count:
                break
            try:
                best_photos.append(self.load_photo(self.photo_ids[idx]))
                best_scores.append(similarities[idx])
            except Exception:
                continue

        return best_photos, best_scores

    def load_photo(self, photo_id):
        # Get the URL of the photo resized to have a width of 320px
        photo_image_url = f"https://unsplash.com/photos/{photo_id}/download?w=320"
        response = requests.get(photo_image_url)
        return Image.open(BytesIO(response.content))

    def search_unsplash(self, search_query, results_count=3):
        # Encode the search query
        text_features = self.encode_search_query(search_query)

        # Find the best matches
        best_photos, best_scores = self.find_best_matches(text_features, self.photo_features, self.photo_ids, results_count)
        # Display the best photos
        return best_photos, best_scores
