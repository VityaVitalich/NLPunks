from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import torch
import torch.nn.functional as F
from clip_image_finder import CLIPImageFinder
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from functools import partial
from tqdm.auto import tqdm
import numpy as np
import pickle as pkl
from sklearn.preprocessing import normalize
import logging

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


seed = 0xAB0BA
torch.manual_seed(seed)
device = 'cuda:0'if torch.cuda.is_available() else 'cpu'

try:

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    model.eval()
    model = model.to(device)

    logger.info('DialoGPT loaded')

    qa_model = SentenceTransformer('clips/mfaq')
    qa_model = qa_model.to(device)

    logger.info('Sentence Transformer loaded')

    model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    logger.info('CLIP loaded')


    photo_ids_path = '/home/admin/unsplash-dataset/photo_ids.csv'
    photo_features_path = '/home/admin/unsplash-dataset/features.npy'

    image_finder = CLIPImageFinder(photo_ids_path, photo_features_path, device=device)

    #print('CLIP features loaded')
    logger.info('CLIP features loaded')

except Exception as e:
    
    logger.exception(e)
    
    raise e

def normalize(x, x_max, x_min):
    return (x-x_min)/(x_max-x_min)
    

def calculate_gpt_confidence(phrases, prefix = 'Whats on this picture? '):
    gpt_max = -6.414142608642578
    gpt_min = -12.518062591552734
    
    
    texts = [prefix + elem for elem in phrases]

    tokens = [tokenizer.encode(text, return_tensors='pt').squeeze(0) for text in texts]
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True,
                                                 padding_value=tokenizer.eos_token_id)
    tokens = tokens.to(device)

    outputs = model(tokens, labels=tokens)

    log_proba = F.log_softmax(outputs.logits, dim=-1)

    true_proba = log_proba[torch.arange(log_proba.shape[0]).unsqueeze(-1),
                                       torch.arange(log_proba.shape[1]).unsqueeze(0),
                                       tokens]
    
    mean_proba_gpt = true_proba.mean(dim=-1)

    return normalize(mean_proba_gpt, gpt_max, gpt_min)


def calculate_qa_score(phrases, prefix = '<Q>What do you see on this image?'):
    qa_max = 160
    qa_min = 50
    
    texts = ['<A>' + phrase for phrase in phrases]
    embeddings = qa_model.encode([prefix] + texts)
    q_emb, *ans_emb = embeddings
    qa_scores = []
    for emb in ans_emb:
        qa_scores.append(np.dot(q_emb, emb))
        
    qa_scores = torch.tensor(qa_scores)

    return normalize(qa_scores, qa_max, qa_min)

def calculate_clip_score(phrases, image_finder = image_finder):
    
    clip_max = 0.395263671875
    clip_min = 0.2374267578125
    
    scores = []
    images = []
    for phrase in phrases:
        img, score = image_finder.search_unsplash(phrase, results_count=1)
        scores.append(score)
        images.append(img)
    
    return images, normalize(torch.tensor(scores), clip_max, clip_min)

def score_image(phrase, show_image=False):
    gpt_score = calculate_gpt_confidence(phrase)
    image, conf_clip = calculate_clip_score(phrase)
    qa_score = calculate_qa_score(phrase)
    
    gpt_score = gpt_score.to(device)
    conf_clip = conf_clip.to(device)
    qa_score = qa_score.to(device)
    
    normed_score = gpt_score + conf_clip.flatten() + qa_score
    
    if show_image:
        for img in image:
            img[0].show()
    
    return image, normed_score
