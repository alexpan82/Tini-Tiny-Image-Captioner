import torch
import skimage.io as io
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
from encoder import Encoder

def main(clip_model_type: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model_name = clip_model_type.replace('/', '_')
    out_path = f"./data/coco/oscar_split_{clip_model_name}_train.pkl"
    if clip_model_type == 'tinyvit':
        tinyvit_model = Encoder(device).to(device)
        _, preprocess = clip.load('ViT-B/32', device=device, jit=False)
    else:
        clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    with open('./data/coco/annotations/train_caption.json', 'r') as f:
        data = json.load(f)
    print("%0d captions loaded from json " % len(data))
    all_embeddings = []
    all_captions = []
    for i in tqdm(range(len(data))):
        d = data[i]
        img_id = d["image_id"]
        filename = f"./data/coco/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        if not os.path.isfile(filename):
            filename = f"./data/coco/val2014/COCO_val2014_{int(img_id):012d}.jpg"
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        # image.shape = torch.Size([1, 3, 224, 224]) with values [0, 1]
        with torch.no_grad():
            if clip_model_type == 'tinyvit':
                logits = tinyvit_model.forward(image).cpu()
                prefix = tinyvit_model.norm_head
                # tinyvit_out.shape = torch.Size([1, 576]) dtype=torch.float32
            else:
                prefix = clip_model.encode_image(image).cpu()
                # prefix.shape = torch.Size([1, 512]) dtype=torch.float16
                
        d["clip_embedding"] = i
        all_embeddings.append(prefix)
        all_captions.append(d)
        if (i + 1) % 10000 == 0:
            with open(out_path, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    with open(out_path, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="tinyvit", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32', 'tinyvit'))
    args = parser.parse_args()
    exit(main(args.clip_model_type))
