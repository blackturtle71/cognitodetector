import re
import os
from tqdm import tqdm
from pathlib import Path

import json
import pandas as pd
import numpy as np

from pydub import AudioSegment
import opensmile

from transformers import AutoTokenizer, AutoModel
import torch

def parse_cha_file(file_path):
    data = {
        "metadata": {},
        "utterances": []
    }

    timecode_pattern = re.compile(r'\x15(\d+_\d+)\x15')  # Extract timecodes
    chat_annotations = re.compile(
        r'''
        \[.*?\]     # Remove [//], [*], etc.
        |[/+.,!?]   # Remove standalone CHAT symbols like /, +, ., etc.
        |&[^\s]+    # Remove &-prefixed codes (e.g., &sm)
        |[<>]       # Remove < and > (but keep content inside)
        |\^[0-9]    # Remove ^ markers (e.g., ^3)
        |::         # Remove elongated colons
        |\x15       # Remove control characters
        |\u201e
        |\u2021
        |\u00e6
        |\u0279
        |\u026a
        |\u0259
        |\u025c
        |x+ # Remove x+ (e.g., "x", "xx", "xxx")
        |\((\w+)\)  # Remove parentheses but keep the word (e.g., "fallin(g)" → "fallin")
        ''', re.VERBOSE
    )

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("@"):
                if ":" in line:
                    if line.startswith("@ID"): # some lines have multiple @ID
                        if 'PAR' in line:
                            key, value = line[1:].split(":", 1)
                            data["metadata"]['ID'] = value.strip()
                        else:
                            continue
                    key, value = line[1:].split(":", 1)
                    data["metadata"][key.strip()] = value.strip()
            if line.startswith('*PAR:'):
                clean_line = line[5:].strip()

                timecode = re.search(timecode_pattern, clean_line)
                timecode = timecode.group(1) if timecode else None
                clean_line = re.sub(timecode_pattern, '', clean_line)

                clean_line = re.sub(r'[&+-^]+(\w+)', r'\1', clean_line)

                clean_line = re.sub(chat_annotations, ' ', clean_line)

                clean_line = re.sub(r'\s+', ' ', clean_line).strip()

                clean_line = clean_line.rstrip('.')

                if clean_line and timecode:
                    data['utterances'].append({
                        'line': clean_line,
                        'time': timecode
                    })

    return data

def extract_audio_segments(audio_path, utterances, output_dir):
    audio = AudioSegment.from_wav(audio_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for i, utterance in enumerate(utterances):
        start, end = map(int, utterance['time'].split('_'))  # convert ms to seconds
        segment = audio[start:end]
        segment.export(f"{output_dir}/utterance_{i}.mp3", format="mp3")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def aggregate_features(participant_dir):
    features = []
    for utterance_file in os.listdir(participant_dir):
        utterance_path = os.path.join(participant_dir, utterance_file)
        feat = smile.process_file(utterance_path) 
        features.append(feat.squeeze())  
    
    return pd.DataFrame(features).mean(axis=0).to_dict() 


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()  


def parse_text_data(task='cookie', dementia=True, overwrite=False):
    if dementia:
        PATH_TO_CHA = f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADReSS-IS2020-train/ADReSS-IS2020-data/train/transcription/cd/"
        PATH_TO_OUT = Path(f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ARESSO_parsed/Dementia/{task}/parsed_data.json")
    else:
        PATH_TO_CHA = f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADReSS-IS2020-train/ADReSS-IS2020-data/train/transcription/cc/" 
        PATH_TO_OUT = Path(f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ARESSO_parsed/Control/{task}/parsed_data.json")

    if overwrite == False and PATH_TO_OUT.exists():
        with open(PATH_TO_OUT, "r") as fin:
            parsed_data = json.load(fin)
    else:
        PATH_TO_OUT.parent.mkdir(parents=True, exist_ok=True)

        parsed_data = []
        for filename in os.listdir(PATH_TO_CHA):
            if filename.endswith('.cha'):
                file_path = os.path.join(PATH_TO_CHA, filename)
                parsed_data.append(parse_cha_file(file_path))

        with open(PATH_TO_OUT, "w") as outfile:
            json.dump(parsed_data, outfile, indent=4)

    data = []
    lost_data = []
    for participant in parsed_data:
        id = re.search(r"S\d+", participant['metadata']['Media']).group(0)
        age_sex = re.search(r'(?:\w+\|){3}(\d{2})[^|]*\|(\w+)\|', participant['metadata']['ID'])
        dementia_type = re.search(r'^eng\|Pitt\|PAR\|\d+;[\d.]*\|[^|]*\|([^|]+)', participant['metadata']['ID'])
        utterances = ' '.join([u['line'] for u in participant['utterances']])
        if age_sex and dementia_type:
            data.append({'ID': f'{id}_{task}', 'age': age_sex.group(1), 'sex': age_sex.group(2), 'dementia':dementia_type.group(1), 'text': utterances})
        else:
            lost_data.append({'ID': id})

    df = pd.DataFrame(data)
    df['age'] = df['age'].astype(int)
    if dementia:
        print(f"Lost dementia {task} data ({len(lost_data)}): {lost_data}")
    else:
        print(f"Lost control {task} data ({len(lost_data)}): {lost_data}")
    return df, parsed_data

def extract_audio_featsures(parsed_data, task='cookie', dementia=True, overwrite=False, overwirte_feats=False):
    if dementia:
        PATH_TO_AUDIO = f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADReSS-IS2020-train/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd/"
        PATH_TO_OUT_AUDIO = Path(f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADRESSO_parsed_audio/Dementia/{task}/")
        PATH_TO_OUT_AUDIO_FEATS = Path(f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADRESSO_audio_features/Dementia/{task}/audio_features.tsv")
    else:
        PATH_TO_AUDIO = f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADReSS-IS2020-train/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc/"
        PATH_TO_OUT_AUDIO = Path(f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADRESSO_parsed_audio/Control/{task}/")
        PATH_TO_OUT_AUDIO_FEATS = Path(f"/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/ADRESSO_audio_features/Control/{task}/audio_features.tsv")
    
    if overwirte_feats == False and PATH_TO_OUT_AUDIO_FEATS.exists():
        return pd.read_csv(PATH_TO_OUT_AUDIO_FEATS, sep='\t')
    
    PATH_TO_OUT_AUDIO_FEATS.parent.mkdir(parents=True, exist_ok=True)
    PATH_TO_OUT_AUDIO.mkdir(parents=True, exist_ok=True)

    for par in tqdm(parsed_data, desc=f"Extracting audio segments for {'Dementia' if dementia == True else 'Control'} {task}"):
        id = re.search(r"S\d+", par['metadata']['Media']).group(0)
        audio_in = f"{PATH_TO_AUDIO}{id}.wav"
        audio_out = Path(f"{PATH_TO_OUT_AUDIO}/{id}/")
        if overwrite == False and audio_out.exists():
            continue
        else:
            audio_out.mkdir(parents=True, exist_ok=True)
        utterances = par['utterances']
        extract_audio_segments(audio_in, utterances, audio_out)
    
    audio_features = []
    for participant in tqdm(parsed_data, desc=f"Extracting audio features for {'Dementia' if dementia == True else 'Control'} {task}"):
        id = re.search(r"S\d+", participant['metadata']['Media']).group(0)
        participant_dir = f"{PATH_TO_OUT_AUDIO}/{id}/"
        
        audio_features.append({'ID': f'{id}_{task}', **aggregate_features(participant_dir)})

    audio_df = pd.DataFrame(audio_features)
    audio_df.to_csv(PATH_TO_OUT_AUDIO_FEATS, index=False, sep='\t')

    return audio_df

if __name__ == "__main__":
    # PARSE TEXT DATA
    cookie_dem_df, cookie_dem_parsed = parse_text_data(task='cookie', dementia=True, overwrite=False)

    cookie_con_df, cookie_con_parsed = parse_text_data(task='cookie', dementia=False, overwrite=False)

    # PARSE AUDIO
    cookie_dem_audio_df = extract_audio_featsures(cookie_dem_parsed, task='cookie', dementia=True, overwrite=False, overwirte_feats=False)


    cookie_con_audio_df = extract_audio_featsures(cookie_con_parsed, task='cookie', dementia=False, overwrite=False, overwirte_feats=False)


    # GET EMBEDDINGS
    dementia_df = cookie_dem_df
    control_df = cookie_con_df

    demented_audio_df = cookie_dem_audio_df
    control_audio_df = cookie_con_audio_df

    tqdm.pandas(desc="Generating demented embeddings")
    dementia_df['embeddings'] = dementia_df['text'].progress_apply(get_embeddings)
    demented_text_audio = pd.merge(dementia_df, demented_audio_df, on='ID')

    tqdm.pandas(desc="Generating control embeddings")
    control_df['embeddings'] = control_df['text'].progress_apply(get_embeddings)
    control_text_audio = pd.merge(control_df, control_audio_df, on='ID')

    # MERGE DATA
    merged_df = pd.concat([demented_text_audio, control_text_audio], ignore_index=True)
    merged_df = merged_df.drop(columns=['text'])
    merged_df.to_pickle("/home/gabs/Documents/lab-NN/cognitive_impairments/data/DementiaBank/parsed_ADRESSO.pkl")


    
