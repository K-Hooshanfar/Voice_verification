"""
This script provides functionality for processing .wav audio files for speaker verification.
It includes resampling audio files to a consistent frame rate, running a speaker verification
model to generate embeddings, and then clustering these embeddings based on their cosine similarity.
This can be used to identify similar speakers or verify speaker identity in a collection of audio files.
"""
import argparse
import os
import subprocess

import numpy as np
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity


def resample_wav_files(wav_directory, resampled_directory):
    """
    Resample WAV files to a consistent frame rate of 16kHz.

    Args:
        wav_directory (str): Directory containing original .wav files.
        resampled_directory (str): Directory where resampled .wav files will be stored.

    Returns:
        str: Path to the file containing the list of resampled .wav files.
    """
    wav_files = [os.path.join(wav_directory, file) for file in os.listdir(wav_directory) if file.endswith('.wav')]
    os.makedirs(resampled_directory, exist_ok=True)
    local_wav_list_path = os.path.join(resampled_directory, 'wav_list.txt')
    with open(local_wav_list_path, 'w') as f:
        for wav_file in wav_files:
            audio = AudioSegment.from_wav(wav_file).set_frame_rate(16000)
            resampled_file = os.path.join(resampled_directory, os.path.basename(wav_file))
            audio.export(resampled_file, format="wav")
            f.write(f"{resampled_file}\n")
    return local_wav_list_path


def run_speaker_verification(model_id, wav_list_path):
    """
    Run the speaker verification process using a specified model.

    Args:
        model_id (str): Identifier for the speaker verification model.
        wav_list_path (str): Path to the file containing the list of .wav files for verification.
    """
    subprocess.run(["python", "3D-Speaker/speakerlab/bin/infer_sv.py", "--model_id", model_id, "--wavs", wav_list_path])


def compare_embeddings(embedding_directory, threshold=0.6):
    """
    Compare embeddings and cluster them based on a cosine similarity threshold.

    Args:
        embedding_directory (str): Directory containing the speaker embeddings.
        threshold (float): Cosine similarity threshold for clustering.

    Returns:
        None: Clusters are printed to stdout.
    """
    os.makedirs(embedding_directory, exist_ok=True)
    embedding_files = [os.path.join(embedding_directory, file) for file in os.listdir(embedding_directory) if
                       file.endswith('.npy')]
    embeddings = [np.load(file).reshape(1, -1) for file in embedding_files]

    clusters = {}
    next_cluster_id = 0

    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarity_score = cosine_similarity(embeddings[i], embeddings[j])[0][0]

            if similarity_score >= threshold:
                file_i = os.path.basename(embedding_files[i])
                file_j = os.path.basename(embedding_files[j])
                cluster_id_i = next((k for k, v in clusters.items() if file_i in v), None)
                cluster_id_j = next((k for k, v in clusters.items() if file_j in v), None)

                if cluster_id_i is None and cluster_id_j is None:
                    # Neither file is in a cluster yet, create a new one
                    clusters[next_cluster_id] = {file_i, file_j}
                    next_cluster_id += 1
                elif cluster_id_i is not None and cluster_id_j is None:
                    # File i is in a cluster, add file j to it
                    clusters[cluster_id_i].add(file_j)
                elif cluster_id_i is None and cluster_id_j is not None:
                    # File j is in a cluster, add file i to it
                    clusters[cluster_id_j].add(file_i)
                elif cluster_id_i != cluster_id_j:
                    # Both files are in different clusters, merge them
                    clusters[cluster_id_i].update(clusters[cluster_id_j])
                    del clusters[cluster_id_j]

    # Print the clusters
    for cluster_id, files in clusters.items():
        print(f"Cluster {cluster_id}: {', '.join(files)}")


if __name__ == "__main__":
    """
    Main script for running speaker verification and clustering.

    This script takes input parameters for directories and model ID, resamples WAV files,
    runs speaker verification, and clusters the resulting embeddings based on similarity.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_directory', type=str, required=True, help='Directory containing .wav files')
    parser.add_argument('--resampled_directory', type=str, required=True,
                        help='Directory for storing resampled .wav files')
    parser.add_argument('--embedding_directory', type=str,
                        default='/content/pretrained/speech_campplus_sv_en_voxceleb_16k/embeddings/',
                        help='Directory containing embeddings')
    parser.add_argument('--model_id', type=str, required=True, help='Model ID for speaker verification')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for speaker similarity')

    args = parser.parse_args()

    wav_list_path1 = resample_wav_files(args.wav_directory, args.resampled_directory)
    run_speaker_verification(args.model_id, wav_list_path1)
    compare_embeddings(args.embedding_directory, args.threshold)
