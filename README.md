# Speaker Verification 

This tool provides functionality for processing .wav audio files for speaker verification. It includes resampling audio files to a consistent frame rate, running a speaker verification model to generate embeddings, and then clustering these embeddings based on their cosine similarity. This can be used to identify similar speakers or verify speaker identity in a collection of audio files.

## Getting Started

### Prerequisites

Before running the tool, ensure you have the following prerequisites installed:

- Python 3.8
- [3D-Speaker](https://github.com/alibaba-damo-academy/3D-Speaker) library
- Necessary Python packages

### Installation

Follow these steps to set up your environment and install required libraries:

1. **Clone the 3D-Speaker Repository:**
    ```
    git clone https://github.com/alibaba-damo-academy/3D-Speaker.git && cd 3D-Speaker
    ```

2. **Create and Activate a Conda Environment:**
    ```
    conda create -n 3D-Speaker python=3.8
    conda activate 3D-Speaker
    ```

3. **Install Dependencies for 3D-Speaker:**
    ```
    pip install -r /content/3D-Speaker/requirements.txt
    ```

4. **Install Additional Requirements for This Tool:**
    ```
    pip install -r requirements.txt
    ```
    *Note: Ensure you have a `requirements.txt` file in your project directory with all the necessary packages.*

### Usage

To run the voice verification script, you need to provide the following arguments:

1. The path to the directory containing your .wav files (--wav_directory).
2. The path to the directory where the resampled .wav files will be stored (--resampled_directory).
3. The path to the directory containing embeddings, or use the default path (--embedding_directory), which is set to /content/pretrained/speech_campplus_sv_en_voxceleb_16k/embeddings/.
4. The model ID for speaker verification (--model_id).
5. Optionally, you can set the threshold for speaker similarity using the --threshold parameter (default: 0.6).

```bash
!python /content/voice_verification_v2.py --wav_directory /content/wavs --resampled_directory /content/resampled --embedding_directory /content/pretrained/speech_campplus_sv_en_voxceleb_16k/embeddings --model_id 'damo/speech_campplus_sv_en_voxceleb_16k'
```