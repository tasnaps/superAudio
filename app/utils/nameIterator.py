import os
import glob
from collections import defaultdict

hq_dir = "C:/Users/tapio/PycharmProjects/superAudio/app/storage/HighQualityAudios"
pq_dir = "C:/Users/tapio/PycharmProjects/superAudio/app/storage/processed_audios"

# get list of all high-quality .flac audio paths
hq_audio_paths = sorted(glob.glob(os.path.join(hq_dir, '*.flac')))

# Rename HQ files and create a mapping dictionary with old and new names
song_name_mapping = {}
for i, hq_path in enumerate(hq_audio_paths, start=1):
    old_name = os.path.basename(hq_path)
    print(old_name)
    new_name = f"{i:02}"
    print(new_name)

    # rename HQ file
    new_path = os.path.join(hq_dir, new_name)
    os.rename(hq_path, new_path + ".flac")
    song_name_mapping[old_name[:old_name.index(".")]] = new_name

# get list of all processed .wav audio paths
pq_audio_paths = sorted(glob.glob(os.path.join(pq_dir, '*.wav')))

# Create a dictionary to hold filename prefixes and their count
prefix_count = defaultdict(int)

# Rename PQ files
for pq_path in pq_audio_paths:
    old_name = os.path.basename(pq_path)
    old_prefix = old_name[:old_name.index(".")]
    new_prefix = song_name_mapping.get(old_prefix, None)

    if new_prefix:
        prefix_count[new_prefix] += 1
        new_name = f"{new_prefix}-{prefix_count[new_prefix]}"

        # rename PQ file
        new_path = os.path.join(pq_dir, new_name)
        os.rename(pq_path, new_path + ".wav")
    else:
        print(f"Warning: Couldn't find high-quality audio that matches '{old_name}'")
