import os
import subprocess
import locale

locale.getpreferredencoding = lambda: "UTF-8"

def download(lang, tgt_dir="./datasets/"):
  if not os.path.exists(tgt_dir):
    os.makedirs(tgt_dir)

  lang_fn, lang_dir = os.path.join(tgt_dir, lang + '.tar.gz'), os.path.join(tgt_dir, lang)

  # Extract file ID from Google Drive share link
  file_id = "1OgYbPfXFAv3TbwP1Qcwt_CC9cVWSJaco"  # Extracted from the URL

  # Download and extract directly in target directory
  cmd = " && ".join([
    f"gdown https://drive.google.com/uc?id={file_id} -O {lang_fn}",
    f"tar zxvf {lang_fn} -C {tgt_dir}"
  ])

  print(f"Downloading model for language: {lang}")
  subprocess.check_output(cmd, shell=True)
  print(f"Model checkpoints in {lang_dir}: {os.listdir(lang_dir)}")
  return lang_dir

LANG = "indosum"
ckpt_dir = download(LANG)