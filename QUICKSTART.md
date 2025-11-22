# 🚀 Quick Start Guide - GPU Training

## En Hızlı Yol (3 Adım)

### 1. GPU Node Al
```bash
srun --partition=cuda --qos=cuda --gres=gpu:1 --time=8:00:00 --mem=64G --pty bash
```

### 2. Otomatik Setup
```bash
bash setup_gpu.sh
```

### 3. Training Başlat
```bash
# Dataset'i data/ klasörüne koyduktan sonra:
bash run_training.sh codebert data/train.csv data/val.csv
```

## Detaylı Adımlar

### Adım 1: GPU Node ve Ortam
```bash
# GPU node al
srun --partition=cuda --qos=cuda --gres=gpu:1 --time=8:00:00 --mem=64G --pty bash

# Modülleri yükle
module load miniconda3/22.11.1-oneapi-2024.0.2-vdx5rot
module load cuda/10.2.89-gcc-8.5.0-h3fatfr

# Environment oluştur/aktif et
conda create -n semeval python=3.10 -y
conda activate semeval

# Proje dizinine git
cd /cta/users/ide.yilmaz/Sem-eval-task-13

# Dependencies yükle
pip install -r requirements.txt
```

### Adım 2: Dataset Hazırla
```bash
# Dataset'i data/ klasörüne koy:
# - data/train.csv
# - data/val.csv
# - data/test.csv (opsiyonel)

# Veya download scriptini kullan:
python download_data.py --data_dir data
```

### Adım 3: (Opsiyonel) Zaman Tahmini
```bash
python estimate_time.py \
    --model_name codebert \
    --train_data data/train.csv \
    --val_data data/val.csv
```

### Adım 4: Training Başlat

**Seçenek A: Background (nohup) - Önerilen**
```bash
bash run_training.sh codebert data/train.csv data/val.csv
```

**Seçenek B: Interactive**
```bash
python train.py \
    --model_name codebert \
    --train_data data/train.csv \
    --val_data data/val.csv \
    --save_every_epoch \
    --cache_dir ./cache \
    --log_dir ./logs
```

**Seçenek C: Tüm Modeller**
```bash
bash run_all_models_server.sh
```

### Adım 5: Progress Kontrol
```bash
# Yeni terminal aç ve:
tail -f logs/training_codebert_*.out
tail -f logs/codebert_*.log
```

### Adım 6: Evaluation
```bash
python evaluate.py \
    --model_path models/codebert/best_model \
    --test_data data/test.csv \
    --output_file results/codebert_results.json
```

## Önemli Notlar

- ✅ **Validation set** training sırasında kullanılıyor (test set değil!)
- ✅ Her epoch sonunda **checkpoint** kaydediliyor
- ✅ Preprocessed data **cache'leniyor** (ilk run yavaş, sonrakiler hızlı)
- ✅ Tüm loglar `logs/` klasörüne kaydediliyor
- ✅ Best model `models/{model_name}/best_model/` altında

## Model Seçenekleri

- `codebert` - Microsoft CodeBERT
- `graphcodebert` - Microsoft GraphCodeBERT
- `codet5` - Salesforce CodeT5
- `starcoder` - BigCode StarCoder
- `distilbert` - DistilBERT

## Sorun Giderme

**GPU bulunamıyor:**
```bash
nvidia-smi  # GPU var mı kontrol et
```

**Conda environment bulunamıyor:**
```bash
conda env list  # Mevcut environment'ları listele
conda activate semeval  # Veya mevcut environment'ı aktif et
```

**Dataset bulunamıyor:**
```bash
ls -la data/  # Dataset var mı kontrol et
```

**Training durdu:**
```bash
ps aux | grep train.py  # Process hala çalışıyor mu?
tail -f logs/training_*.out  # Son logları kontrol et
```

