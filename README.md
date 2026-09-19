# AI Personal Health Assistant

A 3-module Streamlit health assistant for:

- Symptoms Prediction
- Chest X-ray Pneumonia Detection
- Skin Cancer Detection

## Project Files

- `app.py` - Main Streamlit application
- `Symptoms.ipynb` - Training and testing notebook
- `requirements.txt` - Python dependencies
- `dataset.csv` - Symptoms prediction dataset with 4,920 records
- `symptom_Description.csv` - Disease descriptions for 41 diseases
- `symptom_precaution.csv` - Precautions for 41 diseases
- `Symptom-severity.csv` - 133 symptom severity values

## Datasets Used

The large image datasets are not committed to GitHub. Download them separately and place them in this structure:

```text
chest_xray/
  train/
  test/
  val/

skin_cancer/
  HAM10000_metadata.csv
  HAM10000_images_part_1/
  HAM10000_images_part_2/
```

Chest X-ray module dataset:

- `chest_xray/train` - 5,216 training images
- `chest_xray/test` - 624 testing images

Skin Cancer module dataset:

- `skin_cancer/HAM10000_metadata.csv`
- `skin_cancer/HAM10000_images_part_1`
- `skin_cancer/HAM10000_images_part_2`
- Total skin lesion images: 10,015

## Dataset Links

Chest X-ray Dataset:

```text
https://drive.google.com/drive/folders/12Gd2aoQ3Koo8IcSb8vGlPKzPbiFHfoQ5?usp=sharing
```

Skin Cancer Dataset:

```text
https://drive.google.com/drive/folders/12Gd2aoQ3Koo8IcSb8vGlPKzPbiFHfoQ5?usp=sharing
```

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
