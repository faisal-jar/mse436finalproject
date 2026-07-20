# Chart Compass, an IDSS for release single strategy

**Course:** MSE 436: Decision Support Systems
**Stakeholder:** an Artists and Repertoire team / marketing lead at a record label
**Decision:** *how much promotional budget to commit behind a
candidate single, and through which channels, before it is released.*

Chart Compass predicts a track's likely Billboard chart tier from its audio
profile and the artist's track record, then converts that prediction into an
**adjustable release decision**: a recommended budget, a channel
split, and the expected value of committing that spend.

---

## How to Start

```bash
# 1. install
pip install -r requirements.txt

# 2. generate the dataset using Billboard & Spotify data from Kaggle
python data/prepare_data.py

# 3. train the model (writes models/chart_model.joblib + model_meta.json)
python -m src.train_model

# 4. launch the IDSS
streamlit run app/streamlit_app.py
```

> **Data Note:** Spotify deprecated the audio-features API for new apps on
> 27 Nov 2024. Ongoing feature collection requires re-derived audio
> features (librosa / Essentia) or a hosted alternative, which is why the system
> is built around **periodic retraining** rather than a live Spotify pull.

AID Statement: 
Artificial Intelligence Tool: Claude Opus 4.8 used July 2026; Execution: Implemented code to build bulk of app interface based on existing ML model created by author; Data Curation: Merged data files from Kaggle as needed to work for model training.
