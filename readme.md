# FireCastRL

FireCastRL is a wildfire decision-support app.

## Forecasting Pipeline

Inferno-Tactix is built on the FireCastRL research pipeline described in the accompanying paper. The wildfire-forecasting stack combines historical ignition records from IRWIN with gridded meteorological sequences from GRIDMET to predict where wildfire ignition is likely to occur.

## Dataset Preparation

The first stage of the FireCastRL pipeline is dataset preparation. Following the methodology described in the paper, we built a spatiotemporal wildfire dataset by combining IRWIN wildfire incident records with GRIDMET meteorological sequences across the continental United States.

The dataset-preparation workflow:

- starts from raw IRWIN wildfire reports and filters them to isolate independent ignition events,

- removes likely duplicate or update records using spatial and temporal filtering,

- constrains samples to the CONUS geographic extent,

- synthesizes negative examples to complement positive ignition events,

- expands each labeled sample into a 75-day meteorological context window,

- joins each sample with GRIDMET weather and fire-weather variables for forecasting.

The resulting public dataset used in this work is available on Kaggle:

- `US Wildfire Dataset`: `https://www.kaggle.com/datasets/firecastrl/us-wildfire-dataset`

At a high level, the pipeline:

- Starts from IRWIN incident records across the continental United States and filters them to isolate spatiotemporally independent ignition events.

- Builds a supervised forecasting dataset by pairing positive ignition samples with synthesized negative samples designed to capture low-risk, seasonally shifted, and historically matched non-ignition cases.

- Expands each sample into a 75-day context window using GRIDMET data at 4 km spatial resolution.

- Uses meteorological and fire-weather variables including precipitation, humidity, vapor pressure deficit, wind speed, temperature, solar radiation, evapotranspiration, burning index, energy release component, and fuel-moisture indicators.

- Trains a deep spatiotemporal forecasting model based on a CNN + Bi-LSTM architecture for wildfire-ignition prediction.

As described in the FireCastRL paper, the resulting dataset contains 126,800 labeled sequences, including 50,720 positive ignition events and 76,080 negative samples, and expands to more than 9.5 million daily labeled data points across the 75-day windows. In the full FireCastRL pipeline, high-risk forecasts can then trigger downstream simulation, reporting, and decision-support workflows.

## RL Wildfire Environment

For suppression training, we built a separate Gymnasium-compatible wildfire environment package that models fire spread over real terrain and supports reinforcement-learning research for helitack response. That environment is maintained in a separate repository: [FireCastRL environment](https://github.com/aisystems-lab/firecast-rl).

The environment package is designed to render a wildfire scenario from real geographic inputs rather than a synthetic grid alone. It incorporates:

- terrain elevation derived from real-world elevation maps,

- landcover / vegetation structure derived from real landcover data,

- wind-driven spread effects using weather inputs sourced through Google Earth Engine,

- a physics-informed fire-spread engine with helicopter-based suppression actions.

Within that environment, the controllable agent is a helitack firefighting helicopter that learns to contain fire growth, reduce burned area, and use suppressant drops effectively. In the FireCastRL pipeline, we trained a PPO policy on this environment for suppression decision-making after a high-risk ignition forecast is produced.

## FireCastBot

FireCastBot is a wildfire-focused assistant for firefighters, emergency responders, non-firefighters, and members of the public. It helps users understand incident reports, doctrine, and practical wildfire-preparedness actions while staying within the wildfire domain.

Key features:

- Incident-report-grounded chat using uploaded PDFs or built-in low-, medium-, and high-risk incident presets.

- Context-aware answers that use incident location, likely spread direction, terrain, weather, and `Overall Risk Level` when those facts are available.

- Doctrine-based wildfire guidance for both operational and public-safety questions.

- Configurable model-backed inference, including Grok as a supported provider.

- Multimodal interaction with speech-to-text (STT), text chat, and browser or provider-backed text-to-speech (TTS).

- Session-based conversations tied to a single incident context until a new session is started.

- Markdown-rendered responses in the web UI for clearer structured answers.

- PDF export / briefing workflow support for concise wildfire reports and summaries.

## Local Development

Bootstrap dependencies:

```bash

./ops/scripts/bootstrap.sh

```

Run the API and web app in separate terminals:

```bash

cd  apps/api && PYTHONPATH=src  python  -m  inferno_api

cd  apps/web && npm  start

```

Local URLs:

- Web: `http://localhost:8080`

- API: `http://localhost:6969`

## Docker Development

```bash

docker  compose  up  --build

```

---

## Citation

If you use FirecastRL in your research, please cite:

```bibtex
@software{firecastrl2025,
  title={Spatiotemporal Wildfire Prediction and Reinforcement Learning for Helitack Suppression},
  author={Shaurya Mathur, Shreyas Bellary Manjunath, Nitin Kulkarni, Alina Vereshchaka},
  year={2025},
  url={https://sites.google.com/view/firecastrl},
  doi={10.48550/arXiv.2601.14238}
}
```

Paper DOI: https://doi.org/10.48550/arXiv.2601.14238

---

## Contact

**Shaurya Mathur** - shauryamathur2001@gmail.com
**Shreyas Bellary Manjunath** - sbellary@buffalo.edu

---
