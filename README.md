## ⚙️ Getting Started
To get started with `LegalSearch-R1`, follow these steps:

### Package Installation
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Setup API Keys
Edit the configuration files `./user/tools/config.yaml` to include your OpenAI API key and Serper API key.
```
openai_api_key: "your_openai_api_key"
serper_api_key: "your_serper_api_key"
```
For recording the experimental logs, set the Wandb API key in your `./user/train.sh` and `./user/evaluate.sh` and `./user/evaluate_ood.sh` scripts:
```bash
export WANDB_API_KEY="your_wandb_api_key"
```

### Data Preparation
If you want to customize the dataset, please modify `./user/legalsearch_data_process.py` and run the data processing script:
```bash
python ./user/legalsearch_data_process.py
```

## 🚀 Training
To train the LegalSearch-R1 model, run the training script:
```bash
bash ./user/train.sh
```

## 🔬 Evaluation
To evaluate the trained model, use the evaluation script:
```bash
bash ./user/evaluate.sh
```
To evaluate the model on out-of-domain data, use the following script:
```bash
bash ./user/evaluate_ood.sh
```

## 🌍 Citation
If you find this work useful in your research, please consider citing:
