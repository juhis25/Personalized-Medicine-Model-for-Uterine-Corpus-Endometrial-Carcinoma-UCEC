import pandas as pd
import requests
import os

def download_tcga_data(project, data_type, output_dir):
    url = f"https://api.gdc.cancer.gov/data?filters={{%22op%22:%22and%22,%22content%22:[{{%22op%22:%22in%22,%22content%22:{{%22field%22:%22cases.project.project_id%22,%22value%22:[%22{project}%22]}}}},{{%22op%22:%22in%22,%22content%22:{{%22field%22:%22files.data_type%22,%22value%22:[%22{data_type}%22]}}}}]}}&pretty=true&size=1000&format=TSV"
    response = requests.get(url)
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{project}_{data_type}.tsv", 'wb') as f:
        f.write(response.content)

project = "TCGA-UCEC"
data_type = "Gene Expression Quantification"
output_dir = "../data"
download_tcga_data(project, data_type, output_dir)
