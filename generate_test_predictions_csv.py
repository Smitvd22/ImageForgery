import csv
import os
from core.config import *
from predict_optimized import OptimizedPredictor

def main():
    input_csv = './data/imsplice_labels.csv'
    output_csv = './results_imsplice/test_predictions.csv'
    predictor = OptimizedPredictor(dataset='imsplice')
    predictor.load_models()

    with open(input_csv, 'r', newline='') as infile, open(output_csv, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['filepath', 'actual_label', 'predicted_label']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            filepath = row['filepath']
            actual_label = 'Authentic' if row['label'] == '0' else 'Forged'
            pred, _ = predictor.predict(filepath)
            predicted_label = 'Authentic' if pred == 0 else 'Forged'
            writer.writerow({'filepath': filepath, 'actual_label': actual_label, 'predicted_label': predicted_label})

if __name__ == '__main__':
    main()
