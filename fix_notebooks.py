import json

for notebook in ['crop_yield_pred.ipynb', 'recommentation_model.ipynb']:
    try:
        with open(notebook, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        changed = False
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                for i, line in enumerate(cell.get('source', [])):
                    if '/content/drive/MyDrive/crop_production.csv' in line:
                        cell['source'][i] = line.replace('/content/drive/MyDrive/crop_production.csv', 'crop_production.csv')
                        changed = True
                    if '/content/Crop_Yield_Prediction.csv' in line:
                        cell['source'][i] = line.replace('/content/Crop_Yield_Prediction.csv', 'Crop_Yield_Prediction.csv')
                        changed = True
        
        if changed:
            with open(notebook, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(f"Updated {notebook}")
    except FileNotFoundError:
        pass
