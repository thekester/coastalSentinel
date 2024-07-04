import runpy
import sys
from pathlib import Path

# Définir le chemin vers le script Python
script_path = Path("autolabeler4.py").resolve()

# Définir les arguments comme s'ils étaient passés en ligne de commande
sys.argv = [
    str(script_path),
    "--img-height", "150",
    "--img-width", "150",
    "--batch-size", "32",
    "--epochs", "20",
    "--base-dir", "classification",
    "--model-path", "best_segmentation_model.keras",
    "--continue-training"
]

# Exécuter le script avec les arguments
runpy.run_path(script_path)
