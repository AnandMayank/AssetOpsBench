#!/usr/bin/env bash
# Set up the gauge_reader conda environment for running the ETH analog gauge reader.
#
# Creates env at /media/adityapachauri/second_drive/envs/gauge_reader (Python 3.8)
# to preserve space on the primary conda drive.
#
# Usage:
#   bash src/perception/setup_eth_env.sh
#   conda activate /media/adityapachauri/second_drive/envs/gauge_reader

set -euo pipefail

CONDA_BIN="/media/adityapachauri/a93669a1-5154-48cd-91ef-105c3fceb0d7/miniconda3/bin/conda"
ENV_PREFIX="/media/adityapachauri/second_drive/envs/gauge_reader"
ETH_DIR="/home/adityapachauri/analog_gauge_reader"

echo "==> Creating conda env at $ENV_PREFIX (Python 3.8)"
$CONDA_BIN create --prefix "$ENV_PREFIX" python=3.8 -y

echo "==> Activating env"
# shellcheck disable=SC1091
source "$($CONDA_BIN info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "==> Installing PyTorch 2.0.0"
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
    -c pytorch -c nvidia -y

echo "==> Installing OpenMMLab stack"
pip install -U openmim
mim install mmengine==0.7.2
mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmocr==1.0.0

echo "==> Installing ultralytics and sklearn"
pip install "ultralytics==8.0.66" "scikit-learn==1.2.2"

echo "==> Verifying"
python -c "import mmdet; import mmocr; import ultralytics; print('All ETH deps OK')"

echo ""
echo "==> Pulling ETH model weights (git lfs)"
cd "$ETH_DIR" && git lfs pull
ls -lh models/

echo ""
echo "Setup complete. To run the ETH pipeline:"
echo "  conda activate $ENV_PREFIX"
echo "  cd $ETH_DIR"
echo "  python pipeline.py \\"
echo "    --input /media/adityapachauri/second_drive/syncg_data/syncG/images/test/ \\"
echo "    --base_path /home/adityapachauri/AssetOpsBench/src/perception/eth_results/ \\"
echo "    --detection_model models/gauge_detection_model.pt \\"
echo "    --key_point_model models/key_point_model.pt \\"
echo "    --segmentation_model models/segmentation_model.pt"
