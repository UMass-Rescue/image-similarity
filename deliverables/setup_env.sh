conda install pip

conda install --yes --file requirements.txt

echo "Requirements installed"

pip install torchvision==0.2.2

echo "Torchvision installed"

pip install image_similarity-0.1-py3-none-any.whl

echo "Image similarity module installed"

python -m ipykernel install --user --name cs-696-image_similarity --display-name "Python (cs-696-image_similarity)"

python create_db.py
