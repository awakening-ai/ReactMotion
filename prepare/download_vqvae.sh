mkdir -p external
cd external

echo "The pretrained_vqvae will be stored in the './external' folder"
echo "Downloading"
gdown "https://drive.google.com/uc?id=1tOw9wiu6jkzBy-bLe2iy47KAjE50DgjP"

echo "Extracting"
unzip pretrained_vqvae.zip

echo "Cleaning"
rm pretrained_vqvae.zip

echo "Downloading done!"