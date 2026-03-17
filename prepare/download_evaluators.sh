mkdir -p external
cd external

echo -e "Downloading extractors"
gdown --fuzzy https://drive.google.com/file/d/1FIiqtkt4F-GVWmnBgtZnv9W3cPWS-oM-/view


unzip t2m.zip

echo -e "Cleaning\n"
rm t2m.zip
echo -e "Downloading done!"