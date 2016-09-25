mkdir -p features
#python code/keystroke-feature-extractor/main.py extract --features linguistic data/keystroke2.csv features/linguistic.tmp
#cat features/linguistic.tmp | tr ',' '\t' > features/linguistic
#rm features/linguistic.tmp
#python code/keystroke-feature-extractor/main.py extract --features letters data/keystroke2.csv features/letters.tmp
#cat features/letters.tmp | tr ',' '\t' > features/letters
#rm features/letters.tmp
#python src/get-text.py > features/text

## split into train/dev/test
python src/create-data.py features/linguistic
python src/create-data.py features/letters
python src/create-data.py features/text


cut -f3 features/text > tmp
paste features/linguistic tmp > features/linguistic+text
paste features/letters tmp > features/letters+text

python src/create-data.py features/linguistic+text
python src/create-data.py features/letters+text


