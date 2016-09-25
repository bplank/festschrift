###
echo "basic keystroke feats"
python src/classifier.py features/letters.train features/letters.dev features/letters.test
echo "+++++++"
echo "extended keystroke feats"
python src/classifier.py features/linguistic.train features/linguistic.dev features/linguistic.test
echo "========"
python src/classifier.py features/letters+text.train features/letters+text.dev features/letters+text.test  --embeds
python src/classifier.py features/linguistic+text.train features/linguistic+text.dev features/linguistic+text.test  --embeds

python src/classifier.py features/letters+text.train features/letters+text.dev features/letters+text.test --ngrams 1
python src/classifier.py features/linguistic+text.train features/linguistic+text.dev features/linguistic+text.test --ngrams 1 