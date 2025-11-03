from pyspark import SparkConf, SparkContext
import re
import nltk
import chardet
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# === НАСТРОЙКА SPARK ===
conf = SparkConf().setAppName("GertsenAnalysis").setMaster("local[*]")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# === ФАЙЛ ===
file_path = "Герцен_Кто_виноват.txt"  # Положи рядом с этим .py файлом!

# === ОПРЕДЕЛЕНИЕ КОДИРОВКИ ===
with open(file_path, "rb") as f:
    raw_data = f.read(20000)
    enc = chardet.detect(raw_data)["encoding"]
print(f"Определена кодировка файла: {enc}")

# === ЧТЕНИЕ ТЕКСТА ===
with open(file_path, "r", encoding=enc, errors="ignore") as f:
    text_data = f.readlines()

text_rdd = sc.parallelize(text_data)

# === СТОП-СЛОВА ===
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("russian"))
stop_words.update({"это", "еще", "очень", "весь", "такой", "который", "—", "«", "»", "..."})

# === ОЧИСТКА ТЕКСТА ===
def clean_text(line):
    line = re.sub(r"[^а-яё\-]", " ", line.lower())  # оставляем только буквы и дефис
    line = re.sub(r"\s+", " ", line).strip()
    words = line.split()
    return [w for w in words if w not in stop_words and len(w) > 2]

cleaned_words = text_rdd.flatMap(clean_text).cache()

# === СТАТИСТИКА ===
total_words = cleaned_words.count()
unique_words = cleaned_words.distinct().count()
print(f"\nПосле очистки: {total_words} слов, {unique_words} уникальных\n")

# === WORDCOUNT ===
word_counts = cleaned_words.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)

top_50 = word_counts.takeOrdered(50, key=lambda x: -x[1])
least_50 = word_counts.filter(lambda x: x[1] >= 2).takeOrdered(50, key=lambda x: x[1])

print("ТОП-50 наиболее частых слов:")
for w, c in top_50:
    print(f"{w:<15} → {c}")

print("\nТОП-50 наименее частых (≥2 раза):")
for w, c in least_50:
    print(f"{w:<15} → {c}")

# === СТЕММИНГ ===
stemmer = SnowballStemmer("russian")
stemmed_rdd = cleaned_words.map(lambda w: stemmer.stem(w))
stemmed_counts = stemmed_rdd.map(lambda w: (w, 1)).reduceByKey(lambda a, b: a + b)

top_50_stemmed = stemmed_counts.takeOrdered(50, key=lambda x: -x[1])
least_50_stemmed = stemmed_counts.filter(lambda x: x[1] >= 2).takeOrdered(50, key=lambda x: x[1])

print("\nТОП-50 после стемминга:")
for w, c in top_50_stemmed:
    print(f"{w:<12} → {c}")

print("\nТОП-50 редких после стемминга:")
for w, c in least_50_stemmed:
    print(f"{w:<12} → {c}")

# === ИТОГИ ===
total_stemmed = stemmed_rdd.count()
unique_stemmed = stemmed_rdd.distinct().count()
print(f"\nПосле стемминга: {total_stemmed} слов, {unique_stemmed} уникальных")
print("Уникальных слов стало на", unique_words - unique_stemmed, "меньше!")

sc.stop()
print("\nГотово! Spark остановлен.")
