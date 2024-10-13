from flask import Blueprint, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Crea un blueprint per la pulizia dei dati
clean_data_bp = Blueprint('clean_data', __name__)

# Inizializza Spark
spark = SparkSession.builder \
    .appName("DataCleaningApp") \
    .getOrCreate()

@clean_data_bp.route('/clean_data', methods=['POST'])
def clean_data():
    # Recupera i dati dalla richiesta
    data = request.get_json()

    if not data or 'raw_data' not in data:
        return jsonify({"error": "Dati grezzi non forniti."}), 400

    raw_data = data['raw_data']

    # Crea un DataFrame a partire dai dati grezzi
    raw_data_lines = raw_data.splitlines()
    rdd = spark.sparkContext.parallelize(raw_data_lines)
    
    # Consideriamo che i dati abbiano un header
    header = rdd.first()
    df = rdd.filter(lambda line: line != header).map(lambda line: line.split(",")) \
             .toDF(header.split(","))

    # Operazioni di pulizia
    cleaned_df = clean_dataframe(df)

    # Converti il DataFrame pulito in un formato JSON
    cleaned_data = cleaned_df.toJSON().collect()

    return jsonify({"clean_data": cleaned_data})

def clean_dataframe(df):
    # Esegui operazioni di pulizia sui dati
    # 1. Rimuovi le righe con valori nulli
    df = df.dropna()

    # 2. Rimuovi i duplicati
    df = df.dropDuplicates()

    # 3. Normalizza i nomi delle colonne (rimuovi spazi e metti in minuscolo)
    df = df.select([col(column).alias(column.strip().lower()) for column in df.columns])

    # 4. Esempio di normalizzazione: convertire una colonna specifica in un formato
    # Modifica 'your_column' con il nome reale della colonna
    # df = df.withColumn('your_column', col('your_column').cast('double'))

    # 5. Altre operazioni di pulizia possono essere aggiunte qui

    return df
