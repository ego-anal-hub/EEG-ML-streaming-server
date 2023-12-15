from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,from_json,regexp_replace,expr,to_timestamp,udf,window,collect_list,date_format,struct,to_json
from pyspark.sql.types import *
import numpy as np
from scipy import signal
from joblib import load

import findspark
findspark.init()

# SparkSession 생성
# spark = SparkSession.builder \
#     .appName("EEG-Analysis-SparkStreaming") \
#     .master("local[*]") \
#     .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") \
#     .config("spark.sql.shuffle.partitions", "4") \
#     .config("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false") \
#     .conf.set("spark.sql.execution.arrow.enabled", "true") \
#     .getOrCreate()

conf = SparkConf()
conf.set("spark.app.name", "EEG-Analysis-State-Reasonal-SparkStreaming")
conf.set("spark.master", "local[*]")
conf.set("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0")
conf.set("spark.sql.shuffle.partitions", "4")
conf.set("spark.sql.streaming.statefulOperator.checkCorrectness.enabled", "false")
conf.set("spark.sql.execution.arrow.enabled", "true")
# conf.set("spark.driver.memory", "2g")  # Increase this as needed
# conf.set("spark.executor.memory", "2g")  # Increase this as needed

spark = SparkSession.builder.config(conf=conf).getOrCreate()

# Kafka에서 스트림 데이터 읽기
kafka_df = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka_ip:9092") \
    .option("subscribe", "raw-eeg") \
    .option('client_id', 'eeg-da-server-consumer-2') \
    .option('group_id', 'da-group') \
    .load() \
    .selectExpr("CAST(value AS STRING)")

eeg_schema = StructType([
    StructField("time", StringType()),
    StructField("space", StructType([
        StructField("id", StringType())
    ])),
    StructField("person", StructType([
        StructField("id", StringType()),
        StructField("eeg", StructType([
            StructField("ch1", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ])),
            StructField("ch2", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ])),
            StructField("ch3", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ])),
            StructField("ch4", StructType([
                StructField("pos", StringType()),
                StructField("value", FloatType())
            ]))
        ]))
    ]))
])

eeg_data = kafka_df.select(from_json(regexp_replace(col("value"), '\\\\|(^")|("$)', ''), eeg_schema).alias("eeg_data")) \
    .select(
        col("eeg_data.time").alias("time"),
        col("eeg_data.space.id").alias("spaceId"),
        col("eeg_data.person.id").alias("personalId"),
        col("eeg_data.person.eeg.ch1.value").alias("ch1_value"),
        col("eeg_data.person.eeg.ch2.value").alias("ch2_value"),
        col("eeg_data.person.eeg.ch3.value").alias("ch3_value"),
        col("eeg_data.person.eeg.ch4.value").alias("ch4_value")
    )

# 타임스탬프를 Spark의 'timestamp' 형식으로 변환
eeg_data = eeg_data.withColumn("time", 
                   expr("concat(substr(time, 1, 4), '-', substr(time, 5, 2), '-', substr(time, 7, 2), ' ', substr(time, 9, 2), ':', substr(time, 11, 2), ':', substr(time, 13, 2), '.', substr(time, 15, 3))"))
eeg_data = eeg_data.withColumn("time", to_timestamp(col("time"), "yyyy-MM-dd HH:mm:ss.SSS"))

@udf(returnType=ArrayType(IntegerType()))
def udf_reasonal(ch1_values, ch2_values, ch3_values, ch4_values):
    channels = np.array([ch1_values, ch2_values, ch3_values, ch4_values])
    # 주파수 대역 선택
    #sampling_rate = 256  # 샘플링 주파수 (Hz)
    #frequency_resolution = sampling_rate / len(values)  # 주파수 해상도

    domain_fband_data = {
        'delta' : {
            'power' : float,
            'range' : [0.5, 4], # 연수,뇌교,중뇌 / 무의식 / 깊은수면,내면의식,수면,무의식,기본활동,회복,재생 / RGB (0, 0, 100) / 0
            'rgb_ratio' : (0, 0, 100)
        },
        'theta' : {
            'power' : float,
            'range' : [4, 8], # 구피질 / 내적의식 / 졸음,얕은수면,창의력,상상력,명상,꿈,감정,감성,예술적노력 / RGB (255, 215, 0)
            'rgb_ratio' : (255, 215, 0)
        },
        'slow-alpha' : {
            'power' : float,
            'range' : [8, 9], # 내적의식 / 명상,무념무상
            'rgb_ratio' : None
        },
        'middle-alpha' : {
            'power' : float,
            'range' : [10, 12], # 내적의식 / 지감,번득임,문제해결,스트레스해소,학습능률향상,기억력,집중력최대
            'rgb_ratio' : None
        },
        'smr' : {
            'power' : float,
            'range' : [12, 15], # 후방신피질 / 내적의식 / 각성,일학업능률최적,주의집중,약간의긴장,간단한집중,수동적두뇌활동
            'rgb_ratio' : None
        },
        'alpha' : {
            'power' : float,
            'range' : [8, 13], # 후두엽 / 내적의식 / 휴식,긴장해소,집중력향상,안정감,근육이완,눈뜬상태과도->과거미래환상 / RGB (0, 255, 255)
            'rgb_ratio' : (0, 255, 255)
        },
        'beta' : {
            'power' : float,
            'range' : [14, 30], # 눈감았을때측두엽,떴을때전두엽 / 외적의식 / 각성,인지활동,집중,의식적사고,육체활동,몰두,복잡한업무 / RGB (255, 192, 203)
            'rgb_ratio' : (255, 192, 203)
        },
        'high-beta' : {
            'power' : float,
            'range' : [18, 30], # 외적의식 / 불안,긴장,경직
            'rgb_ratio' : None
        },
        'gamma' : {
            'power' : float,
            'range' : [30, 100], # 전두엽,두정엽 / 외적의식 / 문제해결흥분,고급인지기능,불안,순간인지,능동적복합정신기능 / RGB (128, 0, 128)
            'rgb_ratio' : (128, 0, 128)
        }
    }
    
    def get_powers(channel, FS=256):
        # channel = channel - np.mean(channel)
        freq, psd = signal.periodogram(channel, fs=FS, nfft=256)

        for band_name, band_info in domain_fband_data.items():
            low, high = band_info['range']
            tmp = psd[(freq >= low) & (freq < high)]
            if len(tmp) == 0: continue

            min_val = np.min(tmp)
            max_val = np.max(tmp)

            if (max_val - min_val) == 0 : domain_fband_data[band_name]['power'] = 0
            # Apply min-max normalization
            # else: domain_fband_data[band_name]['power'] = len(psd)
            else: domain_fband_data[band_name]['power'] = np.mean([(x - min_val) / (max_val - min_val) for x in tmp])

    max_bands = []
    for ch_values in channels:
        get_powers(ch_values)
        max_key = max(domain_fband_data, key=lambda k: domain_fband_data[k]['power'])
        tmp = list(domain_fband_data.keys()).index(max_key) - 5
        if tmp > 0: tmp += 2
        max_bands.append(abs(tmp))

    activity = int(sum(max_bands) / len(max_bands))
    # activity = int(sum(max_bands))

    # y_pred = []
    # for ch,ch_values in enumerate([ch2_values, ch3_values]):
    #     saxvsm = load('../models/SAX-VSM/saxvsm_model_ch' + str(ch + 2) + '.joblib')
    #     y_pred.append(saxvsm.predict(ch_values))
    # concentration = sum(y_pred) + 2
    if len(ch2_values) >= 256:
        data = ch2_values[:len(ch2_values) - (len(ch2_values) % 256)]
        data = np.array(data).reshape(1, -1)
        saxvsm = load('../models/SAX-VSM/saxvsm_model_ch2.joblib')
        concentration = int(saxvsm.predict(data)[0])
    else: concentration = 0

    # concentration{집중: 1, 보통: 0, 이완: -1}
    # activity{0~5}
    return [concentration, activity]

# 사용자 정의 함수로 변환
# udf_apply_fft = udf(udf_reasonal, StringType())
spark.udf.register("udf_reasonal", udf_reasonal)

window_duration = "10 seconds"
slide_duration = "5 seconds"
# .withWatermark("time", "1 minutes")
eeg_data_windowed = eeg_data.withWatermark("time", "1 microseconds").withColumn(
    "window",
    window(col("time"), window_duration, slide_duration)
).groupBy(
    "window", "spaceId", "personalId"
).agg(
    udf_reasonal(collect_list(col("ch1_value")), collect_list(col("ch2_value")), collect_list(col("ch3_value")), collect_list(col("ch4_value"))).alias("result")
)

result_data = eeg_data_windowed.select(
    date_format(col("window.end"), "yyyyMMddHHmmssSSS").alias("time"),
    # col("window").alias("timestamp"),
    struct(
        col("spaceId").alias("id")
    ).alias("space"),
    struct(
        col("personalId").alias("id"),
        struct(
            struct(col("result")[0].alias("concentration"), col("result")[1].alias("activity")).alias("reasonal")
        ).alias("state")
    ).alias("person")
)


# JSON 변환
result_data_json = result_data.select(to_json(struct(col("*"))).alias("value"))

# # Kafka로 내보내기
query = result_data_json \
    .writeStream \
    .outputMode("append") \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka_ip:9092") \
    .option("topic", "state-reasonal") \
    .option("client_id", 'eeg-da-server-producer-2') \
    .option("checkpointLocation", "C:/Users/cshiz/spark/spark-3.4.0-bin-hadoop3/checkpoint_2") \
    .start()

# query = eeg_data_windowed \
#     .writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()

query.awaitTermination()
