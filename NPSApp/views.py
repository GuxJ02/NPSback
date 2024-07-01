from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import numpy as np
from googletrans import Translator, LANGUAGES
from unidecode import unidecode
import json
import re
import os
from django.conf import settings
import traceback
import sys
import requests
from multiprocessing import Pool, cpu_count
import time
from concurrent.futures import ThreadPoolExecutor
import threading

stop_processing_lock = threading.Lock()
stop_processing = False

# Create your views here.
def holamundo(request):
    print("Printing something before return")
    return render(request, 'index.html')

# Remover emojis y otros caracteres especiales
def remove_emojis(text):
    if text is None:
        return ""
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def translate_word(word):
    translator = Translator()
    try:
        translation = translator.translate(word, dest='en')
        return translation.text.lower()
    except Exception as e:
        print(f"Error al traducir palabra '{word}': {e}")
        return word.lower()

def translate_text_with_googletrans(text):
    translator = Translator()
    try:
        detected_lang = translator.detect(text).lang
        if detected_lang not in LANGUAGES:
            print(f"Idioma no detectado correctamente o no soportado: {detected_lang}")
            return text  # Retorna el texto original si el idioma no es soportado
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print(f"Error durante la traducción: {e}")
        return text  # Retorna el texto original en caso de error

def process_reviews(json_source, filtro):
    global stop_processing
    with stop_processing_lock:
        stop_processing = False 

    frases_traducidas = []
    estrellas = []

    # Dividir el filtro en palabras basadas en '-', normalizarlas y luego traducirlas
    palabras_originales = [unidecode(word).lower() for word in filtro.split('-')] if filtro else []
    palabras_traducidas = [translate_word(word) for word in palabras_originales]
    # Unificar listas de palabras originales y traducidas
    filtros = list(set(palabras_originales + palabras_traducidas))

    print(f"Iniciando análisis para la universidad con filtros: {filtros}")

    if json_source.startswith('http'):
        response = requests.get(json_source)
        if response.status_code == 200:
            data = response.json()
        else:
            response.raise_for_status()
    else:
        with open(json_source, 'r', encoding='utf-8') as file:
            data = json.load(file)

    def process_review(review):
        global stop_processing
        with stop_processing_lock:
            if stop_processing:
                print("El procesamiento ha sido interrumpido.")
                return None, None

        review_text = review.get("review_text")
        review_rating = review.get("review_rating", None)

        if review_text is None:
            return None, None  # Saltar esta reseña si el texto es None

        # Normalizar el texto de la reseña
        review_text_normalizado = unidecode(review_text).lower() if review_text else ""

        # Comprobar si alguna palabra del filtro está en la reseña utilizando expresiones regulares
        if any(re.search(rf'\b{filtro}\b', review_text_normalizado) for filtro in filtros):
            print(f"Procesando reseña: {review_text}")
            # Limpiar texto de emojis y traducir
            review_text_clean = remove_emojis(review_text).replace('\n', ' ').strip() if review_text else ""
            frase_limpia = unidecode(review_text_clean)
            start_time = time.time()
            frase_traducida = translate_text_with_googletrans(frase_limpia)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"El tiempo de ejecución es: {elapsed_time} segundos")
           
            return frase_traducida, review_rating
        else:
            return None, None  # Si ninguna palabra del filtro está presente, continuar con la siguiente reseña

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_review, review) for item in data for review in item.get("reviews_data", [])]
        for future in futures:
            with stop_processing_lock:
                if stop_processing:
                    print("El procesamiento ha sido interrumpido.")
                     # Apagar el ThreadPoolExecutor
                    return None, None
            frase_traducida, review_rating = future.result()
            if frase_traducida is not None:
                frases_traducidas.append(frase_traducida)
                estrellas.append(review_rating)

    media_estrellas = sum(estrellas) / len(estrellas) if estrellas else 0
    return frases_traducidas, media_estrellas
def stop_processing_view(request):
    global stop_processing
    with stop_processing_lock:
        stop_processing = True
    return JsonResponse({'status': 'Procesamiento detenido'})

def analyze_batch(batch_textos, model, tokenizer, umbral):
    global stop_processing
    with stop_processing_lock:
        if stop_processing:
            return ([], [], [], [])
    
    encoded_input = tokenizer(batch_textos, padding=True, truncation=True, return_tensors='pt', max_length=512)

    with torch.no_grad():
        output = model(**encoded_input)
    scores = output.logits.detach().numpy()
    scores = softmax(scores, axis=1)
    
    puntuaciones_positivas_generales = []
    puntuaciones_negativas = []
    puntuaciones_neutrales = []
    puntuaciones_positivas = []

    for score in scores:
        puntuaciones_positivas_generales.append(score[2])
        puntuaciones_negativas.append(score[0])
        puntuaciones_neutrales.append(score[1])
        if score[2] < umbral or score[0] < umbral:
            if not (score[1] > score[2] and score[1] > score[0] and score[1] > 0.50):
                puntuaciones_positivas.append(score[2])
    
    return puntuaciones_positivas_generales, puntuaciones_negativas, puntuaciones_neutrales, puntuaciones_positivas

def load_roberta_model():
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()
    with stop_processing_lock:
        if stop_processing:
            print("El procesamiento ha sido interrumpido.")
            return None, None
    return tokenizer, model

tokenizer, model = load_roberta_model()

def analisis_Roberta(textos, umbral, tokenizer, model, batch_size=10):
    global stop_processing
    procesados = 0
    all_puntuaciones_positivas_generales = []
    all_puntuaciones_negativas = []
    all_puntuaciones_neutrales = []
    all_puntuaciones_positivas = []

    num_batches = (len(textos) + batch_size - 1) // batch_size
    batches = [textos[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

    with Pool(processes=2) as pool:
        try:
            results = pool.starmap(analyze_batch, [(batch, model, tokenizer, umbral) for batch in batches])
            
            for result in results:
                with stop_processing_lock:
                    if stop_processing:
                        print("El procesamiento ha sido interrumpido.")
                        pool.terminate()  # Terminar el pool de procesos
                        pool.join()       # Esperar a que los procesos terminen
                        return [0, 0, 0, 0, 0, 0]

                puntuaciones_positivas_generales, puntuaciones_negativas, puntuaciones_neutrales, puntuaciones_positivas = result
                all_puntuaciones_positivas_generales.extend(puntuaciones_positivas_generales)
                all_puntuaciones_negativas.extend(puntuaciones_negativas)
                all_puntuaciones_neutrales.extend(puntuaciones_neutrales)
                all_puntuaciones_positivas.extend(puntuaciones_positivas)
                procesados += len(puntuaciones_positivas_generales)
                print(f"Número de frases analizadas: {procesados}")

        except Exception as e:
            print(f"Error durante el análisis: {e}")
            pool.terminate()
            pool.join()
            raise e

    promotores = [p for p in all_puntuaciones_positivas if p >= 0.8]
    pasivos = [p for p in all_puntuaciones_positivas if 0.6 <= p < 0.8]
    detractores = [p for p in all_puntuaciones_positivas if p < 0.6]

    # Cálculo del porcentaje de promotores y detractores
    total_respuestas = len(all_puntuaciones_positivas)
    porcentaje_promotores = len(promotores) / total_respuestas * 100 if total_respuestas else 0
    porcentaje_detractores = len(detractores) / total_respuestas * 100 if total_respuestas else 0
    media_positivos = porcentaje_promotores - porcentaje_detractores
    if media_positivos != 0:
        nps = (media_positivos + 100) / 2
    else:
        nps = 0
    
    print(f"NPS: {nps}")

    media_positivos_generales = sum(all_puntuaciones_positivas_generales) / len(all_puntuaciones_positivas_generales) if all_puntuaciones_positivas_generales else 0
    media_negativos = sum(all_puntuaciones_negativas) / len(all_puntuaciones_negativas) if all_puntuaciones_negativas else 0
    media_neutrales = sum(all_puntuaciones_neutrales) / len(all_puntuaciones_neutrales) if all_puntuaciones_neutrales else 0
    desviacion_estandar = np.std(all_puntuaciones_positivas) if all_puntuaciones_positivas else 0
    desviacion_estandar = desviacion_estandar * 100
    media_positivos_generales = media_positivos_generales * 100
    media_negativos = media_negativos * 100
    media_neutrales = media_neutrales * 100
    
    print(f"Media de puntuaciones positivas: {media_positivos_generales}")
    print(f"Media de puntuaciones negativas: {media_negativos}")
    print(f"Media de puntuaciones neutrales: {media_neutrales}")
    
    return [nps, desviacion_estandar, procesados, media_positivos_generales, media_negativos, media_neutrales]

def get_reviews_tags(universidad_id):
    # Agregar la extensión .json al nombre del archivo
    if universidad_id == "UniversidadValencia":
        json_file_url = 'https://drive.google.com/uc?export=download&id=1EEQlzsR0ZdnYUKfR_8gxzYOWNj024tKC'
    if universidad_id == "UniversidadNebrija":
        json_file_url = 'https://drive.google.com/uc?export=download&id=1okjIAjLcF_9tmNvSfy3ET7EgQOzuKx9F'
    if universidad_id == "UniversidadFranciscoVitoria":
        json_file_url = "https://drive.google.com/uc?export=download&id=1rJ3hJBnoioYVOx3-BCKHBCZ21PDNLi_Y"
    if universidad_id == "UniversidadEuropea":
        json_file_url = "https://drive.google.com/uc?export=download&id=1Oxg4Vnq1ORW9PN3NvX4GVPekzcVUecJG"
    if universidad_id == "UniversidadDeZaragoza":
        json_file_url = "https://drive.google.com/uc?export=download&id=1oRcZiY-b4yaOYHgEBB6OpkFy5VlnBRxv"
    if universidad_id == "UniversidadComplutense":
        json_file_url = "https://drive.google.com/uc?export=download&id=14E2Tp4Z3f8Yc79WCKsr3oqWJTvbnEgqE"
    if universidad_id == "UniversidadCarlos":
        json_file_url = "https://drive.google.com/uc?export=download&id=1PwWucpPP4Ho-dsy6eH2YpEMzQAMUqRQX"
    if universidad_id == "UniversidadAutonomaMadrid":
        json_file_url = "https://drive.google.com/uc?export=download&id=1redoAZkFyZ7QuvS3Ga7QKUlfhJc1PYO2"
    if universidad_id == "UniversidadAutonomaBarcelona":
        json_file_url = "https://drive.google.com/uc?export=download&id=1QLR-jryV0DY2iFIwrk5uRYHTlwpDIubY"

    print(f"Obteniendo palabras de interés para el archivo: {universidad_id}")
    nombre_archivo = f'{universidad_id}.json'
    if json_file_url.startswith('http'):
        response = requests.get(json_file_url)
        if response.status_code == 200:
            data = response.json()
        else:
            response.raise_for_status()
    else:
        with open(json_file_url, 'r', encoding='utf-8') as file:
            data = json.load(file)
    
    # Inicializar una lista para almacenar las palabras
    todas_las_palabras = []

    # Extraer las palabras del campo review_tags
    for entry in data:
        if 'reviews_tags' in entry:
            todas_las_palabras.extend(entry['reviews_tags'])
    
    return todas_las_palabras

def obtener_palabras_de_interes(request, universidad_id):
    try:
        print(f"Obteniendo palabras de interés para la universidad: {universidad_id}")
        reviews_tags = get_reviews_tags(universidad_id)
        if reviews_tags is None:
            return
        return JsonResponse({'reviews_tags': reviews_tags})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def obtener_datos_json(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Asume que la respuesta es un JSON
    else:
        response.raise_for_status()

def analizar_universidad(request, universidad_id):
    global stop_processing
    with stop_processing_lock:
        stop_processing = False

    try:
        filtro = request.GET.get('filtro', '')  # Obtiene el filtro de la URL
        print(f"Iniciando análisis para la universidad: {universidad_id} con filtro: {filtro}")

        json_file_urls = {
            "UniversidadValencia": 'https://drive.google.com/uc?export=download&id=1EEQlzsR0ZdnYUKfR_8gxzYOWNj024tKC',
            "UniversidadNebrija": 'https://drive.google.com/uc?export=download&id=1okjIAjLcF_9tmNvSfy3ET7EgQOzuKx9F',
            "UniversidadFranciscoVitoria": 'https://drive.google.com/uc?export=download&id=1rJ3hJBnoioYVOx3-BCKHBCZ21PDNLi_Y',
            "UniversidadEuropea": 'https://drive.google.com/uc?export=download&id=1Oxg4Vnq1ORW9PN3NvX4GVPekzcVUecJG',
            "UniversidadDeZaragoza": 'https://drive.google.com/uc?export=download&id=1oRcZiY-b4yaOYHgEBB6OpkFy5VlnBRxv',
            "UniversidadComplutense": 'https://drive.google.com/uc?export=download&id=14E2Tp4Z3f8Yc79WCKsr3oqWJTvbnEgqE',
            "UniversidadCarlos": 'https://drive.google.com/uc?export=download&id=1PwWucpPP4Ho-dsy6eH2YpEMzQAMUqRQX',
            "UniversidadAutonomaMadrid": 'https://drive.google.com/uc?export=download&id=1redoAZkFyZ7QuvS3Ga7QKUlfhJc1PYO2',
            "UniversidadAutonomaBarcelona": 'https://drive.google.com/uc?export=download&id=1QLR-jryV0DY2iFIwrk5uRYHTlwpDIubY',
        }

        json_file_url = json_file_urls.get(universidad_id)
        if not json_file_url:
            return JsonResponse({'status': 'error', 'message': 'Universidad no encontrada.'})

        textos, mediaEstrella = process_reviews(json_file_url, filtro)
        print("Reseñas procesadas, iniciando análisis de sentimientos...")
        with stop_processing_lock:
            if stop_processing:
                print("El procesamiento ha sido interrumpido.")
                return JsonResponse({'status': 'Procesamiento detenido'})

        if not textos:
            return JsonResponse({'status': 'error', 'message': 'No se encontraron reseñas que coincidan con el filtro.'})

        puntuacion = analisis_Roberta(textos, 0.5, tokenizer, model)   # Asume que devuelve [media_positivos, desviacion_estandar, numero_frases]

        data = {
            'nps': puntuacion[0],
            'desviacion_estandar': puntuacion[1],  # Agrega la desviación estándar
            'numero_frases': puntuacion[2],
            'media_estrellas': mediaEstrella,
            'media_positivos_generales': puntuacion[3],
            'media_negativos': puntuacion[4],
            'media_neutrales': puntuacion[5]
        }
        print(data)  # Imprimir para depuración

        return JsonResponse(data)  # Devuelve los datos como JSON

    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        traceback.print_exc()  # Esto imprimirá la traza del error en la consola
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)