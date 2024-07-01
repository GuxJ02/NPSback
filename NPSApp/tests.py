from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch
import json

class ViewsTestCase(TestCase):
    def setUp(self):
        self.client = Client()

    def test_holamundo_view(self):
        response = self.client.get(reverse('holamundo'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'index.html')

    @patch('NPSApp.views.requests.get')
    @patch('NPSApp.views.translate_text_with_googletrans')
    def test_analizar_universidad(self, mock_translate, mock_requests):
        # Setup mock
        mock_translate.side_effect = lambda text: f"translated {text}"
        mock_requests.return_value.status_code = 200
        mock_requests.return_value.json.return_value = [
            {"reviews_data": [{"review_text": "Buenísima experiencia", "review_rating": 5}]}
        ]
        
        url = reverse('analizar_universidad', args=['UniversidadValencia'])
        response = self.client.get(url, {'filtro': 'experiencia'}, HTTP_X_REQUESTED_WITH='XMLHttpRequest')
        self.assertEqual(response.status_code, 200)
        self.assertIn('nps', response.json())

    @patch('NPSApp.views.requests.get')
    def test_obtener_palabras_de_interes(self, mock_requests):
        mock_requests.return_value.status_code = 200
        mock_requests.return_value.json.return_value = [
            {"reviews_tags": ["excelente", "bueno", "malo"]}
        ]
        
        url = reverse('obtener_palabras_de_interes', args=['UniversidadValencia'])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('reviews_tags', response.json())

    def test_stop_processing_view(self):
        url = reverse('detener_procesamiento')
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'status': 'Procesamiento detenido'})
