| Método | Exactitud (test) | F1 macro | Exactitud con giros y desplazamientos | Entrenamiento | Tiempo por imagen | Parámetros |
|---|---:|---:|---:|---:|---:|---:|
| Similitud media por dígito (SSIM) | 68,9 % | 0,679 | 30,0 % | — | 27,45 ms | — |
| Plantillas medias + SSIM | 70,2 % | 0,701 | 29,9 % | < 1 s | 0,49 ms | — |
| Vecino más parecido (SSIM) | 87,7 % | 0,875 | 41,2 % | — | 27,45 ms | — |
| HOG + SVM lineal | 97,5 % | 0,975 | 67,7 % | 30 s | 0,80 ms | 12.970 |
| Perceptrón multicapa (PyTorch) | 98,0 % | 0,980 | 55,8 % | 11 s | 0,04 ms | 203.530 |
| CNN LeNet (PyTorch) | 99,2 % | 0,992 | 82,5 % | 198 s | 0,79 ms | 1.528.078 |
| CNN + aumento de datos | 99,4 % | 0,994 | 98,9 % | 337 s | 0,77 ms | 1.528.362 |
