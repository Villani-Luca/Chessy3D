# Chessy3D

Abbiamo identificato i seguenti moduli:
- Identificazione della scacchiera nella immagine
- Identificare l'orientazione della scacchiera nell'immagine ( soggetta a possibile limitazioni )
- Trovare le celle della scacchiera
- Distinguere i vari componenti della cella
- Rappresentazione digitale 2d dello stato estratto
- Modulo di retrieval dato uno stato della scacchiera con partite simili


Moduli nella src:
- Modulo 1: 
  - deve estrarre i corner della scacchiera 
  - darla in input che sarà un array di array di 4 punti
  - deve estrarre anche l'angolazione dell'immagine e i dati aggiuntivi della telecamera
- Modulo 2:
  - input: array e immagine
  - metodo: LARRIS e a manazza
  - output: array di 64 elementi con 4 punti per elemento + colore; darà in output solo le scacchiere valide
- Modulo 3:
  - object detection con transformers (tipo e colore)
- Modulo 4:
  - matching modulo 3  + modulo 2
- Modulo 5:
  - direzione + orientamento (opzionale) di default arriva da bianco + scacchiera 2D finale
- Modulo 6: 
  - retrieval per similarità (calcolo similarita, ricerca db per partite simili)
- Modilo 7: 
  - UI