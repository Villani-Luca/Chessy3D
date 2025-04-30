dataset di partite con mosse e metadati https://lumbrasgigabase.com/en/


nel db per ogni posizione salvata:
    - avere un feature vector o qualcosa su cui si possa fare una similarità
    - id / qualche  modo di indentificare a quali partite sono collegati 

tenere tutti questi dati all'interno di una singola collezzione potrebbe generare dei problemi.
di conseguenza si creano due collezzioni di documenti:
    - una prima collezzione che contiene la posizione vettorizzata e un id 
    - una seconda collezzione che contiene relativo al id della posizione vetorizzata la posizione delle partite in cui si trova quelal posizione
    - una terza collezzione in cui si hanno tutte le partite con le relative posizioni e metadati


DB vettoriale https://docs.trychroma.com/docs/overview/introduction con sdk per python "pip install chromadb"

un modo per rendere più veloce l'import sarebbe spezzare il file in molteplici

## 28/04/2025
effetetuato il primo testing per buttare roba dentro chroma e 

# 29/04/2025
un problema che si ha é cosa usare come id della posizione dato che dovrá essere efficiente 
per poter indicizzare possibilmente delle tabelle con decine di milioni di record.

il primo tentativo é stato usare una descrizione della posizione chiamata FEN ma troppo lunga e inefficiente
per poi scoprire Zobrist hashing 

inoltre le performance di python del parsing del database erano troppo inferiori al necessario passando quindi ad un parsing custom c#