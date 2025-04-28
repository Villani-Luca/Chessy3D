
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