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

purtroppo peró la seconda parte di geenrazione dei embedding e salvataggio sono lentiessime in chroma 5 secondi per 5000 record
da trovare soluzioni

# 30/04/2025

testando con vari database vettoriali, quelli forniti localmente 
o comunque in sistemi piú semplici ( ovvero docker ecc.. ) si rivelano molto lenti 
e per l'ammontare di dati che si vuole utilizzare ( TODO: da riveredere ) circa 75 milioni posizioni uniche
non si prestano benissimo

un metodo esplorato é una codifica dei pezzi ideata da stockfish e adattata da lichess
https://lichess.org/@/revoof/blog/adapting-nnue-pytorchs-binary-position-format-for-lichess/cpeeAMeY

messa in pausa per indagare 

Retrieval of Similar Chess Positions
Debasis Ganguly Johannes Leveling Gareth J. F. Jones
School of Computing, Centre for Next Generation Localisation
Dublin City University, Dublin 9, Ireland
{dganguly, jleveling, gjones}@computing.dcu.ie

https://doras.dcu.ie/20378/1/ganguly-sigir2014.pdf

oggi ci si é concentrati sulla definizione di una metrica di similaritá, di fatti fare un encoding naive della posizione in 8x8x6x2 non é particolarmente
efficace poiché perde tutte le relazioni logiche della posizione che potrebbe aiutare a rendere simile la posizione ( es: attacco su una diagonale da parte di una regina o di un alfiere )

il paper offre i seguenti punti chiavi per una similaritá della posizione
    - posizione 
    - activity
    - attacking
    - defending
    - ray attacks

to then construct an inverse index with the previous created tokens

# 05/05/2025
praticamente tutto il tempo é stato usato per caricare e creare i vettori dentro milvus, problemi con chroma per lentezza.
