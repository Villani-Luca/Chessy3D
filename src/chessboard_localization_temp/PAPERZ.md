# Piece recog using sift 
https://web.stanford.edu/class/cs231a/prev_projects_2016/CS_231A_Final_Report.pdf

# 17/05/2025
si e provato a fare un approccio tramite heatmap e rete neurale che identifichi i lattice points
delle scacchiere come definito nel paper https://github.com/maciejczyzewski/neural-chessboard

questo pero si e rivelato poco soddisfacente e preciso, si suppone che la nn relativa a lattice points 
non funzionasse correttamente con immagini su cui non e stato fatto il train.

# 25/05/2025

oggi ci siamo concentrati nella ricerca degli angoli della scacchiera provando vari metodi e capendo veramente cosa sta succedendo nei vari pezzi della pipeline.
siamo arrivati alla seguenti pipeline che mostra risultati concreti

blur, otsu, canny, dilation, hough, extract contours till numbers of polygons with 4 sides is the most, approx polygons, dilations, biggest contours and extract corners.

questo giro é stato creato poiché si é notato che l'estrazione direteta da hough a contours é molto soggetta a noise e all'input, di fatti hough fornisce dei risultati che all'occhio umano hanno senso ma che la macchina tramite le funzxioni di findContours non interpreta correttamente poiché non si ha abbastanza overlap tra le righe riconosciute.

ripetendo dilations tante volte questo ci permette di raffinare in maniera precisa il risultato di hough in modo da poter riconoscere con maggiore chiarezza i poligoni con 4 lati che nelle nostre immagini dovrebbero essere presenti in grande quantitá.

effettuando poi dei filtri possiamo estrarre solo o la maggior parte dei quadrati trovati dai contours che ci permette poi di approssimare la scacchiera nella sua interezza.

al momento l'algo si aspetta un contorno esterno grande che incapsuli tutto, un possibile miglioramento da testare é se non presente ( ovvero se il contour identificato ha una area non soddisfacente rispetto all'area dell'immagine ) un contour esterno di crearne uno valutando iterativamente i quadrati al loro interno 
costruendo piano piano un quadrato che incapsuli il resto per poi usarlo come confine empirico della tastiera.

l'idea della dilation é venuta osservando dei diagrammi dei vari poligoni all'interno dell'immagine e il come differivano in immagine diverse, inoltre per il nostro caso d'uso é stato esplorato un fine tuning dei parametri di hough che peró a parte per la thresholds non si sono rivelati impattanti ( ripetere il test nel caso ) anche con notevoli variazioni.
un approccio di studio é stato creare combinazioni dei parametri di hough per portare avanti nella pipeline molteplici immagini da studiare in parallelo per poi capire quale fosse la migliore per quell'immagine. questo approccio é stato messo in sospeso in favore di filtri rispetto all'output di hough per avere un risultato maggiormente deterministico e di piú facile ottimizzazione.

# 26/06/2025
continuando con l'analisi della pipeline e la quest per la riduzione del rimore nei passaggi pre hough, sono passato all'input originale ovvero a canny.

analizzando varie combinazioni di thresholds si é in primis notato che come teoricamente prevedibile, effettuando un operazione di thresholding
precedente a canny questa rende la parametrizzazione di canny praticamente ininfluente.

portandoci quindi a testare i parametri direttamente sull'immagine grigia alla ricerca di un migliore input, ( immagini canny_params_gray ), avendo dei risultati 
soddisfacenti con thresholds tra 300-400 e 500-700 apparantemente ottimali per il nostro scopo

un idea classica per ridurre il rumore é di effetutare un filtro gaussiano sull'immagine griglia per smoothing, applicandolo alla nostra analisi come previsto si é visto una diminuzione notevole della noise spostando di conseguenza i parametri ottimali di canny nel range di 100-300 e 300-600

in questi range empirici si sospetta che si abbia una buona risoluzione per hough in modo da evitare noise e di conseguenza avere migliori risultati nei passaggi di identificazione della board.

( immagini nella cartella sono per possibili esempi nel paaper, salvo anche una copia del notebook usato per crearle in modo da poterlo modificare nel caso )
