#### prometni_podatki.ipynb
Demonstracija izvedene analize.

#### compare.py
Vsebuje funkcije, za pripravo podatkov na klic metode compare_by_component() iz datoteke data_processing.py.

#### data_processing.py
Glavna datoteka, vsebuje funkcije za izgranjo, primerjavo modelov in iskanje najbolj ustreznega modela.

#### helpers.py
Vsebuje pomožne funkcije za prikaz in izdelavo modelov.

#### cron-job
Mapa cron-job vsebuje datoteke in kodo, ki se izvede kot cron opravilo. Kliče se vsako uro in preko API-ja pridobi prometne podatke. Podatki se shranijo v MongoDB bazo. Skripto zaženemo z ukazom: python cron_script.py

#### results
V mapo results se shranjujejo grafi in rezultati, ki se generirajo med izvajanjem kode.


--------------------------------------

Koda je očiščena tako, da so odstranjeni vsi API ključi in podatki. API ključe je potrebno dodati na naslednjih mestih:
* weather.py, vrstica 12
* data.py, vrstica 50 (potrebno urediti tudi ime baze in zbirke)
* cron_script.py, vrstica 11 in vrstica 13 (potrebno urediti tudi ime baze in zbirke)

Analizo na podatkih lahko izvedemo tudi brez zgoraj omenjenih API ključev. API ključi so namenjeni le pridobitvi podatkov.
