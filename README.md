#### main.py
Glaven program. Z zagonom main.py se izvede celotna računska metoda.

#### compare.py
Vsebuje funkcije, za pripravo podatkov na klic metode compare_by_component() iz datoteke data_processing.py.

#### helpers.py in data_processing.py
Glavni datoteki, vsebujeta funkcije za izgranjo, primerjavo modelov in iskanje najbolj ustreznega modela.

#### cron-job
Mapa cron-job vsebuje datoteke in kodo, ki se izvede kot cron opravilo. Kliče se vsako uro in preko API-ja pridobi prometne podatke. Podatki se shranijo v MongoDB bazo. Skripto zaženemo z ukazom: python cron_script.py

#### data
V mapi data so shranjeni podatki v .csv obliki. Podatkovni zbirki južne ljubljanske obvoznice in Šmartinske ceste nista javno objavljeni v repozitoriju. Poleg podatkov, se nahaja tudi datoteka data.py in weather.py. Data.py uredi prometne podatke do ustrezne oblike. Weather.py pa vsebuje klic API-ja in pridobitev zgodovinskih vremenskih podatkov. 

#### results
V mapo results se shranjujejo grafi, ki se generirajo med izvajanjem kode.


--------------------------------------

Koda je očiščena, odstranjeni so vsi API ključi in podatki. API ključe je potrebno dodati na naslednjih mestih:
* weather.py, vrstica 12
* data.py, vrstica 50 (potrebno urediti tudi ime baze in zbirke)
* cron_script.py, vrstica 11 in vrstica 13 (potrebno urediti tudi ime baze in zbirke)

main.py se izvede tudi brez zgornjih API ključev.
