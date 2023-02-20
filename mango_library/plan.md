https://www.jetbrains.com/help/idea/markdown.html

**Logik beim Handel**
- verkaufte single_negotiations einberechnen
- Agent0 kann 3/2 mehr Energie umwandeln, aber hat gerade das Recht sie zu verkaufen, also bietet er es an.
  - Agent3 kann 1/1 mehr Energie umwandeln, aber hat gerade das Recht sie zu verkaufen, also bietet er es an.
  - Agent1 kaufte 2/0 & Agent2 kaufte 0/1.
  - Die Käufer von Energie Recht müssen mindestens das ins Netz einspeisen.
    - [0,0][2,0][0,1][0,0]
  - Die Verkäufer müssen mindestens diese Energie umwandeln.
    - [2,0][0,0][0,0][0,1]
  - open_schedule berechnen für Agent 1, aus letztem Stand, aber die Käufe werden garantiert.

- [2,3][0,3][4,4][2,2] letzter Stand
- [0,0][2,0][0,1][0,0] gekaufte Rechte
- [2,3][2,3][4,4][2,2]
- [2,3]     [4,4][2,2] = [8,9], aber Ziel ist [15,15]
- Also: sind noch [7,6] offen, aber [2,0] ist mindestens nötig.

**single_negotiations komprimieren**
- SingleNegotiations.get_combined_single_negotiations()
  - Sortieren, nach Preis günstigstes zuerst
  - Dann immer einzeln zu allen vorhandenen hinzufügen
    -  if Ergebnis ist neu:
      - Ergebnis zu
  - Die neuen Ergebnisse, die es schon gab, dann löschen, damit die nächste Runde das nicht doppelt testen muss.
  - In der Addition sind die Preise sofort summiert
  - (evtl. Minus nur so viel wie produziert werden kann und Plus so viel wie benötigt wird)

**ACHTUNG:**
- Bedenken, wenn der Preis positiv/negativ ist und wie viel dann angeboten wird.
- Negotiations können nicht weiterverkauft werden. 
  Verkaufte Energie muss selbst umgewandelt werden. 
  Gekaufte Energie muss erzeugt werden.


- convert_amount:     3
- kwh_price:          1.1
- schedules_provider: [4,3][5,7][8,4]
- converted_price:    0.8

[1, 0 .8]
[0, 1 .8]
[0,-1 .8]
[1, 0 .9]
[1, 0 .9]
[0, 1 .9]

- [x] fertig
- [x] SingleNegotiation wird als inaktiv gesehen von anderen, wenn schon jemand interessiert ist, denn dann kann man es eh nicht kaufen
  - wenn schon Interessenten angemeldet sind, dann wird es nach hinten sortiert, denn eventuell kann der Verkäufer das zukünftig wählen
- [ ] MinMax
- [ ] Algorithmus für die Erstellung der SingleNegotioatiton und fürs Annehmen.
- [ ] Metaheuristik Evolution FireFly PSO Variante
- [ ] Kombinatorisches Problem ('memetic') https://www.youtube.com/watch?v=JVeMvYqrvw0 'Ameisenalgorithmus'
- [ ] Mixed Intger
- [ ] aproximations algorithmus
- [ ] Rucksack Problem Knappsackproblem
- [ ] JariPerformance kommt vielleicht nicht auf ein Ende. Eventuell muss man das am Ende alles kaufen. Preis für Änderung erhöhen, um das Ende zu erzwingen.
- [ ] convert_amounts kann direkt berechnet werden
- [ ] Metheuristik Fitness Funktion ist das wichtigste und da kann alles rein.

2022-09-07
- ein Agent muss mit dem schedule verhandeln, den er hat, wenn er was verkauft muss er das mindestens verhandeln und wenn er was kauft, dann kann er nicht mehr so viel verhandeln, denn die Erlaubnis wird schon vom Verkäufer verhandelt.
- ein Agent bietet, seinen kompletten offenen 'covert_amount' an zu dem Preis den er dadurch weniger bekommt.
- im Endergebnis steht wie viel Energie von wem genutzt wird, wie viel wer verhandelt hat, wie viel wer verkauft/umwandelt
  - used_schedule, schedule, sold, converted

2022-09-08
- Termination Detection MangoLibrary
  - Es soll enden, sobald sich nichts ändert und alle SingleNegotiations gesehen wurden
    - damit alle gesehen werden können, dürfen deaktivierte nicht immer neue erstellt werden

2022-09-08
- Durch SingleNego käufe wird die Strafe nicht größer, weil dann jemand anders weniger verkauft.
- termination: Wenn alle Agenten keine schedule änderung haben und keine Käufe getätigt haben in der letzten Runde.

2022-10-11
- welche Negation wählen für die Berechnung?
  - zuerst das eigene Ziel berechnen (Ziel - Summe anderer)
  - dann eigene mögliche Verläufe (mit Abzug von den schon verkauften) auflisten
  - dann die günstigsten Negationen wählen, die den Verläufen am nächsten sind
    - sortierung nach Preis der Negation
    - Dann nehmen bis ein Limit erreicht ist
  - dann "Calculate the schedules"

2022-10-12
- IDEE: Vgl. Strompreis wird Merit-Order genutzt, um zu viel andere Umwandlung zu vermeiden
  - das bedeutet, dass die ersten Käufe alleine den geringeren Preis ausgleichen, wobei es später immer weiter auf alle verteilt wird, weil alle profitieren, dass einzelne nicht direkt den maximalen Betrag erhalten
- IDEE: Jeder Kauf von Umwandlung wird von allen ausgeglichen, aber nicht, wenn ein Agent von sich selbst kauft.

2022-10-13
- _watch_active_negotiations: alte negotiations beenden
- open_convert_amount - VerkaufteEnergie = anzubietende Energie

2022-10-14
```
ERROR:asyncio:Exception in callback Agent.raise_exceptions(<Task cancell...agent.py:210>>)
handle: <Handle Agent.raise_exceptions(<Task cancell...agent.py:210>>)>
Traceback (most recent call last):
  File "/usr/lib/python3.8/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.8/asyncio/base_events.py", line 616, in run_until_complete
    return future.result()
  File "/home/jari/PycharmProjects/mango-library-master/mango_library/jari.py", line 76, in test_case
    await asyncio.wait_for(wait_for_coalition_built(agents), timeout=5)
  File "/usr/lib/python3.8/asyncio/tasks.py", line 501, in wait_for
    raise exceptions.TimeoutError()
asyncio.exceptions.TimeoutError
During handling of the above exception, another exception occurred:
```

2022-10-16
- Idee für das Angebot
  - 1/2/4/8/... Dann sortieren die Käufer die großen zuerst.
Dann ist es schneller

2022-10-24
- Kauf und Verkauf wurde doppelt abgezogen. Das hat zu dem Fehler geführt.

2022-10-27
- mehrfach negs ([7, 7, 7, 7, 7, 7, 7, 7, 7, 7]), aber davon soll nur das Günstigste in 'Calculate the schedules'
- diese Liste kann mehr verschiedene Werte beinhalten
- IDEE DAFÜR: Die Agenten geben an welche Mengen sie zu welchem Zeitpunkt anbieten
  - ZUERST filtert der Käufer die SNs, damit nur noch aktive SNs ohne Interessenten vorhanden sind.
  - DANN werden die SNs gruppiert nach Zeitpunkten, an denen sie liegen
    - Ein SN kann in mehreren Zeitpunkt-Arrays sein
  - DANN werden die Gruppen sortiert, sodass der geringste Preis zuerst genommen wird, danach wird nach der Größe des Angebots sortiert (groß zuerst)
  - DANN kann der Käufer die verschiedenen Angebote schnell fusionieren, um sich seinem Zielwert zu nähern
    - jeweils wird geprüft, ob das SN über dem Zielwert liegt
  - DAMIT kann schnell gezogen werden welche SingleNegs am günstigsten sind für verschiedene Mengen, diese können dann in 'Calculate the schedules'

2022-10-30
- es endet bevor alle wissen, ob ihre SNs bearbeitet wurden
  - muss eine Nachricht geschickt werden, wenn es bearbeitet wurde, damit jeder nochmal nach sieht?
- Alle anderen Möglichkeiten werden zuerst genutzt, egal was sie kosten
  - Wenn dann Energie gebraucht wird, schalten die Energieumwandler Stück für Stück runter, aber wenn eh zu viel Energie vorhanden ist, dann bleibt es beim Maximum, bzw. wenn gesamt gar nicht genug vorhanden ist, dann wird nichts für weniger umgewandelt.
- Einzige Frage: Wie viel bekommen Umwandler, wenn es sowieso zu viele Umwandler gibt.

2022-10-31
- Jeder Agent bekommt den minderwert der anderen Energie ausgeglichen
- INPUT:
  1. Alle OUTPUTs
  2. Mögliche Schedules eigener elektrischer Energie
  3. Maximalen Schedule eigener andere Energieumwandlung
- CALC:
  - elektrische Energie: ToDo: ZIEL - SUM(Andere Agenten)
  - wertigsten Schedule wählen mit möglichen Umwandlungen
  - ALLE Schedules mit MAX Energieumwandlung probieren und den Wert vergleichen
  ```
  Agent0:
  ZIEL: [50 50] Strafe: 2 (evetuell erhöhen mit der Zeit?)
  Aktuelle Schedules: 
  [16 13]*1(0 0)*0.8 - SELBST
  [10 10]*1(5 5)*0.7
  [10 10]*1(5 5)*0.8
  [10 10]*1(5 5)*0.8
  [10 10]*1(5 5)*0.8
  
  [10 10] ist offen kWh: 1.0 umgewandelt: 0.8
  [12 9] oder [16 13] kann erzeugt werden
  [3 4] kann umgewandelt werden
  offen   erzeug    schedu  umwandl strafe
  [10 10] [12  9] | [10  9] [ 2  0] [0 1]
  [10 10] [16 13] | [10 10] [ 3  3] [3 0]
  
  schedu  umwandl strafe
  [10  9] [ 2  0] [0 1] = 19*1+2*0.8-1*2 = 18.6
  [10 10] [ 3  3] [3 0] = 20*1+7*0.8-3*2 = 19.6
  
  anderen Umwandlung abnehmen, wenn sie teurer ist
  Agent1 hat (5 5), aber kostet mehr, also ändert sich das OUTPUT:
  statt:
  [10 10]*1(3 3)*0.8 - SELBST
  [10 10]*1(5 5)*0.7
  jetzt:
  [10  9]*1(3 4)*0.8 - SELBST
  [10 11]*1(5 4)*0.7
  
  OUTPUT: [10  9]*1(3 4)*0.8
  ```
- OUTPUT:
  1. Schedule & kWh-Preis eigener elektrische Energie
  2. Schedule & kWh-Preis eigener andere Energie

2022-11-05
- TODO: wait_for_coalition_built

2022-11-08
- Wichtigkeit einzelner Zeitpunkte wie in COHDA beachten
- Wert der gesamten Gruppe & gleichzeitig Median & Mean betrachten
- Ziel erweitern? Gruppe macht Wärme und Strom
  - Ziel Wärme Fahrplan + Strom Fahrplan
  - System Evaluieren - Recherche zu möglichen Preisen, usw.
  - Recherche --> Introduction

2022-11-22
- Meine COHDA flexibel in andere Formen UND andere TargetSchedules berechnen
- DANN gibt es mehrere OperatingSchedules
- mit Umformungen kann dann verbunden werden
  - Beispiel GAS in Wärme UND Strom!

2022-11-23
- neues 'wait_for_term', denn '._weight_map.values()' ist nicht korrekt
```
Agent0:
ZIEL: Wärme kWh:[100 100]Strom kWh:[100 100] flexibel StromToX. Strafe: Verkaufspreis
Aktuelle Schedules: 
[30 30][16 13]*1(0 0)*0.8[15 15] - SELBST
[10 10]*1(5 5)*0.7
[10 10]*1(5 5)*0.8
```
- Wärme in kWh + Verkaufspreis
- Strom in kWh + Verkaufspreis
- Umwandlung in max kW + kWh + Verkaufspreis
- GasTo in max kW >kWhGas + Einkaufspreis + kWhWärmeProkWh> + kWhStromProkWh>
- StromToWärme in max kW + >kWhStrom + WärmeFaktor
- WärmeToStrom in max kW + >kWhWärme + StromFaktor

|     >Strom      |   Wärme    |   Strom    | Umwandlung |    GasToHeat+Power    | PowerToHeat | Value |
|:---------------:|:----------:|:----------:|:----------:|:---------------------:|:-----------:|:-----:|
|                 |  [20 20]   |  [20 20]   |            |                       |             |       |
| [12 09] [16 13] | [00 00]*.8 | [00 00]*.4 | 5 [0 0]*.2 | 30 [00 00]*0.07 .5 .5 | 7 [0 0]*0.9 |   0   |
| [10 10] [15 15] | [00 00]*.9 | [00 00]*.3 | 5 [0 0]*.2 | 30 [00 00]*0.07 .5 .4 | 7 [0 0]*0.8 |   0   |

- Nur 1 Agent bekannt:
- Strom füllen:

| >Strom  |    Heat    |   Strom    | Umwandlung |    GasToHeat+Power    | PowerToHeat |    Value    |    braucht     | 
|:-------:|:----------:|:----------:|:----------:|:---------------------:|:-----------:|:-----------:|:--------------:|
|         |  [20 20]   |  [20 20]   |            |                       |             |             |                |
| [16 13] | [00 00]*.8 | [16 13]*.4 | 5 [0 0]*.2 | 30 [00 00]*0.07 .5 .5 | 7 [0 0]*0.9 | 29*.4= 11.6 | [20 20][04 07] |
| [15 15] | [00 00]*.9 | [15 15]*.3 | 5 [0 0]*.2 | 30 [00 00]*0.07 .5 .4 | 7 [0 0]*0.8 |  30*.3= 9   | [20 20][05 05] |

- Wärme füllen, mit Gas2Heat+Power:

| >Strom  |    Heat    |   Strom    | Umwandlung |     GasToHeat+Power     | PowerToHeat |           Value           |    braucht     | 
|:-------:|:----------:|:----------:|:----------:|:-----------------------:|:-----------:|:-------------------------:|:--------------:|
|         |  [20 20]   |  [20 20]   |            |                         |             |                           |                |
| [16 13] | [15 15]*.8 | [16 13]*.4 | 5 [0 0]*.2 | 30 [30 30]*0.07 [.5 .5] | 7 [0 0]*0.9 | 30*.8+29*.4-60*0.07= 31.4 | [05 05][04 07] |
| [15 15] | [15 12]*.9 | [15 15]*.3 | 5 [0 0]*.2 | 30 [30 30]*0.07 [.5 .4] | 7 [0 0]*0.8 | 27*.9+30*.3-60*0.07= 29.1 | [05 08][05 05] |

- PowerToHeat, wenn günstiger: Dazu berechnen welche Menge Gas optimal pro Zeitpunkt gekauft werden muss. Mit der Möglichkeit PowerToHeat.
- TO-DO: Dafür Gleichung erstellen

| >Strom  |     Heat     |        Strom        | Umwandlung |     GasToHeat+Power     |      PowerToHeat      |              Value              |    braucht     | 
|:-------:|:------------:|:-------------------:|:----------:|:-----------------------:|:---------------------:|:-------------------------------:|:--------------:|
|         |   [20 20]    |       [20 20]       |            |                         |                       |                                 |                |
| [16 13] |  [20 20]*.8  | [10.4444 7.4444]*.4 | 5 [0 0]*.2 | 30 [30 30]*0.07 [.5 .5] | 7 [5.5555 5.5555]*0.9 | 40*.8+17.8888*.4-60*0.07= 34.95 | [05 05][04 07] |
| [15 15] | [20 17.6]*.9 |     [8.75 8]*.3     | 5 [0 0]*.2 | 30 [30 30]*0.07 [.5 .4] |    7 [6.25 7]*0.8     | 37.6*.9+16.75*.3-60*0.07= 34.67 | [05 08][05 05] |

```
[10 10] ist offen kWh: 1.0 umgewandelt: 0.8
[12 9] oder [16 13] kann erzeugt werden
[3 4] kann umgewandelt werden
offen   erzeug    schedu  umwandl strafe
[10 10] [12  9] | [10  9] [ 2  0] [0 1]
[10 10] [16 13] | [10 10] [ 3  3] [3 0]
schedu  umwandl strafe
[10  9] [ 2  0] [0 1] = 19*1+2*0.8-1*2 = 18.6
[10 10] [ 3  3] [3 0] = 20*1+7*0.8-3*2 = 19.6
anderen Umwandlung abnehmen, wenn sie teurer ist
Agent1 hat (5 5), aber kostet mehr, also ändert sich das OUTPUT:
statt:
[10 10]*1(3 3)*0.8 - SELBST
[10 10]*1(5 5)*0.7
jetzt:
[10  9]*1(3 4)*0.8 - SELBST
[10 11]*1(5 4)*0.7
OUTPUT: [10  9]*1(3 4)*0.8
```

2022-11-28
- INPUT:
  - Zielmenge Heat+Power
  - OUTPUTs anderer Agenten
- BERECHNUNG:
  - Solar Menge ausprobieren
  - Gas Menge ausprobieren, mit Value Höhepunkt finden
  - MIN(max Power für Power2Heat + max benötigte Heat für Power2Heat) = Power2Heat Menge
  - Wenn Power > Zielmenge, dann Umwandlung
  - Berechne Value
- BEKANNTES:
  - z.B. Heat+Power+Umwandlung Verkaufspreis, ...
- OUTPUT:
  - Mengen Solarpower, Gas, PowerToHeat, Umwandlung

2022-11-29

- Wenn man NUR Solar, Strafe ist höher, wenn der Abstand größer ist. z.B. Quadratisch
- Wiederverwertbarkeit für Erweiterung auf beliebig viele Energieformen/Zielfahrpläne und Umwandlungen

2023-02-05

- schedule kann sich nicht ändern wenn alle nur das angeben was perfekt ist
- wird das heat_target überhaupt betrachtet?
- gas_purchases
- Jetzt alle schedules verbinden?

2023-02-06

- Heat vernünftig einbinden
- neben dem alten schedule eine Liste von schedule_with_max_value geben. Der beste schedule wird an die Koalition weiter
  gegeben, damit die anderen darauf reagieren können
- nach self.test_gas_amount() nicht mehr einzelne schedules betrachten, sondern den schedule_with_max_value
- dann können andere agenten auf die verschiedenen schedule_with_max_value reagieren

2023-02-14
- wie perf vergleichen?
  - jeder speichert sein perf wert in sein

2023-02-20
- deviation_to_target_schedule wieder nutzen?
