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
- nach self.calculate_gas_amount() nicht mehr einzelne schedules betrachten, sondern den schedule_with_max_value
- dann können andere agenten auf die verschiedenen schedule_with_max_value reagieren

2023-02-14
- wie perf vergleichen?
  - jeder speichert sein perf wert in sein

2023-02-20
- deviation_to_target_schedule wieder nutzen?

# max_iterations 100
# .06926153846153000000000000000000000 178745+168445 347191 = 68306
# .06926153846153500000000000000000000 173715+163657 337373 = 67626
# .06926153846153600000000000000000000 174415+164323 338738 = 67721
# .06926153846153700000000000000000000 169352+159504 328856 = 67036
# .06926153846153800000000000000000000 167045+157308 324353 = 66724
# .06926153846153820000000000000000000 160424+151004 311429 = 65829
# .06926153846153830000000000000000000 157415+148140 305556 = 65422
# .06926153846153840000000000000000000 142437+133882 276320 = 63397
# .06926153846153842000000000000000000 139189+130790 269980 = 62958
# .06926153846153843000000000000000000 130689+122698 253387 = 61809
# .06926153846153845000000000000000000 126748+118947 245696 = 61276
# .06926153846153846000000000000000000 115945+108663 224608 = 59816
# .06926153846153846300000000000000000 114024+106834 220858 = 59556
# .06926153846153846400000000000000000 111859+104773 216632 = 59263
# .06926153846153846450000000000000000 115676+108406 224082 = 59779
# .06926153846153846460000000000000000 111475+104408 215883 = 59211
# .06926153846153846465000000000000000 111201+104147 215348 = 59174
# .06926153846153846466000000000000000 113671+106498 220169 = 59508
# .06926153846153846466100000000000000 112474+105359 217833 = 59346
# .06926153846153846466150000000000000 116403+109098 225502 = 59878
# .06926153846153846466160000000000000 114161+106964 221126 = 59575
# .06926153846153846466160200000000000 112491+105374 217865 = 59349
# .06926153846153846466160300000000000 110760+103726 214487 = 59115
# .06926153846153846466160400000000000 112518+105401 217920 = 59352
# .06926153846153846466160420000000000 113171+106022 219194 = 59441
# .06926153846153846466160430000000000 114856+107626 222482 = 59668
# .06926153846153846466160430500000000 116085+108795 224881 = 59835
# .06926153846153846466160430800000000 112999+105858 218858 = 59417
# .06926153846153846466160430850000000 113492+106327 219820 = 59484
# .06926153846153846466160430850100000 111485+104416 215901 = 59213
# .06926153846153846466160430850300000 111145+104094 215239 = 59167
# .06926153846153846466160430850320000 111821+104737 216559 = 59258
# .06926153846153846466160430850322500 116394+109090 225484 = 59876
# .06926153846153846466160430850322600 115271+108021 223293 = 59725
# .06926153846153846466160430850322675 111882+104795 216677 = 59266
# .06926153846153846466160430850322700 116301+109001 225302 = 59864
# .06926153846153846466160430850322713 115996+108711 = 224708 59823
# .06926153846153846466160430850322716 115518+108256 = 223774 59758
# .06926153846153846466160430850322718 111643+104567 = 216210 59234
# .06926153846153846466160430850322719 112625+105502 = 218128 59367
# .069261538461538464661604308503227195 115211+107964 = 223175 59716
# .069261538461538464661604308503227195 116524+109213 = 225738 59894
# .0692615384615384646616043085032271975 115550+108287 = 223838 59762
# .0692615384615384646616043085032271975 115211+107964 = 223175 59716
# .0692615384615384646616043085032271985 117767+110397 = 228164 60062
# .0692615384615384646616043085032271985 115232+107984 = 223216 59719
# .0692615384615384646616043085032271985 114177+106980 = 221157 59577
# .069261538461538464661604308503227200 108302+101386 = 209689 58782
# .069261538461538464661604308503227250 109147+102192 = 211339 58897
# .069261538461538464661604308503227500 109119+102165 211285 = 58893
# .069261538461538464661604308503227893 108934+101988 = 210922 58868
# .06926153846153846466160430850322850 114981+107745 = 222726 59685
# .06926153846153846466160430850322875 112772+105642 = 218414 59387
# .069261538461538464661604308503228800 114981+107745 = 222726 59685
# .0692615384615384646616043085032288125 114282+107079 = 221361 59591
# .0692615384615384646616043085032288200 112261+105155 = 217416 59318
# .06926153846153846466160430850322882125 114105+106911 = 221017 59567
# .0692615384615384646616043085032288216 115325+108072 = 223398 59732
# .06926153846153846466160430850322882175 113580+106411 = 219992 59496
# .069261538461538464661604308503228821825 103448+96766 = 200215 58126
# .0692615384615384646616043085032288219 93012+86832 = 179845 56715
# .0692615384615384646616043085032288225 97731+91324 = 189055 57353
# .0692615384615384646616043085032288250 100235+93708 = 193944 57692
# .069261538461538464661604308503228850 90845+84768 = 175614 56422
# .069261538461538464661604308503228875 101592+94999 = 196592 57875
# .06926153846153846466160430850322900 95271+88981 = 184253 57021
# .06926153846153846466160430850323000 99139+92664 191803 = 57544
# .06926153846153846466160430850323750 98946+92480 191427 = 57518
# .06926153846153846466160430850325000 96725+90366 187091 = 57217
# .06926153846153846466160430850330000 99627+93128 192755 = 57610
# .06926153846153846466160430850340000 102331+95703 198034 = 57975
# .06926153846153846466160430850350000 105256+98487 203743 = 58371
# .06926153846153846466160430850357000 97187+90806 187994 = 57280
# .06926153846153846466160430850400000 102080+95464 197544 = 57941
# .06926153846153846466160430851000000 100089+93568 193658 = 57672
# .06926153846153846466160430852000000 98398+91958 190356 = 57443
# .06926153846153846466160430855000000 98735+92279 191014 = 57489
# .06926153846153846466160430860000000 100584+94039 194624 = 57739
# .06926153846153846466160430870000000 97032+90658 187690 = 57259
# .06926153846153846466160430880000000 96332+89992 186325 = 57164
# .06926153846153846466160430900000000 98890+92427 191318 = 57510
# .06926153846153846466160431000000000 101469+94882 196352 = 57859
# .06926153846153846466160432000000000 99964+93449 193413 = 57655
# .06926153846153846466160435000000000 96330+89990 186320 = 57164
# .06926153846153846466160438000000000 96586+90233 186819 = 57198
# .06926153846153846466160440000000000 98038+91616 189654 = 57395
# .06926153846153846466160450000000000 101174+94601 195776 = 57819
# .06926153846153846466160500000000000 97062+90687 187749 = 57263
# .06926153846153846466160600000000000 97657+91253 188910 = 57343
# .06926153846153846466161000000000000 94169+87933 182103 = 56872
# .06926153846153846466162000000000000 97912+91496 189409 = 57378
# .06926153846153846466164000000000000 100142+93619 193762 = 57679
# .06926153846153846466165000000000000 97538+91140 188679 = 57327
# .06926153846153846466170000000000000 103244+96572 199816 = 58099
# .06926153846153846466180000000000000 101643+95048 196692 = 57882
# .06926153846153846466190000000000000 99936+93422 193359 = 57651
# .06926153846153846466200000000000000 92239+86095 178335 = 56611
# .06926153846153846466300000000000000 94409+88161 182570 = 56904
# .06926153846153846466500000000000000 100094+93573 193667 = 57673
# .06926153846153846467000000000000000 99638+93139 192778 = 57611
# .06926153846153846470000000000000000 101427+94842 196270 = 57853
# .06926153846153846480000000000000000 99248+92768 192016 = 57558
# .06926153846153846500000000000000000 93939+87714 181654 = 56841
# .06926153846153846600000000000000000 94118+87885 182003 = 56865
# .06926153846153846670000000000000000 97619+91217 188837 = 57338
# .06926153846153846700000000000000000 98895+92432 191327 = 57511
# .06926153846153847000000000000000000 98876+92414 191291 = 57508
# .06926153846153847500000000000000000 94434+88185 182620 = 56908
# .06926153846153848000000000000000000 84625+78848 163474 = 55581
# .06926153846153849000000000000000000 85062+79264 164326 = 55641
# .06926153846153850000000000000000000 72108+66932 139040 = 53889
# .06926153846153851000000000000000000 70286+65198 135484 = 53643
# .06926153846153855000000000000000000 59506+54936 114443 = 52185
# .06926153846153856000000000000000000 59994+55400 115395 = 52251
# .06926153846153857000000000000000000 58080+53578 111658 = 51993
# .06926153846153858000000000000000000 59181+54626 113808 = 52142
# .06926153846153859000000000000000000 48989+44924 93913 = 50764
# .06926153846153860000000000000000000 55099+50740 105840 = 51590
# .06926153846153861000000000000000000 57030+52578 109608 = 51851
# .06926153846153862000000000000000000 52037+47825 99863 = 51176
# .06926153846153863000000000000000000 53422+49143 102566 = 51363
# .06926153846153870000000000000000000 45581+41679 87260 = 50303
# .06926153846153880000000000000000000 42256+38514 80771 = 49853
# .06926153846153890000000000000000000 44589+40735 85324 = 50169
# .06926153846153900000000000000000000 36100+32654 68754 = 49021
# .06926153846154000000000000000000000 26117+23150 49267 = 47671
# .06926153846154200000000000000000000 16861+14339 31200 = 46420
# .06926153846154500000000000000000000 7417+5349 12766 = 45143
# .06926153846155000000000000000000000 4101+2193 6295 = 44695
# .06926153846156000000000000000000000 2405+578 2984 = 44466
# .06926153846157000000000000000000000 2403+576 2980 44465


# max_iterations = 4
# 0/0       169461+159880 = 329341 67047
# .01/.01   168527+158719 = 327246 66925
# .02/.02   166374+156940 = 323314 66629
# .03/.03   166457+156748 = 323206 66645
# .04/.04   167624+157858 = 325482 66802
# .05/.05   166606+157162 = 323768 66661
# .06/.06   166123+156429 = 322553 66600
# .065/.065 167935+158427 = 326362 66840
# .068/.068 166267+156567 = 322834 66619
# .069/.069 167549+157787 = 325337 66792
# .0692     166178+156755 = 322934 66603
# .06925    166903+157445 = 324348 66701
# .06926    168135+158345 = 326480 66872
# .069261   166885+157155 = 324040 66703
# .0692615  168597+159057 = 327654 66930
# .06926152 166534+156821 = 323356 66655
# .06926153 168139+158349 = 326489 66872
# .069261535 168225+158431 = 326657 66884
# .069261538 168553+158743 = 327296 66928
# .0692615382 165965+156279 = 322245 66578
# .0692615383 166562+157120 = 323682 66655
# .0692615384 167022+157285 = 324308 66721
# .06926153845 167243+157495 = 324739 66751
# .06926153846 165965+156279 = 322245 66578
# .069261538461 164664+155041 = 319705 66402
# .0692615384615 167675+158179 = 325854 66805
# .06926153846152 167728+158230 = 325958 66812
# .06926153846153 165742+156067 = 321809 66548
# .069261538461535 160652+151221 = 311873 65860
# .069261538461537 155399+146221 = 301620 65150
# .069261538461538 153531+144714 = 298246 64893
# .0692615384615384 145778+137062 = 282841 63849
# .06926153846153845 134924+126730 = 261654 62382
# .06926153846153846 119890+112418 = 232309 60349
# .06926153846153847 100419+93882 = 194302 57717
# .06926153846153850 82098+76442 = 158540 55240
# .06926153846153855 57267+52804 = 110071 51883
# .069261538461539 41666+37953 = 79619 49774
# .06926153846154 26867+23865 = 50732 47773
# .06926153846155 3939+2038 = 5977 44673
# .06926153846155 2501+669 = 3170 44479
# .0692615384616 2403+576 = 2980 44465
# .0692615384618 2403+576 = 2980 44465
# .069261538462 2403+576 = 2980 44465
# .069261538465 2403+576 = 2980 44465
# .06926153847 2403+576 = 2980 44465
# .06926153849 2403+576 = 2980 44465
# .0692615385 2403+576 = 2980 44465
# .069261539 2403+576 = 2980 44465
# .06926154 2403+576 = 2980 44465
# .06926156 2403+576 = 2980 44465
# .0692616  2403+576 = 2980 44465
# .0692617  2403+576 = 2980 44465
# .069262   2403+576 = 2980 44465
# .069265   2403+576 = 2980 44465
# .06927    2403+576 = 2980 44465
# .06928    2403+576 = 2980 44465
# .0693     2403+576 = 2980 44465
# .07/.07   2403+576 = 2980 44465
# .08/.08   2319+546 = 2866 44456
# .09/.09   2241+532 = 2774 44448
# .10/.10   2108+524 = 2633 44434
# .11/.11   2037+519 = 2557 44427
# .12/.12   1871+515 = 2387 44411
# 1/1       931+823 = 1755  44261
# 2/2       819+819 = 1639  44249
# 100/105   815+815 = 1630  44248
# 100/105   893+905 = 1798  44263
# 100/100   62+1702 = 1765  44139
# 100/100   53+1580 = 1633  44159
# 10/10     60+1751 = 1811  44149
# 10/10     50+2061 = 2112  44173
# 10/10.5   925+747 = 1672  44277
# 10/10.5   1182+746 = 1928 44301
# 10/11     1800+50 = 1850  44157
# 10/11     1788+47 = 1836  44155
# 951/999   831+795 = 1626  44236
# 951/999   946+905 = 1851  44268