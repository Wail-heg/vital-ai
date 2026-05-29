# Ethische toelichting — Vroege sepsisvoorspelling Isala × Windesheim

**Opdrachtgever:** Isala ziekenhuis Zwolle
**Onderzoeksgroep:** IT-innovations in Healthcare — Windesheim
**Opleiding:** HBO-ICT — Data Science (tweedejaars)
**Datum:** mei 2026
**Bijlage:** Moreel stappenplan (laatste hoofdstuk)

---

## 1. Inleiding en scope

Dit document is geen formele bijlage achteraf. Het hoort bij het hart van
deze opdracht: een machine-learning model dat met enige waarschijnlijkheid
gaat fout adviseren over leven en dood, en daarover moeten we voor de
oplevering kunnen verantwoorden wat we hebben afgewogen, op welke gronden,
en wat we *niet* doen omdat we het niet kunnen verantwoorden.

De ethische analyse is hier expliciet gekoppeld aan **één concreet dilemma
dat tijdens het project is opgedoken** — de keuze van de alarmdrempel,
omdat onze evaluatie laat zien dat verschillende redelijke drempelkeuzes
elkaar ethisch uitsluiten. Dat dilemma is in hoofdstuk 8 uitgewerkt en
vormt de basis van het moreel stappenplan in de bijlage.

### Reikwijdte
- We werken met openbare, geanonimiseerde PhysioNet-data uit twee
  Amerikaanse academische ziekenhuizen. Géén Isala-data.
- Het opgeleverde model is een **studieobject**, geen klinisch hulpmiddel.
  Het wordt niet ingezet zonder ten minste een DPIA, een CE-traject
  (klasse IIa) en een prospectieve validatie.
- Deze toelichting beperkt zich tot wat dit project zelf raakt; bredere
  AI-ethische discussies (bv. arbeidsmarkt-effecten) zijn buiten scope.

### Methodische verantwoording
We gebruiken het kader van Beauchamp & Childress (vier biomedische
principes) als hoofdraamwerk en het stappenplan van Van de Poel & Royakkers
voor de uitwerking van het concrete dilemma. Aanvullend volgen we de
EU-richtsnoeren voor betrouwbare AI (HLEG, 2019) waar die specifieker zijn
dan de algemene biomedische principes (m.n. uitlegbaarheid en menselijke
controle).

---

## 2. Belanghebbendenanalyse

Een goede ethische analyse begint bij wie er belang bij heeft — en wat dat
belang precies inhoudt. Onderstaande tabel ordent de partijen op nabijheid
tot de patiënt.

| Belanghebbende | Hun belang | Risico bij ons model |
|---|---|---|
| **De patiënt** | Tijdige herkenning van sepsis, geen onnodige behandeling, privacy van medische data | Vals-negatief = gemiste of late diagnose, mogelijk overlijden. Vals-positief = onnodige antibiotica, lijnen, IC-opname-overschrijding |
| **Intensivist (eindverantwoordelijke)** | Beslissingsondersteuning die het werk verbetert, niet juridisch risico vergroot | Black-box-model dat fouten maakt waarvoor de arts uiteindelijk aansprakelijk is |
| **IC-verpleegkundige** | Werkbare alarmsignalen, geen alarmmoeheid | Te veel valse alarmen leiden tot negeren van échte alarmen |
| **Isala (organisatie)** | Betere zorguitkomsten, kostenbesparing, reputatiebescherming | Aansprakelijkheid, MDR-naleving, mogelijk averechtse uitkomsten |
| **Familie van de patiënt** | Inzicht in welke informatie de beslissing stuurt | Onvoldoende uitleg bij ongunstige afloop |
| **Maatschappij** | Doelmatige zorg, beperkte antibioticaresistentie | Overbehandeling draagt bij aan AMR; ongelijke prestatie tussen groepen ondergraaft vertrouwen |
| **Toezichthouders (IGJ, AP)** | Compliance met AVG, MDR, WGBO | Onduidelijke verantwoordingsketen rond AI-beslissingen |
| **Wij als onderzoekers** | Methodologisch verantwoord werk, transparantie over beperkingen | Verleiding tot ‘te mooie’ resultatenpresentatie |

> **Conflict zien.** De patiënt en het ziekenhuis zijn meestal *een* partij
> in deze opsomming, maar bij de drempelkeuze in hoofdstuk 8 zien we hun
> belangen wél divergeren — de patiënt heeft baat bij hoge sensitiviteit,
> het ziekenhuis bij minder alarmen.

---

## 3. Morele waarden in conflict

Vier waarden lopen door dit hele project. Ze versterken elkaar meestal,
maar wringen op specifieke beslismomenten.

1. **Weldoen (beneficence).** Het model moet patiënten daadwerkelijk
   beschermen tegen vermijdbare verslechtering.
2. **Niet-schaden (non-maleficence).** Het model mag geen schade toevoegen
   die er zonder model niet zou zijn — bv. onnodige antibiotica, alarmmoeheid
   die *andere* patiënten benadeelt, of stigmatisering van subgroepen.
3. **Autonomie.** De patiënt heeft recht op informatie over hoe een
   AI-systeem hun zorg beïnvloedt. De intensivist heeft recht op zinvolle
   menselijke controle, niet alleen op een toestemmingsknop.
4. **Rechtvaardigheid.** Het model moet voor verschillende subgroepen
   (geslacht, leeftijd, etniciteit) vergelijkbaar presteren.

We voegen één **instrumentele waarde** toe die deze vier verbindt:
**transparantie & uitlegbaarheid**. Zonder uitleg kunnen autonomie en
controle niet gerealiseerd worden, en zonder uitleg blijft de fairness-
discussie kwalitatief in plaats van toetsbaar.

---

## 4. Toepassing van Beauchamp & Childress op dit project

### 4.1 Weldoen
Het model haalt op de PhysioNet-validatieset AUROC 0,83 en een mediane
lead time van 17,5 uur bij gedetecteerde patiënten. Op zichzelf lijkt dat
gunstig — vroege herkenning kan levens redden. *Maar:* de PhysioNet utility
score, die expliciet rekening houdt met *wanneer* een voorspelling valt,
is bij de fp-budget-drempel **−0,21**: het model doet daar gemiddeld
*schade* in plaats van goed. Onze inzet moet daarom voorwaardelijk zijn,
niet enthousiast.

### 4.2 Niet-schaden
De schade die ons model kan veroorzaken is *verborgen* en *verspreid*:

- **Alarmmoeheid.** Een verpleegkundige die structureel vals-positieve
  signalen ziet, leert het systeem te negeren — schadelijk voor *latere*
  patiënten waar het signaal terecht is.
- **Overbehandeling.** Vals-positieven leiden tot onnodig brede
  antibiotica, lijnen, IC-opname-verlenging — schade die per individuele
  patiënt klein lijkt maar maatschappelijk optelt (resistentie, kosten).
- **Vals vertrouwen.** Een gepubliceerde AUROC van 0,83 kan een
  ziekenhuisbestuurder overtuigen om een instabiel model in productie te
  brengen.

### 4.3 Autonomie
Patiënten zijn op de IC vaak niet in staat geïnformeerde toestemming te
geven. Dat betekent niet dat autonomie geen rol speelt — wel dat de
verantwoordelijkheid voor informeren bij het ziekenhuis ligt:

- Patiëntinformatie over inzet van AI in de zorgverlening (analoog aan
  beeldvormingsadvies).
- Recht op een uitleg achteraf — niet alleen ‘het systeem zei’.
- Op systeemniveau: betekenisvolle menselijke controle, *niet* een
  ‘rubber stamp’ op AI-advies.

### 4.4 Rechtvaardigheid
Onze meetbare fairness-gap (AUROC) is **0,003 voor geslacht** en
**0,035 voor leeftijd** — ruim binnen onze norm van 10 %. Dit is een
positief signaal, maar met drie nuances:

- We meten alleen op zichtbare attributen. Etniciteit ontbreekt in de
  PhysioNet-data; dat is geen geruststelling, maar een blinde vlek.
- Vrouwen presenteren atypisch (Mehta et al., 2018). Een kleine gap in
  AUROC kan een grotere gap in *recall* verbergen — we rapporteren beide.
- De Isala-populatie is anders dan PhysioNet. Een gap die hier klein is,
  kan na transfer groter worden.

---

## 5. Privacy en AVG-grondslag

### 5.1 Huidig project
- We gebruiken de **openbare PhysioNet 2019-dataset**. Deze is door de
  oorspronkelijke onderzoekers gepubliceerd met instemming van de
  betrokken ziekenhuizen en is geanonimiseerd (datum-shift, geen vrije tekst,
  geen direct identificeerbare velden).
- De data wordt **niet** in de cloud verwerkt zonder verwerkersovereenkomst.
- Geen data in git, alleen in een lokale `data/`-map die in `.gitignore`
  staat.

### 5.2 Eventuele inzet met Isala-data
Als Isala in een vervolgstap eigen IC-data wil gebruiken, geldt:

- **AVG-grondslag.** Artikel 9 lid 2 sub h (zorgverlening door
  zorgprofessionals) of i (volksgezondheid) — afhankelijk van
  productie-vs-onderzoeksdoel. Niet automatisch toestemming-gebaseerd,
  omdat IC-patiënten niet allemaal kunnen instemmen.
- **DPIA verplicht.** Hoge risicoclassificatie (gezondheidsgegevens +
  geautomatiseerde besluitvorming).
- **Pseudonimisering.** Vervang `Patient_ID` door geüpgrade hash; bewaar de
  sleutel onder vier-ogen.
- **Re-identificatierisico.** IC-opnamepatroon (tijd + Unit + age) is in
  combinatie potentieel re-identificerend; we hebben Age + Gender + ICULOS
  nodig — een differentially-private alternatief is bewerkelijk en past
  niet bij de tweedejaars-scope.
- **Bewaartermijn.** Dezelfde als het EPD (medische dossierplicht
  doorgaans ≥ 20 jaar voor de patiëntdata, maar modelartefacten
  korter — afspraken vastleggen).

---

## 6. Bias en fairness — verdiept

### 6.1 Bronbias
PhysioNet 2019 bevat data uit Beth Israel (Boston) en Emory (Atlanta).
Twee Amerikaanse academische centra met:

- andere case mix (meer trauma in Boston),
- andere laboratorium-protocollen,
- andere drempels voor IC-opname,
- andere demografische samenstelling dan Zwolle.

Een model dat op deze data getraind is, bevat impliciet de keuzes en
biases van die ziekenhuizen.

### 6.2 Subgroep-bias in dit project
We hebben empirisch getoetst:

| Subgroep | n_pos | AUROC | Recall @ threshold |
|---|---|---|---|
| Vrouw (Gender=0) | 2 026 | 0,829 | 0,589 |
| Man (Gender=1) | 2 880 | 0,832 | 0,580 |
| Leeftijd ≤ 40 | 546 | 0,847 | 0,564 |
| Leeftijd 40–55 | 953 | 0,823 | 0,590 |
| Leeftijd 55–70 | 1 800 | 0,834 | 0,573 |
| Leeftijd 70–80 | 1 035 | 0,835 | 0,622 |
| Leeftijd ≥ 80 | 572 | 0,813 | 0,554 |

De gap is klein. Dat is goed, maar het ontheft ons niet van de
verantwoordelijkheid om bij elke versie opnieuw te meten, want bias kan
*terugkeren* na hertraining.

### 6.3 Informatieve missingness als bias-bron
Onze beste-presterende features bevatten **`_isna`-flags** — een lab is
‘missend’ als de arts geen aanleiding zag voor afname. Dat betekent dat
het model in feite leert: *patiënten waarover de arts zich druk maakt,
worden vaker geprikt; dus de aanwezigheid van metingen is een proxy voor
arts-verdenking*. Dit is mogelijk een **niet-causaal pad**: het model
voorspelt niet sepsis maar de zorgactiviteit-rond-sepsis. Bij implementatie
in een ander ziekenhuis met andere prikgewoonten faalt het model dan
mogelijk subtiel — een vorm van *target leakage door zorgproces*.

Dit is een onderzoeksvraag voor vervolg, niet iets dat we hier oplossen.
Maar we vermelden het expliciet, omdat het bias kan veroorzaken die in
geen enkele subgroep-tabel zichtbaar is.

---

## 7. Transparantie en uitlegbaarheid

Voor het primaire subdoel hebben we drie lagen ingebouwd:

1. **Algemene transparantie (modelkaart).** Een korte beschrijving van
   trainingsdata, beoogde inzet, *out-of-scope*-gebruik en bekende
   beperkingen — analoog aan Mitchell et al. (2019).
2. **Globale uitlegbaarheid (SHAP summary).** Welke features sturen het
   model gemiddeld? Verwachting: rolling-min van bloeddruk,
   shock-index-rolling, lactaat, leukocyten, qSOFA-componenten.
3. **Lokale uitlegbaarheid (SHAP per voorspelling).** Voor elke patiënt
   tonen we de top-3 features die de voorspelling drijven, met
   contributie-richting (positief/negatief).

**Beperking.** SHAP-waarden zijn correlationeel, niet causaal. Wij weten
*welke features* het model gebruikt, niet *waarom* die voorspellend zijn.
Voor een arts is dat onderscheid relevant — een hoge SHAP-waarde voor
‘missend lactaat’ kan een arts verwarren of geruststellen, afhankelijk van
hoe het wordt gecommuniceerd. We bevelen daarom een **gestructureerde
uitlegtemplate** aan in het PDMS, niet een ruwe SHAP-plot.

---

## 8. Concreet dilemma — de drempelkeuze

Dit is *het* dilemma dat dit project oplevert, en de aanleiding voor het
moreel stappenplan in de bijlage.

### 8.1 Het feitelijke probleem
Op onze validatieset met 7 270 patiënten levert het LightGBM-model:

| Drempelstrategie | Drempel | Recall | FP/24u | PhysioNet utility |
|---|---|---|---|---|
| **Klinisch fp-budget** (≤ 3 FP/24u) | 0,59 | 0,58 | 3,0 | **−0,21** |
| **Utility-optimum** | 0,85 | 0,20 | 0,4 | **+0,04** |

We staan voor een keuze waarbij **élke optie ethisch ongunstige
eigenschappen heeft**:

- Optie A (fp-budget): we vangen 58 % van de sepsis-patiënten op tijd, maar
  de PhysioNet utility (die schade door valse alarmen meeweegt) is
  *negatief* — gemiddeld doet het systeem dan schade.
- Optie B (utility-optimum): we doen netto enige goed (utility +0,04), maar
  vangen slechts 20 % van de sepsis-patiënten — 80 % gemist.
- Optie C (qSOFA-basislijn): geen ML, maar slechts 45 % recall en 2,5 %
  precision — ook niet aanvaardbaar.

### 8.2 Waarom dit echt een ethisch dilemma is
Dit is geen technische optimalisatie: er bestaat **geen drempel** waarop
zowel een acceptabele recall (zegmaar ≥ 0,6) als een acceptabele utility
(zegmaar +0,1) gehaald wordt. Dat we deze trade-off niet kunnen
wegoptimaliseren is een feature van *deze* dataset met *dit* model.

De ethische lading komt niet uit een rekenfout maar uit het volgende:
**de mensen die schade ondervinden van optie A zijn andere mensen dan
degenen die schade ondervinden van optie B**.

- Optie A schaadt vooral *niet-septische* patiënten (overbehandeling,
  alarmmoeheid die *andere* patiënten benadeelt).
- Optie B schaadt vooral *septische* patiënten die te laat worden gevangen.

Beide zijn schade. Welke geven we voorrang aan? Dat is een **morele**
vraag, niet een technische.

### 8.3 Onze conclusie (vooruitlopend op het stappenplan)
We bevelen aan om met *geen* van beide drempels in productie te gaan op
basis van de PhysioNet-data alleen. Inzet vereist:

1. Lokale fine-tuning op Isala-IC-data zodat de utility hopelijk
   onomstreden positief wordt.
2. Tot die tijd presenteren we het model intern als
   **beslissingsondersteunings-prototype**, niet als alarm.
3. De keuze van een definitieve drempel is een **bestuursbeslissing**
   (intensivisten + ethische commissie + patiëntenraad), niet een
   data-scientist-beslissing.

---

## 9. Conclusie en handelingsaanbevelingen

1. **Voorwaardelijke inzetbaarheid.** Niet in productie zonder lokale
   validatie en utility > 0,2 op Isala-data.
2. **Twee drempels documenteren.** De ‘klinische’ en de ‘utility-optimale’
   drempel beide aan de bestuurslagen voorleggen — niet één kiezen om de
   discussie te vermijden.
3. **Transparante modelkaart** opnemen in elke versie-release.
4. **Periodieke fairness-monitoring** op subgroepen die in productie
   *daadwerkelijk* aanwezig zijn (niet alleen op PhysioNet-subgroepen).
5. **Stille proefperiode** van ≥ 3 maanden voor élk significant
   model-update.
6. **Klinische beslissing blijft bij de mens.** Het model adviseert, het
   alarmeert niet rechtstreeks.
7. **Patiëntinformatie** over AI-gebruik in de zorgverlening, in voor leken
   begrijpelijke taal.

---

# Bijlage — Moreel stappenplan (Van de Poel & Royakkers)

We werken het stappenplan volledig uit voor het concrete dilemma uit
hoofdstuk 8: **welke alarmdrempel voor het LightGBM sepsismodel?**

### Stap 1 — Beschrijf het probleem

Op onze validatieset zien we dat verschillende redelijke drempelkeuzes
voor het waarschuwingsmodel leiden tot ofwel:

- (A) een hoge sensitiviteit (58 % van de sepsis-patiënten op tijd
  gevangen) maar netto schade volgens de PhysioNet utility (−0,21),
- (B) netto licht voordeel (utility +0,04) maar lage sensitiviteit (20 %).

De vraag: welke drempel adviseren wij Isala te hanteren — of: welk advies
ten aanzien van het *al dan niet* hanteren van een drempel.

### Stap 2 — Analyseer het probleem

**Feitelijke context.** Sepsis is acuut levensbedreigend; vroege
behandeling redt levens. Maar antibiotische overbehandeling op IC niveau
draagt bij aan resistentie en heeft individuele bijwerkingen. De
PhysioNet utility score is een algemene maat van klinische waarde, maar
sluit niet alle relevante schade-vormen in (bv. alarmmoeheid op
afdelingsniveau).

**Relevante feiten in deze case:**
- Data komt uit twee Amerikaanse ziekenhuizen, niet uit Isala.
- Het model bevat mogelijk indirecte bias door informatieve missingness
  (proxy voor arts-attentie).
- Recall- én utility-doelen kunnen tegelijkertijd *niet* worden gehaald
  op deze data.

**Relevante normen.**
- Beauchamp & Childress (zorg-ethiek).
- AVG / MDR / WGBO.
- Hippocratisch *primum non nocere* — niet-schaden.
- HLEG-richtsnoer ‘betrouwbare AI’: menselijke controle, robustheid,
  transparantie, fairness.

**Stakeholders:** zie hoofdstuk 2 van deze toelichting.

### Stap 3 — Stel handelingsmogelijkheden vast

We onderscheiden vier inhoudelijk verschillende opties.

| Optie | Beschrijving |
|---|---|
| **1** | Implementeer drempel 0,59 (recall 0,58 / utility −0,21) — focus op detecteren |
| **2** | Implementeer drempel 0,85 (recall 0,20 / utility +0,04) — focus op netto goed |
| **3** | Gebruik beide drempels gelaagd: ‘oranje’ bij 0,59, ‘rood’ bij 0,85, met verschillende workflow |
| **4** | Implementeer geen alarmsysteem; lever het model als beslissings-ondersteunende risicoscore in het PDMS, geen geautomatiseerde drempel |

> *Niet-optie:* de keuze ‘publiceer een AUROC van 0,83 en laat het
> ziekenhuis maar uitzoeken hoe ze het inzetten’ is geen morele optie —
> dat verschuift de verantwoordelijkheid zonder de afweging te maken.

### Stap 4 — Beoordeel elke optie

We hanteren drie complementaire kaders.

#### 4a. Utilitair perspectief
Welke optie maximaliseert het saldo van goed over kwaad?

- Optie 1: + veel terecht gevangen sepsis-patiënten; − systematisch
  méér valse alarmen dan baat van vroege detectie (utility −0,21).
- Optie 2: + iets meer baat dan inaction; − 80 % van de sepsis-patiënten
  ongedetecteerd. Goed-per-patient is positief maar marginaal.
- Optie 3: − operationeel ingewikkeld, twee classes alarmen geven nieuwe
  vorm van alarmmoeheid; ± nut afhankelijk van workflow-discipline.
- Optie 4: + verschuift de afweging naar de arts (mens-in-de-lus);
  − verspilt de signaal-toegevoegde-waarde van het model deels.

Utilitair gewogen lijken opties 2 en 4 het minst slecht; optie 1 vereist
heel sterke ‘zachte’ baten (vroege detectie redt méér levens dan de
utility-score telt) die we niet kunnen aantonen.

#### 4b. Deontologisch perspectief
Zijn er plichten of rechten die voorrang krijgen?

- Plicht tot niet-schaden (*primum non nocere*) — pleit tegen optie 1.
- Plicht tot weldoen — pleit tegen optie 4 als die optie kennis ongebruikt
  laat (wij hebben een 17,5 uur lead time, dat is informatie die we
  bekend kunnen maken zonder direct te alarmeren).
- Recht van de patiënt op informatie over AI-inzet — vereist een
  uitleg-laag in elke optie.
- Plicht tot verantwoording — verbiedt opties zonder validatie of
  zonder ethische commissie.

Deontologisch valt optie 1 af en wint optie 4 (geen geautomatiseerd
alarm zonder bewijs dat het niet-schaadt).

#### 4c. Deugd-ethisch perspectief
Wat zou een verstandige, voorzichtige arts-onderzoeker doen?

- Voorzichtigheid: niet inzetten wat we niet hebben gevalideerd.
- Eerlijkheid: het ziekenhuis transparant informeren over de
  utility-paradox in plaats van een ‘sterke AUROC’ te verkopen.
- Bescheidenheid: erkennen dat we op data van twee Amerikaanse
  ziekenhuizen werken en dat we niet kunnen claimen wat we niet weten.

Vanuit deugd-ethiek wint optie 4 (met de uitgebreide informatieve route),
gevolgd door optie 3 alleen na lokale validatie.

### Stap 5 — Maak een afweging

Drie kaders wijzen in dezelfde richting met verschillende argumentaties:
**optie 4 voor de korte termijn**, met **optie 3 na lokale validatie als
mogelijk eindstation**. Daar passen wij twee aanvullende condities aan toe:

- **Conditie A:** voor élk gebruik moet de model-output binnen het PDMS
  als ‘ondersteunend, niet sturend’ worden gepresenteerd, met
  SHAP-uitleg en historische context per patiënt.
- **Conditie B:** de prospectieve studie naar utility op Isala-data
  rapporteert *zowel* het utility-getal *als* de absolute aantallen
  vermeden vs. veroorzaakte schade. De bestuurslaag van Isala maakt de
  keuze voor drempel 0,59 of 0,85, op basis van die getallen, met
  patiëntenraad-consultatie.

### Stap 6 — Conclusie en aanbeveling

**Wij adviseren Isala om het LightGBM-model in deze fase niet als
alarmsysteem in te zetten** (zoals optie 1 of 2), maar als
beslissings-ondersteunende risicoscore in het PDMS, zonder geautomatiseerde
drempel (optie 4). Na een succesvolle stille proefperiode en lokale
fine-tuning kan het ziekenhuis een drempelkeuze maken (optie 3), als
**bestuursbeslissing met patiëntenraad-input**, niet als data-science
beslissing.

Wij verbinden dit advies aan onszelf: onze rapportage benoemt de
utility-paradox in de samenvatting (niet in een voetnoot), bevat *beide*
drempels, en raadt commerciële of klinische adoptie expliciet af zonder
de hierboven genoemde voorwaarden.

---

### Reflectie op het stappenplan

Het stappenplan dwong ons om:

- onze keuze niet te baseren op alleen de utilitair-beste optie (die te
  smal was: utility +0,04 is bijna inaction),
- de deontologische plicht van niet-schaden serieus te nemen ook al
  kostte het sensitiviteit,
- en in te zien dat de drempelkeuze geen *data-science* keuze is maar een
  bestuurskeuze die we ondersteunen, niet maken.

De grootste leeropbrengst: een AUROC van 0,83 is geen vrijbrief.
Klinische bruikbaarheid en ethische verantwoordbaarheid worden door
geheel andere metrieken bepaald.

---

### Referenties
- Beauchamp, T. L., & Childress, J. F. (2013). *Principles of Biomedical Ethics* (7th ed.). Oxford University Press.
- High-Level Expert Group on AI (2019). *Ethics Guidelines for Trustworthy AI*. Europese Commissie.
- Kumar, A. et al. (2006). Duration of hypotension before initiation of effective antimicrobial therapy is the critical determinant of survival in human septic shock. *Critical Care Medicine*, 34(6), 1589–1596.
- Mehta, H. B., Li, S., Goodwin, J. S. (2018). Gender disparities in sepsis prevention strategies. *American Journal of Critical Care*, 27(1).
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. *FAT* '19*.
- Reyna, M. A. et al. (2020). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019. *Critical Care Medicine*, 48(2).
- Van de Poel, I., & Royakkers, L. (2011). *Ethics, Technology, and Engineering*. Wiley-Blackwell.
