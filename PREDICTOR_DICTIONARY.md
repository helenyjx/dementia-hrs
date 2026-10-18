# Predictor Dictionary

Definitions, sources, and coding of the 60 baseline predictors (94 model inputs). Source labels are quoted from the RAND HRS Longitudinal File 2020 (V2) codebook (`randhrs1992_2020v2`); race/ethnicity is from the HRS Tracker 2022 (`trk2022tr_r`); "don't do" answers are from the HRS 2000 and 2010 core interviews. `w` in a RAND name is the wave: 5 = 2000 (development baseline), 10 = 2010 (external baseline).

Coding is produced by `preprocess_2000.ipynb` / `preprocess_2002-2016.ipynb` and the data-management notebooks. "Missing (n)" is the number of missing values in the 2000 development cohort (n = 17,336, 6-year file). Missing values are imputed in training data and excluded from validation and test sets (complete cases).

**Missing values.** RAND special missing codes (e.g. `.D` don't know, `.R` refused, `.M` other missing, `.X` "don't do") are read as missing. The one exception is the "don't do" answer for 12 functional items, which is kept as its own category (see Functional limitations).

**Variable names.** The data-management notebooks strip `_`, `.`, and `0` from column names, so `edstg_2.8-11` becomes `edstg2 8to11` and `mapa_dontdo` becomes `mapadontdo`. The names below are the model-input names.

## Demographic and family structure

| Variable | Source (2000 / 2010) | RAND label | Definition and coding | Missing (n) |
|---|---|---|---|---|
| `ageym` | R5AGEY_M / R10AGEY_M | "R Age (years) at Ivw MidMon" | Age in completed years at the interview month (mid-month). Continuous, years; 51–100 in the cohort (eligibility: > 50). Not "years and months" | 0 |
| `gender` | RAGENDER | "R Gender" | 0 = male, 1 = female | 0 |
| `child` | H5CHILD / H10CHILD | "Number of living children R/P" | Number of living children of the respondent and spouse/partner (household-level). Count, 0–11 | 137 |
| `livpar` | R5LIVPAR / R10LIVPAR | "Number of living parents" | Number of the respondent's parents alive, 0–2. Not co-residence | 290 |
| `livsib` | R5LIVSIB / R10LIVSIB | "Number of living siblings" | Number of living siblings, count. Not co-residence | 55 |
| `momage` | R5MOMAGE / R10MOMAGE | "Mother age current/at death" | Mother's current age, or age at death if deceased. Years | 548 |
| `dadage` | R5DADAGE / R10DADAGE | "Father age current/at death" | Father's current age, or age at death if deceased. Years | 708 |

## Socioeconomic and geographic

| Variable | Source | RAND label | Definition and coding | Missing (n) |
|---|---|---|---|---|
| `edstg2 8to11`, `edstg3 12`, `edstg4 13above` | RAEDYRS | "R Years of Education" | Years of education grouped 0–7 (reference), 8–11, 12, 13+ | 0 |
| `lbrf2` … `lbrf7` | R5LBRF / R10LBRF | "R Labor Force Status" | Nominal, one-hot; reference = 1 works full-time. Dummies: 2 works part-time, 3 unemployed, 4 partly retired, 5 retired, 6 disabled, 7 not in labor force | 0 |
| `mstat2` … `mstat8` | R5MSTAT / R10MSTAT | "R Marital Status" | Nominal, one-hot; reference = 1 married. Dummies: 2 married, spouse absent; 3 partnered; 4 separated; 5 divorced; 7 widowed; 8 never married. Code 6 (separated/divorced, unspecified) is set to missing | 18 |
| `cendiv2` … `cendiv11` | R5CENDIV / R10CENDIV | "Census Division" | Nominal, one-hot; reference = 1 New England. Dummies: 2 Mid Atlantic, 3 East North Central, 4 West North Central, 5 South Atlantic, 6 East South Central, 7 West South Central, 8 Mountain, 9 Pacific, 11 not in US / US territory | 12 |
| `raceeth1`, `raceeth2`, `raceeth3` | Tracker RACE, HISPANIC | race/ethnicity; hispanicity type | Nominal, one-hot; reference = **non-Hispanic White**. `raceeth1` = non-Hispanic Black, `raceeth2` = Hispanic (any race), `raceeth3` = **non-Hispanic other**. Unknown race or unknown Hispanic status = missing | 4 |

## Self-reported health and behaviors

| Variable | Source | RAND label | Definition and coding | Missing (n) |
|---|---|---|---|---|
| `shlt` | R5SHLT / R10SHLT | "Self-report of health" | Ordinal: 0 excellent, 1 very good, 2 good, 3 fair, 4 poor | 7 |
| `slfmem` | R5SLFMEM / R10SLFMEM | "SELF RATED MEMORY" | Ordinal: 0 excellent, 1 very good, 2 good, 3 fair, 4 poor. Not asked in proxy interviews | 1,281 |
| `height` | R5HEIGHT / R10HEIGHT | "Self-reported height in meters" | Meters, self-reported | 15 |
| `weight` | R5WEIGHT / R10WEIGHT | "Self-reported weight in kilograms" | Kilograms, self-reported | 216 |
| `smokev` | R5SMOKEV / R10SMOKEV | "R smoke ever" | 0/1 | 154 |
| `smoken` | R5SMOKEN / R10SMOKEN | "R smokes now" | 0/1 | 1 |
| `drink` | R5DRINK / R10DRINK | "R ever drinks any alcohol" | 0/1. Not a frequency | 2 |
| `vigact` | 2000: R5VIGACT; 2010: derived from R10VGACTX | "R Wtr vigorus phys act 3+/wk"; "R Freq vigorous phys activ {finer scale}" | 0/1 vigorous physical activity. 2000: 3+ times/week. 2010: "every day" or ">1 per week" = 1 (closest available match, slightly broader) | 5 |
| `depres` | R5DEPRES / R10DEPRES | "CESD: Felt depressed" | 0/1, single CES-D item (past week). Not a composite score | 1,292 |
| `effort` | R5EFFORT / R10EFFORT | "CESD: Everything an effort" | 0/1, CES-D item | 1,295 |
| `sleepr` | R5SLEEPR / R10SLEEPR | "CESD: Sleep was restless" | 0/1, CES-D item | 1,291 |
| `whappy` | R5WHAPPY / R10WHAPPY | "CESD: Was happy" | 0/1, CES-D item (positively worded) | 1,299 |
| `flone` | R5FLONE / R10FLONE | "CESD: Felt lonely" | 0/1, CES-D item | 1,293 |
| `fsad` | R5FSAD / R10FSAD | "CESD: Felt sad" | 0/1, CES-D item | 1,295 |
| `going` | R5GOING / R10GOING | "CESD: Could not get going" | 0/1, CES-D item. Not "difficulty going out" | 1,293 |
| `enlife` | R5ENLIFE / R10ENLIFE | "CESD: Enjoyed life" | 0/1, CES-D item (positively worded) | 1,299 |

The CES-D items are not asked in proxy interviews, which accounts for most of their missing values.

## Chronic conditions (doctor-diagnosed, ever)

| Variable | Source | RAND label | Coding | Missing (n) |
|---|---|---|---|---|
| `hibpe` | R5HIBPE / R10HIBPE | "R ever had high blood pressure" | 0/1 | 0 |
| `diabe` | R5DIABE / R10DIABE | "R ever had diabetes" | 0/1 | 0 |
| `cancre` | R5CANCRE / R10CANCRE | "R ever had cancer" | 0/1 | 0 |
| `lunge` | R5LUNGE / R10LUNGE | "R ever had lung disease" | 0/1 | 0 |
| `hearte` | R5HEARTE / R10HEARTE | "R ever had heart problems" | 0/1 | 0 |
| `stroke` | R5STROKE / R10STROKE | "R ever had stroke" | 0/1 | 0 |
| `arthre` | R5ARTHRE / R10ARTHRE | "R ever had arthritis" | 0/1 | 0 |

## Health-care utilization and interview type

| Variable | Source | RAND label | Definition and coding | Missing (n) |
|---|---|---|---|---|
| `hosp` | R5HOSP / R10HOSP | "Hospital stay, prv 2 yrs" | 0/1, any overnight hospital stay in the previous 2 years | 29 |
| `nrshom` | R5NRSHOM / R10NRSHOM | "Nurs home stay, prv 2 yrs" | 0/1, any overnight nursing-home stay in the previous 2 years. Not current residence (that is RwNHMLIV) | 18 |
| `nrstim` | R5NRSTIM / R10NRSTIM | "# Nurs home stays, prv 2 yrs" | Number of nursing-home stays, previous 2 years (0 if none). Not nights | 28 |
| `nrsnit` | R5NRSNIT / R10NRSNIT | "# Nights in nurs home, prv 2 yrs" | Number of nights in a nursing home over all stays, previous 2 years (0 if none). Not hospital nights | 34 |
| `doctim` | R5DOCTIM / R10DOCTIM | "# Doctor vists, prv 2 yrs" | Number of doctor visits, previous 2 years | 370 |
| `hlthlm` | R5HLTHLM / R10HLTHLM | "Hlth problems limit work" | 0/1 | 108 |
| `proxy` | R5PROXY / R10PROXY | "Whether Proxy Interview" | 0/1, interview completed by a proxy respondent | 0 |

## Functional limitations

Items are RAND "any difficulty" indicators (0 = no, 1 = any difficulty, including "can't do"). For the 12 items marked †, the HRS core answer "don't do" (which RAND sets to missing) is kept as a third category: `<item> = 0` and `<item>dontdo = 1`. `<item>dontdo = 0` when the item was answered; both are missing when the item is missing for another reason. Core-interview sources: 2000 module E (`H00E_R`), 2010 module G (`H10G_R`).

| Variable | Source | RAND label | Definition | Core item 2000 / 2010 | Missing (n) |
|---|---|---|---|---|---|
| `walksa` † | RwWALKSA | "R Any Diff-Walk sev blocks" | Walking several blocks | G2689 / MG001 | 24 |
| `walk1a` † | RwWALK1A | "R Any Diff-Walk one block" | Walking one block | G2692 / MG003 | 24 |
| `sita` † | RwSITA | "R Any Diff-Sit for 2 hours" | Sitting for about 2 hours | G2695 / MG004 | 26 |
| `chaira` † | RwCHAIRA | "R Any Diff-Get up fr chair" | Getting up from a chair after sitting a long time | G2698 / MG005 | 26 |
| `climsa` † | RwCLIMSA | "R Any Diff-Clmb sev flt str" | Climbing several flights of stairs without resting | G2701 / MG006 | 41 |
| `clim1a` † | RwCLIM1A | "R Any Diff-Clmb 1 flt stair" | Climbing one flight of stairs without resting | G2704 / MG007 | 34 |
| `stoopa` † | RwSTOOPA | "R Any Diff-Stoop/Kneel/Crch" | Stooping, kneeling, or crouching | G2707 / MG008 | 26 |
| `armsa` † | RwARMSA | "R Any Diff-Rch/xtnd arms up" | Reaching or extending arms above shoulder level | G2710 / MG009 | 29 |
| `pusha` † | RwPUSHA | "R Any Diff-Push/pull lg obj" | Pulling or pushing large objects | G2713 / MG010 | 30 |
| `lifta` † | RwLIFTA | "R Any Diff-Lift/carry 10lbs" | Lifting or carrying weights over 10 pounds | G2716 / MG011 | 27 |
| `dimea` † | RwDIMEA | "R Any Diff-Pick up a dime" | Picking up a dime from a table | G2719 / MG012 | 29 |
| `mapa` † | RwMAPA | "R Any Diff-Use a map" | **Using a map** to get around in a strange place (an instrumental activity of daily living). **Not use of a mobility aid** | G2851 / MG040 | 31 |
| `toilta` | RwTOILTA | "R Any Diff-Using the toilet" | Using the toilet (ADL). "Don't do" not kept: only 6 cases, and RAND assigns some of them a value | — | 26 |
| `adl5a` | RwADL5A | "Any Diff-sum of ADLs /0-5" | Count of ADLs with any difficulty, 0–5: bathing, dressing, eating, getting in/out of bed, walking across a room | — | 19 |
| `mobila` | RwMOBILA | "Any Diff-sum of Mobility measures /0-5" | Count, 0–5: walk several blocks, walk one block, walk across a room, climb several flights, climb one flight | — | 27 |
| `lgmusa` | RwLGMUSA | "Any Diff-sum of Large Muscle measures /0-4" | Count, 0–4: sit 2 hours, get up from chair, stoop/kneel/crouch, push/pull large objects | — | 24 |
| `grossa` | RwGROSSA | "Walk1/R,Clim1,Bed,Bath/0-5" | Count, 0–5: walk one block, walk across a room, climb one flight, get in/out of bed, bathe | — | 19 |
| `finea` | RwFINEA | "Dime/Eat/Dress /0-3" | Count, 0–3: pick up a dime, eat, dress | — | 19 |

## Not predictors (excluded to prevent leakage)

The outcome components are excluded from the predictor set: self-respondent TICS-m items `imrc`, `dlrc`, `ser7`, `bwc20`; proxy items `prmem`, `iadl5a`, `prfin`. Baseline `demcls` is used only for eligibility (must be 0).

## Counts

60 variables: 5 continuous (`ageym`, `height`, `weight`, `momage`, `dadage`), 11 counts, 2 ordinal (`shlt`, `slfmem`), 5 nominal (27 dummies), and 37 binary, of which 12 functional items have an extra "don't do" dummy. Total: 5 + 11 + 2 + 27 + 37 + 12 = **94 model inputs**.
