# Trees

Repository per generare codice P4/Tofino dai modelli XGBoost e CatBoost e per creare i plot usati nel paper.

## File principali

- `generate_all.py`: genera tutte le cartelle in `generated/` partendo dai JSON in `jsons/`.
- `xgboost_to_tofino.py`: converte un modello XGBoost in output Tofino.
- `catboost_to_tofino.py`: converte un modello CatBoost in output Tofino, riusando la logica comune XGBoost.
- `plot_all.py`: unico script per rigenerare tutti i plot.
- `compilation_summary.csv`: dati di compilazione usati dal plot TCAM.
- `plots/`: output dei plot in PDF e PNG.

## Generare codice P4

```bash
python3 generate_all.py
```

Output: cartelle `generated/D*_XGBOOST/` e `generated/D*_CATBOOST/` con `.p4`, `includes/`, `build.sh` e `bfrt_shell.py`.

Opzioni utili:

```bash
python3 generate_all.py --merge-strategy greedy
python3 generate_all.py --merge-strategy optimal
python3 generate_all.py --light
```

## Generare i plot

Se usi la virtualenv locale:

```bash
.venv/bin/python plot_all.py
```

Oppure, con un ambiente Python gia' attivato:

```bash
python3 plot_all.py
```

Output:

- `plots/regions.pdf` e `.png`
- `plots/intersections_vs_final.pdf` e `.png`
- `plots/merge.pdf` e `.png`
- `plots/tcam.pdf` e `.png`

Opzioni utili:

```bash
python3 plot_all.py --depths 4 8 12 16
python3 plot_all.py --plots-dir plots
python3 plot_all.py --csv compilation_summary.csv
```

## Legacy

Gli script plot separati sono stati consolidati in `plot_all.py`. I file sotto `.history/` sono snapshot storici e non fanno parte del flusso attivo.
