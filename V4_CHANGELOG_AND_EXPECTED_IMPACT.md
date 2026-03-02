# DnCNN Lite ECA v4 - Changelog And Expected Impact

## Scope

Αυτό το changelog περιγράφει τι άλλαξε από `v3` σε `v4` στο:
- `C:\ESPI_DnCNN\espi_dncnn_lite_eca_FULL_PATCH_v4.py`

και γιατί οι αλλαγές αναμένεται να βελτιώσουν τη συμπεριφορά.

---

## 1. Νέα δυνατότητα: καθαρό no-ECA baseline

### Τι άλλαξε
- Προστέθηκαν CLI flags:
  - `--use-eca`
  - `--no-eca`
- Προστέθηκε `use_eca` στο config (`DnCNNLiteECAConfig`) και στο runtime args.
- Στο model construction, όταν `use_eca=False`, όλα τα attention blocks γίνονται identity.

### Γιατί
- Δίνει πραγματικό A/B χωρίς να αλλάζεις script/αρχιτεκτονική οικογένεια.

### Τι περιμένουμε να βελτιώσει
- Δίκαιη σύγκριση ECA vs no-ECA.
- Λιγότερα "artifact" αποτελέσματα από διαφορετικά scripts.

---

## 2. Νέα δυνατότητα: ελεγχόμενο GroupNorm

### Τι άλλαξε
- Προστέθηκε `--gn-groups`.
- Νέα helper `make_norm_with_groups(...)`:
  - `gn_groups=0` -> auto mode (8/4/2/1)
  - `gn_groups>0` -> επιλέγει το μεγαλύτερο έγκυρο divisor.
- Το `ConvBlock` πλέον παίρνει `gn_groups`.

### Γιατί
- Το GN επηρεάζει ισχυρά τη δυναμική attention/activation.
- Θέλουμε ελεγχόμενο sweep αντί hard-coded behavior.

### Τι περιμένουμε να βελτιώσει
- Πιο σταθερή εκπαίδευση.
- Πιθανή βελτίωση σε SSIM/EdgeF1 σε δύσκολα δείγματα.

---

## 3. Non-finite guards σε training loop

### Τι άλλαξε
- `run_epoch_train(...)` ξαναγράφτηκε ώστε:
  - να ελέγχει finite input/loss/gradients,
  - να παραλείπει bad batches (`nan_action=skip`) ή να σταματά (`nan_action=stop`),
  - να μετρά non-finite περιστατικά,
  - να επιστρέφει structured stats.
- Προστέθηκαν args:
  - `--nan-action {skip,stop}`
  - `--max-nonfinite-batches`
  - `--grad-clip`
  - `--log-grad-norm`

### Γιατί
- Στα παλιά runs είχες NaN collapse από epoch 2.
- Χωρίς guards, γράφονται παραπλανητικά logs/checkpoints.

### Τι περιμένουμε να βελτιώσει
- Σταθερότητα και αξιοπιστία run.
- Άμεσο fail-fast όταν κάτι πάει στραβά.

---

## 4. Non-finite guards σε validation/real eval

### Τι άλλαξε
- `run_validation_fullres(...)` τώρα:
  - ελέγχει finite σε inputs/loss/metrics,
  - παρακολουθεί `samples_seen/samples_used/nonfinite_batches`,
  - επιστρέφει dict metrics + counters.
- `run_real_evaluation(...)` τώρα:
  - κάνει skip non-finite samples,
  - επιστρέφει `used`/`skipped` counts.

### Γιατί
- Η validation είναι source of truth για model selection.
- Πρέπει να ξέρεις αν το metric βγήκε από πλήρη ή “σπασμένο” validation.

### Τι περιμένουμε να βελτιώσει
- Καθαρότερη επιλογή best checkpoint.
- Better diagnosis όταν υπάρχουν outliers/instabilities.

---

## 5. Ασφαλέστερο checkpoint loading

### Τι άλλαξε
- Προστέθηκαν:
  - `--resume-strict` (default)
  - `--resume-nonstrict`
- Το resume πλέον φορτώνει strict by default.
- Σε strict mismatch γίνεται hard error (αντί να προχωράει σιωπηλά).

### Γιατί
- Τα non-strict φορτώματα συχνά κρύβουν architecture mismatch.
- Προκαλούν “ύπουλα” λάθη στην αξιολόγηση.

### Τι περιμένουμε να βελτιώσει
- Συνεπές resume behavior.
- Λιγότερα ψευδο-αποτελέσματα από μισο-ταιριασμένα checkpoints.

---

## 6. Logging/observability upgrade

### Τι άλλαξε
- Νέες στήλες στο `train_log.csv`:
  - `train_nonfinite`, `val_nonfinite`
  - `train_batches`, `val_samples`
  - `grad_norm`
- Νέα TensorBoard scalars:
  - `debug/train_nonfinite`
  - `debug/val_nonfinite`
  - `opt/grad_norm_mean`

### Γιατί
- Το v3 log δεν αρκούσε για να εξηγήσεις γιατί "χάλασε" ένα run.

### Τι περιμένουμε να βελτιώσει
- Γρήγορο root-cause analysis.
- Πιο αξιόπιστο experiment tracking.

---

## 7. Default behavior αλλαγές

### Τι άλλαξε
- `--freeze-norm-epoch` default από `3` σε `0`.

### Γιατί
- Για fair ablation θέλουμε πρώτα χωρίς early freeze.
- Το πρόωρο freeze μπορεί να εμποδίσει adaptation.

### Τι περιμένουμε να βελτιώσει
- Πιο καθαρό συμπέρασμα για ECA/GN επίδραση.

---

## 8. Checkpoint selection hardening

### Τι άλλαξε
- `best`/`best_ssim` updates γίνονται μόνο όταν validation metrics είναι finite.

### Γιατί
- Αποτρέπει "best model" από corrupted validation στιγμές.

### Τι περιμένουμε να βελτιώσει
- Πιο έγκυρα τελικά checkpoints.

---

## 9. Expected impact summary

| Αλλαγή | Κύριο κέρδος | Αναμενόμενο αποτέλεσμα |
|---|---|---|
| `--no-eca` / `--use-eca` | Fair ablation | καθαρή σύγκριση ECA vs baseline |
| `--gn-groups` | Norm control | καλύτερο tuning σε SSIM/EdgeF1 |
| Non-finite guards | Stability | λιγότερα NaN collapse runs |
| Strict resume | Correctness | λιγότερα checkpoint mismatch errors |
| Expanded logs | Observability | ταχύτερο debugging και reproducibility |
| Default no freeze | Fairness | πιο τίμιο πρώτο A/B αποτέλεσμα |

---

## 10. Ρεαλιστικές προσδοκίες από v4

Η v4 είναι κυρίως reliability/fairness upgrade, όχι "μαγικό" architecture jump.

Ρεαλιστικά (ECA - noECA) μετά από σωστό tuning:
- `ΔPSNR`: περίπου `-0.10` έως `+0.30` dB
- `ΔSSIM`: περίπου `-0.01` έως `+0.03`
- `ΔEdgeF1`: περίπου `+0.005` έως `+0.03`

Το μεγαλύτερο κέρδος της v4:
- ότι τα αποτελέσματα αυτά θα είναι πιο αξιόπιστα και αναπαραγώγιμα.
