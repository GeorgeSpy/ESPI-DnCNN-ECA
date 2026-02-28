# Αναφορά Σύγκρισης Εκδόσεων DnCNN-ECA

Αυτή η αναφορά συγκεντρώνει την αρχιτεκτονική εξέλιξη των διαφορετικών εκδόσεων του μοντέλου **DnCNN-Lite ECA** (Από τις αρχικές CPU-safe εκδόσεις έως την ολοκληρωμένη V5), καθώς και τις καταγεγραμμένες επιδόσεις τους τόσο σε μετρικές αποθορυβοποίησης (Denoising) όσο και σε downstream ταξινόμηση (Classification).

---

## 1. Αρχιτεκτονική Εξέλιξη & Χαρακτηριστικά (Features)

| Έκδοση Αρχείου | Τύπος "Attention" (ECA) | Μέθοδος Pooling | Spatial Attention (Χωρική Προσοχή) | Διαχείριση Config | Κύρια Χαρακτηριστικά & Καινοτομίες |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`FIXED.py`**<br>*(V1)* | Απλό ECA<br>*(Σταθερό Kernel)* | Global Average Pooling (GAP) | Όχι | Hardcoded | Απλή προσθήκη `k_size=3` ECA module μετά τις συνελίξεις. Βασική CPU-safe υλοποίηση. |
| **`PATCHED.py`**<br>**`v2.py`** | Απλό ECA<br>*(Σταθερό Kernel)* | GAP | **Ναι**<br>`SpatialLiteAttention` | Hardcoded | Εισαγωγή του **SpatialLiteAttention** (Χωρική Προσοχή 2D), για να μαθαίνει όχι μόνο *ποια* κανάλια είναι σημαντικά αλλά και *πού* στον χώρο να εστιάσει. |
| **`..._v3.py`**<br>*(FIXED_PATCHED)* | Squeeze & Excitation (SE) | GAP | Ναι | Hardcoded | Πειραματική έκδοση. Το ECA αντικαταστάθηκε προσωρινά με **SE Block** (με παράμετρο `reduction=16` και Linear layers). |
| **`FULL_PATCH_v3.py`**| Προηγμένο ECA | GAP | Ναι | Μέσω Κλάσης | **Μεγάλο Refactoring**. Εισαγωγή της κλάσης `DnCNNLiteECAConfig`. Το ECA αποκτά παραμέτρους θερμοκρασίας (`temp=0.75`) και κέρδους (`gain=0.5`). Υποστήριξη Mixed Precision. |
| **`FULL_PATCH_v4.py`**| Προηγμένο ECA | GAP | Ναι | Μέσω Κλάσης | **Η πιο Σταθερή Έκδοση**. Βελτιωμένος κώδικας, διορθωμένο I/O. Η έκδοση που χρησιμοποιήθηκε ευρέως για όλα τα στάνταρ πειράματα της διατριβής. |
| **`FULL_PATCH_v5.py`**| **The Ultimate ECA** | **Dual Pooling**<br>(Max + Avg) | Ναι | Μέσω Κλάσης | 🚀 **Ερευνητική Κορυφή:** <br>1. **Dual Pooling**: Συνδυάζει Max & Avg pooling.<br>2. **Learnable Temp/Gain**: Προσαρμόζονται δυναμικά κατά το training.<br>3. **Multi-scale Kernels**.<br>4. **Placement Presets** (`eca_order`). |

---

## 2. Πίνακας Επιδόσεων Αποθορυβοποίησης (Denoising Metrics)

Οι παρακάτω τιμές προέρχονται από συγκριτικά πειράματα αξιολόγησης σε **συνθετικά** (Synthetic Validation) και **πραγματικά δεδομένα** (Real ESPI pairs/averages).

| Μοντέλο | Εκπαίδευση Σε | Val PSNR (Συνθετικά) | Val SSIM | Val EdgeF1 | Real PSNR (Πραγματικά) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V4 Baseline (NoECA)** | PseudoNoisy (Synthetic) | 27.24 dB | 0.7846 | 0.7686 | 34.58 dB |
| **V4 ECA** | PseudoNoisy (Synthetic) | **27.47 dB** | **0.7972** | **0.7787** | 34.50 dB |
| **V5 Baseline (NoECA)** | PseudoNoisy (Synthetic) | 27.24 dB | 0.7846 | 0.7686 | **34.58 dB** |
| **V5 ECA (Advanced)** | PseudoNoisy (Synthetic) | 27.24 dB | 0.7852 | 0.7712 | 34.22 dB |
| **V4r Baseline (NoECA)** | **Real Pairs (23,891 imgs)**| *N/A* | *N/A* | *N/A* | 23.76 dB *(Real Val)* |
| **V4r ECA** | **Real Pairs (23,891 imgs)**| *N/A* | *N/A* | *N/A* | **23.85 dB** *(Real Val)* |

### 🔍 Σημαντικές Παρατηρήσεις στα Denoising Metrics:
- To **V4 ECA** παρουσιάζει ξεκάθαρη νίκη στα **συνθετικά δεδομένα** (+0.23 dB PSNR και +0.0126 SSIM σε σχέση με το Baseline).
- Το **V5 ECA**, αν και πιο προηγμένο αρχιτεκτονικά (Learnable params, Dual pooling), παρουσιάζει στασιμότητα και **δεν** ξεπερνά το απλό V4 ECA στα συγκεκριμένα πειράματα. 
- Στα **πραγματικά (Real)** δεδομένα, όταν το μοντέλο εκπαιδεύεται εξολοκλήρου σε πραγματικά Noise-Clean pairs (V4r), η αρχιτεκτονική **ECA** προσφέρει μικρή αλλά σταθερή βελτίωση στην αποθορυβοποίηση (+0.09 dB στο test set πραγματικών).

---

## 3. Επιδόσεις Ταξινόμησης (Downstream Classification Task)

Πώς επηρεάζει η χρήση του εκάστοτε Denoiser το τελικό στάδιο ταξινόμησης ατελειών (ResNet18 5-class Classifier);

| Pre-processing Pipeline | Denoiser Training Data | ECA Ενεργό; | Classification Accuracy (%) | Classification Macro F1 (%) | Δ Acc vs Raw |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Καθόλου Denoising (Raw)** | — | — | 97.70 % | 93.99 % | — |
| **V4 Denoised** | PseudoNoisy (243 imgs) | Όχι | 96.39 % | 89.06 % | −1.31 % |
| **V4 Denoised** | PseudoNoisy (243 imgs) | Ναι | 94.77 % | 84.21 % | −2.93 % |
| **V4r Denoised (Real-Trained)** | **Real Pairs (23,891 imgs)**| Όχι | 98.76 % | 96.07 % | +1.06 % |
| **V4r Denoised (Real-Trained)** | **Real Pairs (23,891 imgs)**| **Ναι (V4 ECA)** | **98.87 %** | **96.64 %** | **+1.17 %** |

### 🚀 Σημαντικές Παρατηρήσεις στο Classification:
1. **Η ποιότητα των Δεδομένων κυριαρχεί αρχιτεκτονικής:** Το Denoising μοντέλο που εκπαιδεύτηκε σε συνθετικά δεδομένα (PseudoNoisy) *έριξε* τις επιδόσεις του Classification. Αντίθετα, η εκπαίδευση σε Πραγματικά Ζεύγη (V4r) *βελτίωσε* σημαντικά την ταξινόμηση.
2. Το **V4 ECA** όταν εκπαιδεύεται σε *Πραγματικά (Real)* δεδομένα (V4r ECA), καταγράφει **τις καλύτερες συνολικές επιδόσεις σε ολόκληρο το σύστημα**, φτάνοντας Accuracy **98.87%** (μια σημαντική αύξηση +1.17% έναντι της μη χρήσης φίλτρου) και Macro F1 **96.64%**.
